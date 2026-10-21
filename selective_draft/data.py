"""GSM8K with explicit sample IDs: split manifest, one prompt recipe shared by collection,
training, diagnostics and evaluation, and the legacy answer parser."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .config import Config, DataConfig

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

# Verbatim copy of eval/math500.py::MATH500_SYSTEM_PROMPT (including its stray quote) so that the
# base accuracy is comparable with the legacy evaluation.
MATH_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""

# KLASS (github.com/shkim0116/KLASS @ad87ecb, src/llada_evaluation.py::test_dataset): separate system message.
KLASS_SYSTEM_PROMPT = ("Your task is to answer the question below. Give step by step reasoning before you answer, "
                       "and when you're ready to answer, please use the format 'The final answer is'.")
# DMax (github.com/czg1225/DMax @43fafda, dInfer/evaluations/tasks/math/minerva_math_algebra.yaml doc_to_text, used by
# minerva_math500 with --apply_chat_template): user turn only.
DMAX_BOXED = "\nLet's think step by step, and put your final answer within \\boxed{}."
# Original LLaDA OpenCompass MATH 0-shot (ML-GSAI/LLaDA @9182493, math_0shot_gen_11c4b5.py): appended to the problem.
BOXED_STEP_BY_STEP = "\nPlease reason step by step, and put your final answer within \\boxed{}."

RECIPES = {
    # (instruction, text appended after the generation prompt); legacy recipes put the instruction in the user turn
    "legacy_boxed_reasoning": (GSM_SYSTEM_PROMPT, "<reasoning>"),
    "legacy_math_boxed_reasoning": (MATH_SYSTEM_PROMPT, "<reasoning>"),
    "klass_final_answer": (KLASS_SYSTEM_PROMPT, ""),  # system message, then the bare problem
    "boxed_step_by_step": (BOXED_STEP_BY_STEP, ""),  # problem + instruction in the user turn, no system message
    "dmax_boxed": (DMAX_BOXED, ""),  # same layout, DMax's wording
    "fake": ("Solve.", ""),
}

FAKE_PROBLEMS: List[Tuple[str, str, str]] = [
    (f"fake-{i}", f"What is {a} plus {b}?", f"{a} + {b} = {a + b}\n#### {a + b}")
    for i, (a, b) in enumerate([(2, 3), (5, 7), (10, 4), (1, 1), (8, 9), (6, 6), (3, 14), (20, 1),
                                (7, 2), (9, 9), (11, 3), (4, 4)])
]


@dataclass
class Example:
    example_id: str
    source_split: str
    question: str
    answer_text: str
    answer_value: object  # float for gsm8k/fake, LaTeX string for math
    prompt_text: str
    task: str = "gsm8k"


_PARSER_HELPER = None


def _parser_helper():
    """eval/parser_helper.py (pure re/json) loaded by path: last_boxed_only_string, remove_boxed, is_equiv."""
    global _PARSER_HELPER
    if _PARSER_HELPER is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "parser_helper.py")
        spec = importlib.util.spec_from_file_location("_sd_legacy_parser_helper", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PARSER_HELPER = mod
    return _PARSER_HELPER


def math_boxed(text: str) -> Optional[str]:
    """Last \\boxed{...} content. Unlike the legacy helper, text without any \\boxed / \\fbox returns None
    (the legacy last_boxed_only_string returns the whole text, so its <answer> fallback was unreachable
    and the full generation became the 'prediction'; scoring is unaffected in practice)."""
    if "\\boxed" not in text and "\\fbox" not in text:
        return None
    ph = _parser_helper()
    try:
        boxed = ph.last_boxed_only_string(text)
        return ph.remove_boxed(boxed) if boxed else None
    except Exception:
        return None


def math_gold(row_answer: Optional[str], solution: str) -> Optional[str]:
    if row_answer is not None and str(row_answer).strip():
        return str(row_answer).strip()
    return math_boxed(solution)


def recipe_hash(name: str) -> str:
    sp, suffix = RECIPES[name]
    return hashlib.sha256(f"{name}\n{sp}\n{suffix}".encode()).hexdigest()[:16]


def build_messages(question: str, recipe: str) -> List[Dict[str, str]]:
    sp, _ = RECIPES[recipe]
    if recipe == "klass_final_answer":
        return [{"role": "system", "content": sp}, {"role": "user", "content": question}]
    if recipe in ("boxed_step_by_step", "dmax_boxed"):
        return [{"role": "user", "content": question + sp}]
    return [{"role": "user", "content": sp + "\n\n" + question}]


def build_prompt(tokenizer, question: str, recipe: str) -> str:
    _, suffix = RECIPES[recipe]
    messages = build_messages(question, recipe)
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text + suffix


def gsm8k_gold(answer_text: str) -> Optional[float]:
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", answer_text)
    return float(m.group(1).replace(",", "")) if m else None


def parse_prediction(text: str, task: str = "gsm8k"):
    """gsm8k/fake: mirrors eval/utils.py::get_parsed_answer (first parsable \\boxed{}, else <answer>).
    math: mirrors eval/utils.py::get_parsed_answer_math (last \\boxed{}, else <answer> block), returns str."""
    if task == "math":
        pred = math_boxed(text)
        if not pred:
            m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            pred = m.group(1).strip() if m else None
        return pred or None
    for boxed in re.findall(r"\\boxed{(.*?)}", text):
        boxed = boxed.strip().replace(",", "")
        if not boxed or boxed == "..." or re.match(r"^\.+$", boxed):
            continue
        try:
            return float(boxed)
        except ValueError:
            nums = re.findall(r"-?\d+\.?\d*", boxed)
            if nums:
                try:
                    return float(nums[0])
                except ValueError:
                    pass
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m and m.group(1).strip():
        t = m.group(1).strip()
        try:
            return float(t)
        except ValueError:
            nums = re.findall(r"-?\d+\.?\d*", t)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    pass
    return None


def is_correct(pred, gold, task: str = "gsm8k") -> bool:
    if pred is None or gold is None:
        return False
    if task == "math":
        try:
            return bool(_parser_helper().is_equiv(str(pred), str(gold)))
        except Exception:
            return False
    return abs(float(pred) - float(gold)) < 1e-6


# ----------------------------------------------------------------------------- raw data
def load_raw(cfg: DataConfig) -> Dict[str, List[Tuple[str, str, str]]]:
    """{'train': [(id, question, answer_text)], 'test': [...]}; for math answer_text is the gold
    final answer (the dataset's `answer` field, else the last \\boxed{} of the solution)."""
    if cfg.dataset_id == "fake":
        return {"train": FAKE_PROBLEMS[:8], "test": FAKE_PROBLEMS[8:]}
    from datasets import load_dataset
    out = {}
    for split in ("train", "test"):
        if split == "test" and cfg.test_dataset_id:
            ds = load_dataset(cfg.test_dataset_id, split="test", revision=cfg.test_dataset_revision)
        else:
            ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=split, revision=cfg.dataset_revision)
        if cfg.task == "math":
            out[split] = [(f"{split}-{i}", r["problem"], math_gold(r.get("answer"), r["solution"]) or "")
                          for i, r in enumerate(ds)]
        else:
            out[split] = [(f"{split}-{i}", r["question"], r["answer"]) for i, r in enumerate(ds)]
    return out


def prepare_manifest(cfg: DataConfig, path: str, force: bool = False) -> dict:
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    raw = load_raw(cfg)
    train_ids = [r[0] for r in raw["train"]]
    rng = random.Random(cfg.seed)
    rng.shuffle(train_ids)
    sizes = dict(cfg.manifest_sizes)
    if cfg.dataset_id == "fake":
        sizes = {"train": 4, "dev": 2, "analysis": 2}
    need = sum(sizes.values())
    if need > len(train_ids):
        raise ValueError(f"manifest sizes sum to {need} > {len(train_ids)} official train rows")
    splits, cursor = {}, 0
    for name in ("train", "dev", "analysis"):
        splits[name] = train_ids[cursor:cursor + sizes[name]]
        cursor += sizes[name]
    splits["test"] = [r[0] for r in raw["test"]]
    manifest = {
        "task": cfg.task, "dataset_id": cfg.dataset_id, "dataset_config": cfg.dataset_config,
        "test_dataset_id": cfg.test_dataset_id,
        "dataset_revision": cfg.dataset_revision, "seed": cfg.seed,
        "sizes": {k: len(v) for k, v in splits.items()}, "splits": splits,
        "note": "train/dev/analysis are disjoint subsets of the official train split; test = official test",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# Question-level sub-splits of the parent manifest (plan §6.5): inside each parent split, sort by
# SHA256(tag + question_id) and cut consecutive slices in this order. Not stratified by type/level.
SUBSPLITS = {
    "train": ("mini-readout-v1", [("readout_train_initial", 256), ("readout_train_extension", 256)]),
    "analysis": ("mini-readout-v1", [("protocol", 128), ("readout_val", 64), ("readout_confirm", 64),
                                     ("sampling_pilot", 32), ("sampling_confirm", 32), ("mechanism", 128),
                                     ("analysis_reserve", 52)]),
    "dev": ("mini-dev-fast-v1", [("dev_fast", 128)]),
}


def subsplit_ids(parent_ids: List[str], parent: str, name: str, seed: int = 42) -> List[str]:
    if parent not in SUBSPLITS or name not in dict(SUBSPLITS[parent][1]):
        raise ValueError(f"unknown sub-split {parent}:{name}")
    tag, slices = SUBSPLITS[parent]
    if parent_ids and all(q.startswith("fake-") for q in parent_ids):
        return list(parent_ids)  # CPU fake dataset (a handful of ids): every sub-split is the whole parent
    order = sorted(parent_ids, key=lambda q: hashlib.sha256(f"{tag}|{seed}|{q}".encode("utf-8")).hexdigest())
    cursor = 0
    for sub, size in slices:
        if sub == name:
            if cursor + size > len(order):
                raise ValueError(f"{parent} has {len(order)} ids, too few for {name}")
            return order[cursor:cursor + size]
        cursor += size
    raise AssertionError("unreachable")


def load_split(cfg: Config, tokenizer) -> List[Example]:
    d = cfg.data
    raw = load_raw(d)
    by_id = {r[0]: r for split in raw.values() for r in split}
    if d.split_manifest is None:
        if d.dataset_id != "fake":
            raise ValueError("data.split_manifest is required (run prepare-data); refusing to guess a split")
        manifest = _fake_manifest(d)
    else:
        with open(d.split_manifest) as f:
            manifest = json.load(f)
        if manifest["dataset_id"] != d.dataset_id or manifest["dataset_config"] != d.dataset_config:
            raise ValueError("manifest dataset does not match config")
        if manifest.get("test_dataset_id") != d.test_dataset_id:
            raise ValueError(f"manifest test_dataset_id {manifest.get('test_dataset_id')} != config {d.test_dataset_id}")
        if manifest.get("task", "gsm8k") != d.task:
            raise ValueError(f"manifest task {manifest.get('task', 'gsm8k')} != data.task {d.task}")
    ids = manifest["splits"][d.split]
    if d.subsplit:
        ids = subsplit_ids(ids, d.split, d.subsplit, d.seed)
    if d.limit is not None:
        ids = ids[: d.limit]
    out = []
    for ex_id in ids:
        _, q, a = by_id[ex_id]
        gold = a if d.task == "math" else gsm8k_gold(a)
        out.append(Example(ex_id, ex_id.split("-")[0], q, a, gold, build_prompt(tokenizer, q, d.prompt_recipe), d.task))
    return out


def _fake_manifest(d: DataConfig) -> dict:
    ids = [p[0] for p in FAKE_PROBLEMS]
    return {"task": "fake", "dataset_id": "fake", "dataset_config": d.dataset_config,
            "splits": {"train": ids[:4], "dev": ids[4:6], "analysis": ids[6:8], "test": ids[8:]}}


def tokenize_prompts(tokenizer, prompts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
