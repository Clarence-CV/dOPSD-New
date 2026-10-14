"""GSM8K with explicit sample IDs: split manifest, one prompt recipe shared by collection,
training, diagnostics and evaluation, and the legacy answer parser."""

from __future__ import annotations

import hashlib
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

RECIPES = {
    # (system prompt, text appended after the generation prompt)
    "legacy_boxed_reasoning": (GSM_SYSTEM_PROMPT, "<reasoning>"),
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
    answer_value: Optional[float]
    prompt_text: str


def recipe_hash(name: str) -> str:
    sp, suffix = RECIPES[name]
    return hashlib.sha256(f"{name}\n{sp}\n{suffix}".encode()).hexdigest()[:16]


def build_prompt(tokenizer, question: str, recipe: str) -> str:
    sp, suffix = RECIPES[recipe]
    messages = [{"role": "user", "content": sp + "\n\n" + question}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return text + suffix


def gsm8k_gold(answer_text: str) -> Optional[float]:
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", answer_text)
    return float(m.group(1).replace(",", "")) if m else None


def parse_prediction(text: str) -> Optional[float]:
    """Mirrors eval/utils.py::get_parsed_answer: first parsable \\boxed{}, else <answer> block."""
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


def is_correct(pred: Optional[float], gold: Optional[float]) -> bool:
    return pred is not None and gold is not None and abs(pred - gold) < 1e-6


# ----------------------------------------------------------------------------- raw data
def load_raw(cfg: DataConfig) -> Dict[str, List[Tuple[str, str, str]]]:
    """{'train': [(id, question, answer_text)], 'test': [...]}"""
    if cfg.dataset_id == "fake":
        return {"train": FAKE_PROBLEMS[:8], "test": FAKE_PROBLEMS[8:]}
    from datasets import load_dataset
    out = {}
    for split in ("train", "test"):
        ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=split, revision=cfg.dataset_revision)
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
        "dataset_id": cfg.dataset_id, "dataset_config": cfg.dataset_config,
        "dataset_revision": cfg.dataset_revision, "seed": cfg.seed,
        "sizes": {k: len(v) for k, v in splits.items()}, "splits": splits,
        "note": "train/dev/analysis are disjoint subsets of the official train split; test = official test",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


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
    ids = manifest["splits"][d.split]
    if d.limit is not None:
        ids = ids[: d.limit]
    out = []
    for ex_id in ids:
        _, q, a = by_id[ex_id]
        out.append(Example(ex_id, ex_id.split("-")[0], q, a, gsm8k_gold(a), build_prompt(tokenizer, q, d.prompt_recipe)))
    return out


def _fake_manifest(d: DataConfig) -> dict:
    ids = [p[0] for p in FAKE_PROBLEMS]
    return {"dataset_id": "fake", "dataset_config": d.dataset_config,
            "splits": {"train": ids[:4], "dev": ids[4:6], "analysis": ids[6:8], "test": ids[8:]}}


def tokenize_prompts(tokenizer, prompts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
