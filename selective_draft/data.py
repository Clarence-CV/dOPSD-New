"""MATH with explicit sample IDs: split manifest, question-level sub-splits, the DMax prompt, and a CPU toy task."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .config import Config, DataConfig

# DMax (github.com/czg1225/DMax @43fafda, dInfer/evaluations/tasks/math/minerva_math_algebra.yaml `doc_to_text`,
# used by minerva_math500 with --apply_chat_template): one user turn, no extra system message.
DMAX_BOXED = "\nLet's think step by step, and put your final answer within \\boxed{}."

FAKE_PROBLEMS: List[Tuple[str, str, str]] = [
    (f"fake-{i}", f"What is {a} plus {b}?", f"{a} + {b} = {a + b}\n#### {a + b}")
    for i, (a, b) in enumerate([(2, 3), (5, 7), (10, 4), (1, 1), (8, 9), (6, 6), (3, 14), (20, 1),
                                (7, 2), (9, 9), (11, 3), (4, 4)])
]


@dataclass
class Example:
    example_id: str
    question: str
    answer_value: object  # LaTeX string for math, float for the toy task
    prompt_text: str
    task: str = "math"


def build_messages(question: str, recipe: str) -> List[Dict[str, str]]:
    if recipe == "dmax_boxed":
        return [{"role": "user", "content": question + DMAX_BOXED}]
    return [{"role": "user", "content": "Solve.\n\n" + question}]  # fake


def build_prompt(tokenizer, question: str, recipe: str) -> str:
    return tokenizer.apply_chat_template(build_messages(question, recipe), add_generation_prompt=True, tokenize=False)


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (nested braces) or \\boxed 5 in `text`."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    if text[i + 6:i + 7] == " ":
        return text[i + 7:].split("$")[0].strip() or None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def math_gold(row_answer: Optional[str], solution: str) -> Optional[str]:
    if row_answer is not None and str(row_answer).strip():
        return str(row_answer).strip()
    return last_boxed(solution)


# ----------------------------------------------------------------------------- raw data
def load_raw(cfg: DataConfig) -> Dict[str, List[Tuple[str, str, str]]]:
    """{'train': [(id, question, gold)], 'test': [...]}; gold = dataset `answer`, else the solution's last \\boxed{}."""
    if cfg.dataset_id == "fake":
        return {"train": FAKE_PROBLEMS[:8], "test": FAKE_PROBLEMS[8:]}
    from datasets import load_dataset
    out = {}
    for split in ("train", "test"):
        if split == "test" and cfg.test_dataset_id:
            ds = load_dataset(cfg.test_dataset_id, split="test", revision=cfg.test_dataset_revision)
        else:
            ds = load_dataset(cfg.dataset_id, split=split, revision=cfg.dataset_revision)
        out[split] = [(f"{split}-{i}", r["problem"], math_gold(r.get("answer"), r["solution"]) or "")
                      for i, r in enumerate(ds)]
    return out


def prepare_manifest(cfg: DataConfig, path: str, force: bool = False) -> dict:
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    raw = load_raw(cfg)
    train_ids = [r[0] for r in raw["train"]]
    random.Random(cfg.seed).shuffle(train_ids)
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
        "task": cfg.task, "dataset_id": cfg.dataset_id, "test_dataset_id": cfg.test_dataset_id,
        "dataset_revision": cfg.dataset_revision, "seed": cfg.seed,
        "sizes": {k: len(v) for k, v in splits.items()}, "splits": splits,
        "note": "train/dev/analysis are disjoint subsets of the official train split; test = official test",
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# Question-level sub-splits of the parent manifest: inside each parent split, sort by SHA256(tag|seed|id) and cut
# consecutive slices in this order. Not stratified by type/level.
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
        return list(parent_ids)  # CPU toy dataset (a handful of ids): every sub-split is the whole parent
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
        ids = [p[0] for p in FAKE_PROBLEMS]
        manifest = {"splits": {"train": ids[:4], "dev": ids[4:6], "analysis": ids[6:8], "test": ids[8:]}}
    else:
        with open(d.split_manifest) as f:
            manifest = json.load(f)
        if manifest["dataset_id"] != d.dataset_id:
            raise ValueError("manifest dataset does not match config")
        if manifest.get("test_dataset_id") != d.test_dataset_id:
            raise ValueError(f"manifest test_dataset_id {manifest.get('test_dataset_id')} != config {d.test_dataset_id}")
        if manifest.get("task", "math") != d.task:
            raise ValueError(f"manifest task {manifest.get('task', 'math')} != data.task {d.task}")
    ids = manifest["splits"][d.split]
    if d.subsplit:
        ids = subsplit_ids(ids, d.split, d.subsplit, d.seed)
    if d.limit is not None:
        ids = ids[: d.limit]
    out = []
    for ex_id in ids:
        _, q, a = by_id[ex_id]
        gold = a if d.task == "math" else float(a.rsplit("####", 1)[1])
        out.append(Example(ex_id, q, gold, build_prompt(tokenizer, q, d.prompt_recipe), d.task))
    return out


def tokenize_prompt(tokenizer, prompt: str, device) -> torch.Tensor:
    """[1, P] token ids, no padding (block-causal decoding is batch 1: padding would shift the block grid)."""
    return torch.as_tensor(tokenizer(prompt)["input_ids"], dtype=torch.long, device=device).reshape(1, -1)
