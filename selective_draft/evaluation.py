"""Task + latency evaluation with the shared decoder. One row per example per commit threshold.

LLaDA2-mini MATH runs for hours, so evaluation can be split over GPUs and resumed:
  shard=(i, n)  evaluates examples[i::n] into rows_<tag>.shard<i>of<n>.jsonl;
  resume=True   keeps rows already written under the same fingerprint (in any shard layout) and skips their ids.
Rows are flushed one example at a time. `merge_shards` checks that every example appears exactly once under one
fingerprint, then writes rows_<tag>.jsonl and summary.json like an unsharded run.
Metrics follow DMax's eval script: nfe_dinfer (prefill + unrolled over-run), tpf_dinfer (non-EOS tokens / nfe_dinfer),
averaged per example (tpf_dinfer_mean) and pooled (tpf_dinfer_pooled)."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

from .checkpoint import code_revision
from .config import Config
from .data import Example, tokenize_prompt
from .generation import generate
from .model import GuidedModel
from .scoring import grade
from .trace import StateRecorder, replay_check, save_rollouts

FINGERPRINT_FILE = "eval_fingerprint.json"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eval_fingerprint(cfg: Config, guidance_mode: str, snapshot_id: str, thresholds: List[float]) -> dict:
    """Everything that changes the produced rows. Re-using rows across a fingerprint change is refused."""
    fp = {
        "model": {k: getattr(cfg.model, k) for k in ("path", "revision", "backend", "dtype", "attn_implementation",
                                                      "fake_kwargs")},
        "generation": asdict(cfg.generation),
        "data": {k: getattr(cfg.data, k) for k in ("task", "dataset_id", "dataset_revision", "test_dataset_id",
                                                    "test_dataset_revision", "split", "subsplit", "split_manifest",
                                                    "prompt_recipe")},
        "guidance": {"mode": guidance_mode, "early_layer": cfg.guidance.early_layer, "scale": cfg.guidance.scale},
        "snapshot_id": snapshot_id, "thresholds": thresholds, "seed": cfg.training.seed,
    }
    fp["hash"] = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return fp


def _check_fingerprint(out_dir: str, fp: dict, resume: bool) -> None:
    path = os.path.join(out_dir, FINGERPRINT_FILE)
    if resume and os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("hash") != fp["hash"]:
            raise RuntimeError(f"{out_dir}: existing rows were produced under fingerprint {old.get('hash')}, "
                               f"current is {fp['hash']}; use a new output directory")
    with open(path, "w") as f:
        json.dump(fp, f, indent=1, default=str)


def _rows_name(tag: str, shard: Tuple[int, int]) -> str:
    i, n = shard
    return f"rows_{tag}.jsonl" if n == 1 else f"rows_{tag}.shard{i}of{n}.jsonl"


def _read_rows(path: str) -> List[dict]:
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # torn last line from a killed job: drop it, the example is redone
    return rows


def evaluate(gm: GuidedModel, tokenizer, examples: List[Example], cfg: Config, guidance_mode: str, out_dir: str,
             thresholds: Optional[List[float]] = None, snapshot_id: str = "none", trace_level: str = "none",
             shard: Tuple[int, int] = (0, 1), resume: bool = False) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    device = gm.device
    gen = cfg.generation
    thresholds = thresholds or (cfg.evaluation.commit_thresholds or [gen.commit_threshold])
    si, sn = shard
    if not (0 <= si < sn):
        raise ValueError(f"invalid shard {shard}")
    fp = eval_fingerprint(cfg, guidance_mode, snapshot_id, thresholds)
    _check_fingerprint(out_dir, fp, resume)
    meta = {"snapshot_id": snapshot_id, "fingerprint": fp["hash"], "code_revision": code_revision(),
            "model_revision": cfg.model.revision}
    summaries = {}
    for thr in thresholds:
        tag = f"thr{thr}"
        rows_path = os.path.join(out_dir, _rows_name(tag, shard))
        done_rows = _read_rows(rows_path) if resume else []
        done = {r["example_id"] for r in done_rows if r.get("fingerprint") == fp["hash"]}
        if resume and len(done) != len(done_rows):
            raise RuntimeError(f"{rows_path} holds rows from another fingerprint")
        if resume:  # examples finished under another shard layout (e.g. 2 GPUs before, 3 now) count as done
            for other in glob.glob(os.path.join(out_dir, f"rows_{tag}*.jsonl")):
                if os.path.abspath(other) != os.path.abspath(rows_path):
                    done |= {r["example_id"] for r in _read_rows(other) if r.get("fingerprint") == fp["hash"]}
        with open(rows_path, "w") as f:  # rewrite the intact prefix (drops a torn last line)
            for r in done_rows:
                f.write(json.dumps(r) + "\n")
        rollouts = []
        if examples and cfg.evaluation.warmup:  # untimed
            ids = tokenize_prompt(tokenizer, examples[0].prompt_text, device)
            for _ in range(cfg.evaluation.warmup):
                generate(gm, ids, gen, guidance_mode=guidance_mode, commit_threshold=thr)
        for idx, ex in enumerate(examples):
            if idx % sn != si or ex.example_id in done:
                continue
            row, rolled = _eval_example(gm, tokenizer, ex, idx, cfg, guidance_mode, thr, snapshot_id, trace_level, meta)
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if rolled is not None:
                rollouts.append(rolled)
        if rollouts:
            name = f"rollouts_{tag}.pt" if sn == 1 else f"rollouts_{tag}.shard{si}of{sn}.pt"
            save_rollouts(os.path.join(out_dir, name), rollouts)
        summaries[tag] = summarize_rows(_read_rows(rows_path))
    out = {"snapshot_id": snapshot_id, "guidance_mode": guidance_mode, "generation": asdict(gen),
           "fingerprint": fp["hash"], "shard": [si, sn],
           "timing": "median of identical-RNG repeats; trace recording excluded",
           "n_examples": len(examples), "summaries": summaries}
    name = "summary.json" if sn == 1 else f"summary.shard{si}of{sn}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _eval_example(gm, tokenizer, ex: Example, idx: int, cfg: Config, guidance_mode, thr, snapshot_id, trace_level, meta):
    device = gm.device
    gen = cfg.generation
    # Seeded by the example's position: threshold order, timing repeats, sharding and resume cannot change a
    # sampled trajectory (greedy runs draw nothing).
    rng = torch.Generator(device=device).manual_seed(cfg.training.seed + idx)
    start_rng = rng.get_state()
    ids = tokenize_prompt(tokenizer, ex.prompt_text, device)
    lat, res = [], None
    for _ in range(cfg.evaluation.repeats):
        rng.set_state(start_rng)
        _sync(device)
        t0 = time.perf_counter()
        timed = generate(gm, ids, gen, guidance_mode=guidance_mode, sample_rng=rng, commit_threshold=thr)
        _sync(device)
        lat.append(time.perf_counter() - t0)
        if res is None:
            res = timed
        elif not torch.equal(res.sequences, timed.sequences) or res.steps != timed.steps:
            raise RuntimeError("timing repeats changed the trajectory despite identical RNG state")
    text = tokenizer.decode(res.completion, skip_special_tokens=True)
    g = grade(ex.answer_value, text, ex.task)
    st = res.stats
    row = {
        "example_id": ex.example_id, "prompt_tokens": ids.shape[1],
        "prompt_token_hash": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest()[:16],
        "generation": text, "parsed_answer": g["parsed_answer"], "gold": ex.answer_value, "is_correct": g["is_correct"],
        "stop_reason": res.stop_reason, "eos_position": res.eos_position, "computed_tokens": res.computed_length,
        "output_tokens": st["output_tokens"], "row_steps": res.steps, "commit_counts": st["commit_counts"],
        "nfe_dinfer": st["nfe_dinfer"], "tpf_dinfer": st["tpf_dinfer"], "cache_used": st["cache_used"],
        "max_new_tokens_effective": st["max_new_tokens_effective"], "mask_argmax_fallbacks": st["mask_argmax_fallbacks"],
        "latency_s": statistics.median(lat), "latency_repeats_s": lat,
        "receiver_steps": st["receiver_steps"], "group_steps": st["group_steps"], "group_tokens": st["group_tokens"],
        "group_committed": st["group_committed"], "group_identity_changed": st["group_identity_changed"],
        "fallback_rows": st["fallback_rows"], "high_conf_group_sizes": st["high_conf_group_sizes"],
        "threshold": thr, "guidance_mode": guidance_mode, **meta,
    }
    rolled = None
    if trace_level != "none":
        # Recording performs CPU transfers; exclude it from the latency measurement.
        rec = StateRecorder(gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1],
                            {"example_id": ex.example_id, "prompt_ids": ids[0].tolist(), "gold": ex.answer_value,
                             "guidance_mode": guidance_mode, "snapshot_id": snapshot_id, "threshold": thr})
        rng.set_state(start_rng)
        traced = generate(gm, ids, gen, guidance_mode=guidance_mode, recorder=rec, sample_rng=rng, commit_threshold=thr)
        if not torch.equal(traced.sequences, res.sequences) or traced.steps != res.steps:
            raise RuntimeError("trace replay differs from the timed generation")
        rolled = rec.finalize(traced, {"generation": text, "parsed_answer": g["parsed_answer"],
                                       "is_correct": g["is_correct"], "stats": st})
        replay_check(rolled)
    return row, rolled


def merge_shards(out_dir: str, expected_ids: Optional[List[str]] = None) -> Dict[str, object]:
    """Combine rows_<tag>.shard*of<n>.jsonl into rows_<tag>.jsonl + summary.json (every example exactly once)."""
    with open(os.path.join(out_dir, FINGERPRINT_FILE)) as f:
        fp = json.load(f)
    tags = sorted({os.path.basename(p).split(".shard")[0][len("rows_"):]
                   for p in glob.glob(os.path.join(out_dir, "rows_*.shard*of*.jsonl"))})
    if not tags:
        raise FileNotFoundError(f"no shard rows under {out_dir}")
    summaries = {}
    for tag in tags:
        # shards from different layouts (resumed with another GPU count) are allowed
        paths = sorted(glob.glob(os.path.join(out_dir, f"rows_{tag}.shard*of*.jsonl")))
        rows = [r for p in paths for r in _read_rows(p)]
        if {r.get("fingerprint") for r in rows} != {fp["hash"]}:
            raise RuntimeError(f"{tag}: shard rows disagree with {FINGERPRINT_FILE}")
        ids = [r["example_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{tag}: duplicated example ids across shards")
        if expected_ids is not None and set(ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(ids))
            raise RuntimeError(f"{tag}: {len(missing)} examples missing, e.g. {missing[:5]}")
        order = {e: i for i, e in enumerate(expected_ids or ids)}
        rows.sort(key=lambda r: order[r["example_id"]])
        with open(os.path.join(out_dir, f"rows_{tag}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summaries[tag] = summarize_rows(rows)
    out = {"fingerprint": fp["hash"], "merged_from_shards": True, "n_examples": len(rows), "summaries": summaries}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize_rows(rows: List[dict]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["is_correct"] for r in rows) / n
    lat = [r["latency_s"] for r in rows]
    steps = [r["row_steps"] for r in rows]
    toks = [r["output_tokens"] for r in rows]
    recv = sum(r["receiver_steps"] for r in rows)
    group_tokens = sum(r["group_tokens"] for r in rows)
    return {
        "n": n, "accuracy": acc, "accuracy_se": (acc * (1 - acc) / n) ** 0.5,
        "unparsed": sum(not r.get("parsed_answer") for r in rows),
        "truncation_rate": sum(r["stop_reason"] == "length" for r in rows) / n,
        "mean_row_steps": sum(steps) / n, "median_row_steps": statistics.median(steps), "p90_row_steps": _pct(steps, 0.9),
        "mean_output_tokens": sum(toks) / n, "p90_output_tokens": _pct(toks, 0.9), "max_output_tokens": max(toks),
        "tokens_per_step": sum(toks) / max(1, sum(steps)),
        "tpf_dinfer_mean": sum(r["tpf_dinfer"] for r in rows) / n,  # DMax reports the mean of per-sample TPF
        "tpf_dinfer_pooled": sum(toks) / max(1, sum(r["nfe_dinfer"] for r in rows)),
        "mean_nfe_dinfer": sum(r["nfe_dinfer"] for r in rows) / n,
        "latency_total_s": sum(lat), "latency_median_s": statistics.median(lat), "latency_p90_s": _pct(lat, 0.9),
        "group_steps_per_row": sum(r["group_steps"] for r in rows) / n,
        "group_coverage_receiver_steps": sum(r["group_steps"] for r in rows) / max(1, recv),
        "group_committed_frac": sum(r["group_committed"] for r in rows) / max(1, group_tokens),
        "group_identity_changed_frac": sum(r["group_identity_changed"] for r in rows) / max(1, group_tokens),
        "fallback_per_row": sum(r["fallback_rows"] for r in rows) / n,
        "mask_argmax_fallbacks": sum(r["mask_argmax_fallbacks"] for r in rows),
        "cache_used_frac": sum(bool(r["cache_used"]) for r in rows) / n,
    }
