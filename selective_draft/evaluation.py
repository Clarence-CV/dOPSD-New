"""Task + latency evaluation with the shared decoder. One row per example per threshold;
commit-threshold sweeps give the accuracy–NFE curve (the primary comparison)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
import json
import os
import statistics
import time
from typing import Dict, List, Optional

import torch

from .config import Config
from .data import Example, is_correct, parse_prediction, tokenize_prompts
from .generation import generate
from .model import GuidedModel
from .trace import StateRecorder, replay_check, save_rollouts


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(gm: GuidedModel, tokenizer, examples: List[Example], cfg: Config, guidance_mode: str, out_dir: str,
             thresholds: Optional[List[Optional[float]]] = None, snapshot_id: str = "none",
             trace_level: str = "none") -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    device = gm.device
    gen = cfg.generation
    if gen.mode == "threshold":
        thresholds = thresholds or (cfg.evaluation.commit_thresholds or [gen.commit_threshold])
    else:
        thresholds = [None]
    bs = cfg.evaluation.batch_size
    summaries = {}
    for thr in thresholds:
        tag = f"thr{thr}" if thr is not None else f"fixed{gen.steps}"
        rows, rollouts = [], []
        # warmup (untimed)
        if examples and cfg.evaluation.warmup:
            ids, mask = tokenize_prompts(tokenizer, [examples[0].prompt_text], device)
            for _ in range(cfg.evaluation.warmup):
                generate(gm, ids, mask, gen, guidance_mode=guidance_mode, commit_threshold=thr)
        for s in range(0, len(examples), bs):
            batch = examples[s:s + bs]
            batch_id = s // bs
            # Threshold order and timing repeats must not change a batch's sampled trajectory.
            sample_rng = torch.Generator(device=device).manual_seed(cfg.training.seed + batch_id)
            start_rng = sample_rng.get_state()
            ids, mask = tokenize_prompts(tokenizer, [e.prompt_text for e in batch], device)
            rec = None
            if trace_level != "none":
                metas = [{"example_id": e.example_id, "prompt_ids": ids[i].tolist(),
                          "prompt_attention_mask": mask[i].tolist(), "gold": e.answer_value,
                          "guidance_mode": guidance_mode, "snapshot_id": snapshot_id,
                          "generation": dict(asdict(gen), commit_threshold=thr if thr is not None else gen.commit_threshold),
                          "confidence_policy": "raw_model_probability"} for i, e in enumerate(batch)]
                rec = StateRecorder(trace_level, gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1], metas,
                                    topk=cfg.trace.topk)
            lat = []
            res = None
            for rep in range(cfg.evaluation.repeats):
                sample_rng.set_state(start_rng)
                _sync(device)
                t0 = time.perf_counter()
                timed = generate(gm, ids, mask, gen, guidance_mode=guidance_mode,
                                 sample_rng=sample_rng, commit_threshold=thr)
                _sync(device)
                lat.append(time.perf_counter() - t0)
                if res is None:
                    res = timed
                elif not torch.equal(res.sequences, timed.sequences) or res.row_steps != timed.row_steps:
                    raise RuntimeError("timing repeats changed the trajectory despite identical RNG state")
            if rec is not None:
                # Recording performs CPU transfers; exclude it from the latency measurement.
                sample_rng.set_state(start_rng)
                traced = generate(gm, ids, mask, gen, guidance_mode=guidance_mode, recorder=rec,
                                  sample_rng=sample_rng, commit_threshold=thr)
                if not torch.equal(traced.sequences, res.sequences) or traced.row_steps != res.row_steps:
                    raise RuntimeError("trace replay differs from the timed generation")
            extras = []
            for i, e in enumerate(batch):
                text = tokenizer.decode(res.completion_ids(i), skip_special_tokens=True)
                pred = parse_prediction(text, e.task)
                ok = is_correct(pred, e.answer_value, e.task)
                rows.append({
                    "example_id": e.example_id, "prompt_hash": hashlib.sha256(e.prompt_text.encode()).hexdigest()[:16],
                    "generation": text, "parsed_answer": pred, "gold": e.answer_value, "is_correct": ok,
                    "stop_reason": res.stop_reasons[i], "effective_tokens": res.effective_lengths[i],
                    "computed_tokens": res.computed_lengths[i], "row_steps": res.row_steps[i],
                    "batch_id": batch_id, "batch_nfe": res.stats["nfe"], "batch_size": len(batch),
                    "latency_s": statistics.median(lat), "latency_repeats_s": lat,
                    "gate_open_steps": res.stats["gate_open_steps"], "receiver_steps": res.stats["receiver_steps"],
                    "gate_source_committed": res.stats["gate_source_committed"],
                    "gate_source_identity_changed": res.stats["gate_source_identity_changed"],
                    "group_tokens": res.stats["group_tokens"], "group_committed": res.stats["group_committed"],
                    "group_identity_changed": res.stats["group_identity_changed"],
                    "fallback_rows": res.stats["fallback_rows"],
                    "threshold": thr, "guidance_mode": guidance_mode, "snapshot_id": snapshot_id,
                })
                extras.append({"generation": text, "parsed_answer": pred, "is_correct": ok, "stats": res.stats})
            if rec is not None:
                recorded = rec.finalize(traced, extras)
                for roll in recorded:
                    replay_check(roll)
                rollouts.extend(recorded)
        with open(os.path.join(out_dir, f"rows_{tag}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if rollouts:
            save_rollouts(os.path.join(out_dir, f"rollouts_{tag}.pt"), rollouts)
        summaries[tag] = summarize_rows(rows)
    out = {"snapshot_id": snapshot_id, "guidance_mode": guidance_mode, "generation": gen.__dict__,
           "timing": "median of identical-RNG repeats per batch; trace recording excluded",
           "n_examples": len(examples), "summaries": summaries}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def summarize_rows(rows: List[dict]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["is_correct"] for r in rows) / n
    batches = {}
    for i, row in enumerate(rows):
        if "batch_id" not in row and row["batch_size"] != 1:
            raise ValueError("batched metrics require batch_id to avoid double-counting")
        batches.setdefault(row.get("batch_id", i), row)
    batch_rows = list(batches.values())
    lat = sorted(r["latency_s"] for r in batch_rows)
    nb = len(batch_rows)
    steps = [r["row_steps"] for r in rows]
    toks = [r["effective_tokens"] for r in rows]
    recv = sum(r["receiver_steps"] for r in batch_rows)
    gates = sum(r["gate_open_steps"] for r in batch_rows)
    group_tokens = sum(r.get("group_tokens", 0) for r in batch_rows)
    return {
        "n": n, "batches": nb, "accuracy": acc, "accuracy_se": (acc * (1 - acc) / n) ** 0.5,
        "latency_median_s": statistics.median(lat), "latency_p90_s": lat[min(nb - 1, int(0.9 * nb))],
        "latency_total_s": sum(lat), "mean_row_steps": sum(steps) / n,
        "mean_batch_nfe": sum(r["batch_nfe"] for r in batch_rows) / nb,
        "mean_effective_tokens": sum(toks) / n, "tokens_per_step": sum(toks) / max(1, sum(steps)),
        "truncation_rate": sum(r["stop_reason"] == "length" for r in rows) / n,
        "gate_open_per_row": gates / n,
        "gate_coverage_receiver_steps": gates / max(1, recv),
        "gate_source_committed_frac": sum(r["gate_source_committed"] for r in batch_rows) / max(1, gates),
        "gate_source_identity_changed_frac": sum(r["gate_source_identity_changed"] for r in batch_rows) / max(1, gates),
        "group_committed_frac": sum(r.get("group_committed", 0) for r in batch_rows) / max(1, group_tokens),
        "group_identity_changed_frac": sum(r.get("group_identity_changed", 0) for r in batch_rows) / max(1, group_tokens),
        "fallback_per_row": sum(r.get("fallback_rows", 0) for r in batch_rows) / n,
    }
