"""Task + latency evaluation with the shared decoder. One row per example per threshold;
commit-threshold sweeps give the accuracy–NFE curve (the primary comparison).

Long runs (LLaDA2-mini MATH, up to 2048 new tokens) can be split over GPUs and resumed:
  shard=(i, n)  evaluates examples[i::n] into rows_<tag>.shard<i>of<n>.jsonl (batch size 1 only);
  resume=True   keeps rows already written for the same fingerprint and skips their example IDs.
Rows are flushed one example at a time. `merge_shards` checks that every example appears exactly once
under one fingerprint, then writes rows_<tag>.jsonl and summary.json like an unsharded run."""

from __future__ import annotations

import glob
import hashlib
from dataclasses import asdict
import json
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

import torch

from .checkpoint import code_revision
from .config import Config
from .data import Example, tokenize_prompts
from .generation import generate
from .model import GuidedModel
from .scoring import grade
from .trace import StateRecorder, replay_check, save_rollouts

FINGERPRINT_FILE = "eval_fingerprint.json"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eval_fingerprint(cfg: Config, guidance_mode: str, snapshot_id: str, thresholds: List[Optional[float]]) -> dict:
    """Everything that changes the produced rows. Re-using rows across a fingerprint change is refused."""
    fp = {
        "model": {k: getattr(cfg.model, k) for k in ("path", "revision", "backend", "dtype", "attn_implementation",
                                                      "fake_kwargs")},
        "generation": asdict(cfg.generation),
        "data": {k: getattr(cfg.data, k) for k in ("task", "dataset_id", "dataset_config", "dataset_revision",
                                                    "test_dataset_id", "test_dataset_revision", "split",
                                                    "split_manifest", "prompt_recipe", "scorer")},
        "guidance": {"mode": guidance_mode, "early_layer": cfg.guidance.early_layer,
                     "group_policy": cfg.guidance.group_policy, "group_size": cfg.guidance.group_size,
                     "scale": cfg.guidance.scale},
        "snapshot_id": snapshot_id, "thresholds": thresholds, "seed": cfg.training.seed,
        "batch_size": cfg.evaluation.batch_size,
    }
    fp["hash"] = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return fp


def _check_fingerprint(out_dir: str, fp: dict, resume: bool) -> None:
    path = os.path.join(out_dir, FINGERPRINT_FILE)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("hash") != fp["hash"]:
            if resume:
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
             thresholds: Optional[List[Optional[float]]] = None, snapshot_id: str = "none",
             trace_level: str = "none", shard: Tuple[int, int] = (0, 1), resume: bool = False) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    device = gm.device
    gen = cfg.generation
    if gen.mode == "threshold":
        thresholds = thresholds or (cfg.evaluation.commit_thresholds or [gen.commit_threshold])
    else:
        thresholds = [None]
    bs = cfg.evaluation.batch_size
    si, sn = shard
    if not (0 <= si < sn):
        raise ValueError(f"invalid shard {shard}")
    if sn > 1 and bs != 1:
        raise ValueError("sharded evaluation requires evaluation.batch_size=1")
    fp = eval_fingerprint(cfg, guidance_mode, snapshot_id, thresholds)
    _check_fingerprint(out_dir, fp, resume)
    meta_common = {"snapshot_id": snapshot_id, "fingerprint": fp["hash"], "code_revision": code_revision(),
                   "model_revision": cfg.model.revision}
    summaries = {}
    for thr in thresholds:
        tag = f"thr{thr}" if thr is not None else f"fixed{gen.steps}"
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
            ids, mask = tokenize_prompts(tokenizer, [examples[0].prompt_text], device)
            for _ in range(cfg.evaluation.warmup):
                generate(gm, ids, mask, gen, guidance_mode=guidance_mode, commit_threshold=thr)
        for s in range(0, len(examples), bs):
            batch_id = s // bs
            if sn > 1 and batch_id % sn != si:
                continue
            batch = examples[s:s + bs]
            if all(e.example_id in done for e in batch):
                continue
            new_rows, recorded = _eval_batch(gm, tokenizer, batch, batch_id, cfg, guidance_mode, thr,
                                             snapshot_id, trace_level, meta_common)
            with open(rows_path, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
            rollouts.extend(recorded)
        if rollouts:
            name = f"rollouts_{tag}.pt" if sn == 1 else f"rollouts_{tag}.shard{si}of{sn}.pt"
            save_rollouts(os.path.join(out_dir, name), rollouts)
        summaries[tag] = summarize_rows(_read_rows(rows_path))
    out = {"snapshot_id": snapshot_id, "guidance_mode": guidance_mode, "generation": gen.__dict__,
           "fingerprint": fp["hash"], "shard": [si, sn],
           "timing": "median of identical-RNG repeats per batch; trace recording excluded",
           "n_examples": len(examples), "summaries": summaries}
    name = "summary.json" if sn == 1 else f"summary.shard{si}of{sn}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _eval_batch(gm, tokenizer, batch, batch_id, cfg, guidance_mode, thr, snapshot_id, trace_level, meta_common):
    device = gm.device
    gen = cfg.generation
    # Seeded by the global batch index: threshold order, timing repeats, sharding and resume cannot change
    # a batch's sampled trajectory.
    sample_rng = torch.Generator(device=device).manual_seed(cfg.training.seed + batch_id)
    start_rng = sample_rng.get_state()
    ids, mask = tokenize_prompts(tokenizer, [e.prompt_text for e in batch], device)
    rec = None
    if trace_level != "none":
        metas = [{"example_id": e.example_id, "prompt_ids": ids[i].tolist(),
                  "prompt_attention_mask": mask[i].tolist(), "gold": e.answer_value, "question": e.question,
                  "guidance_mode": guidance_mode, "snapshot_id": snapshot_id,
                  "generation": dict(asdict(gen), commit_threshold=thr if thr is not None else gen.commit_threshold),
                  "confidence_policy": "raw_model_probability"} for i, e in enumerate(batch)]
        rec = StateRecorder(trace_level, gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1], metas,
                            topk=cfg.trace.topk)
    lat = []
    res = None
    for _ in range(cfg.evaluation.repeats):
        sample_rng.set_state(start_rng)
        _sync(device)
        t0 = time.perf_counter()
        timed = generate(gm, ids, mask, gen, guidance_mode=guidance_mode, sample_rng=sample_rng, commit_threshold=thr)
        _sync(device)
        lat.append(time.perf_counter() - t0)
        if res is None:
            res = timed
        elif not torch.equal(res.sequences, timed.sequences) or res.row_steps != timed.row_steps:
            raise RuntimeError("timing repeats changed the trajectory despite identical RNG state")
    traced = None
    if rec is not None:
        # Recording performs CPU transfers; exclude it from the latency measurement.
        sample_rng.set_state(start_rng)
        traced = generate(gm, ids, mask, gen, guidance_mode=guidance_mode, recorder=rec,
                          sample_rng=sample_rng, commit_threshold=thr)
        if not torch.equal(traced.sequences, res.sequences) or traced.row_steps != res.row_steps:
            raise RuntimeError("trace replay differs from the timed generation")
    rows, extras = [], []
    st = res.stats
    for i, e in enumerate(batch):
        comp = res.completion_ids(i)
        text = tokenizer.decode(comp, skip_special_tokens=True)
        g = grade(e.question, e.answer_value, text, e.task, cfg.data.scorer)
        prompt_ids = ids[i][mask[i].bool()].tolist()
        row = {
            "example_id": e.example_id, "prompt_hash": hashlib.sha256(e.prompt_text.encode()).hexdigest()[:16],
            "prompt_token_hash": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16],
            "prompt_tokens": len(prompt_ids),
            "generation": text, "parsed_answer": g["parsed_answer"], "gold": e.answer_value,
            "is_correct": g["is_correct"], "scorer": g["scorer"], "verdicts": g["verdicts"],
            "stop_reason": res.stop_reasons[i], "effective_tokens": res.effective_lengths[i],
            "computed_tokens": res.computed_lengths[i], "row_steps": res.row_steps[i],
            "batch_id": batch_id, "batch_nfe": st["nfe"], "batch_size": len(batch),
            "latency_s": statistics.median(lat), "latency_repeats_s": lat,
            "gate_open_steps": st["gate_open_steps"], "receiver_steps": st["receiver_steps"],
            "gate_source_committed": st["gate_source_committed"],
            "gate_source_identity_changed": st["gate_source_identity_changed"],
            "group_tokens": st["group_tokens"], "group_committed": st["group_committed"],
            "group_identity_changed": st["group_identity_changed"],
            "fallback_rows": st["fallback_rows"], "high_conf_group_sizes": st.get("high_conf_group_sizes", {}),
            "threshold": thr, "guidance_mode": guidance_mode, **meta_common,
        }
        if g.get("klass_error"):
            row["klass_error"] = g["klass_error"]
        if "physical_end" in st:  # block-causal decoder (batch 1)
            row.update(completion_token_ids=comp, eos_position=st["eos_position"],
                       first_role_end=st["first_role_end"], physical_end=st["physical_end"],
                       blocks_decoded=st["blocks_decoded"], commit_counts=st["commit_counts"],
                       # dInfer eval-script accounting: NFE incl. prefill + unrolled over-run, TPF over non-EOS tokens
                       nfe_decode=st["nfe_decode"], nfe_dinfer=st["nfe_dinfer"], output_tokens=st["output_tokens"],
                       tpf_dinfer=st["tpf_dinfer"], cache_used=st["cache_used"],
                       max_new_tokens_effective=st["max_new_tokens_effective"],
                       mask_argmax_fallbacks=st["mask_argmax_fallbacks"])
        rows.append(row)
        extras.append({"generation": text, "parsed_answer": g["parsed_answer"], "is_correct": g["is_correct"],
                       "verdicts": g["verdicts"], "stats": st})
    recorded = []
    if rec is not None:
        recorded = rec.finalize(traced, extras)
        for roll in recorded:
            replay_check(roll)
    return rows, recorded


def merge_shards(out_dir: str, expected_ids: Optional[List[str]] = None) -> Dict[str, object]:
    """Combine rows_<tag>.shard*of<n>.jsonl into rows_<tag>.jsonl + summary.json (all shards, one fingerprint)."""
    with open(os.path.join(out_dir, FINGERPRINT_FILE)) as f:
        fp = json.load(f)
    tags = sorted({os.path.basename(p).split(".shard")[0][len("rows_"):]
                   for p in glob.glob(os.path.join(out_dir, "rows_*.shard*of*.jsonl"))})
    if not tags:
        raise FileNotFoundError(f"no shard rows under {out_dir}")
    summaries = {}
    for tag in tags:
        # shards from different layouts (resumed with another GPU count) are allowed; each example exactly once
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
        rows.sort(key=lambda r: r["batch_id"])
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
    out = {
        "n": n, "batches": nb, "accuracy": acc, "accuracy_se": (acc * (1 - acc) / n) ** 0.5,
        "latency_median_s": statistics.median(lat), "latency_p90_s": lat[min(nb - 1, int(0.9 * nb))],
        "latency_total_s": sum(lat), "mean_row_steps": sum(steps) / n,
        "median_row_steps": statistics.median(steps), "p90_row_steps": _pct(steps, 0.9),
        "mean_batch_nfe": sum(r["batch_nfe"] for r in batch_rows) / nb,
        "mean_effective_tokens": sum(toks) / n, "p90_effective_tokens": _pct(toks, 0.9),
        "max_effective_tokens": max(toks), "tokens_per_step": sum(toks) / max(1, sum(steps)),
        "truncation_rate": sum(r["stop_reason"] == "length" for r in rows) / n,
        "gate_open_per_row": gates / n,
        "gate_coverage_receiver_steps": gates / max(1, recv),
        "gate_source_committed_frac": sum(r["gate_source_committed"] for r in batch_rows) / max(1, gates),
        "gate_source_identity_changed_frac": sum(r["gate_source_identity_changed"] for r in batch_rows) / max(1, gates),
        "group_committed_frac": sum(r.get("group_committed", 0) for r in batch_rows) / max(1, group_tokens),
        "group_identity_changed_frac": sum(r.get("group_identity_changed", 0) for r in batch_rows) / max(1, group_tokens),
        "fallback_per_row": sum(r.get("fallback_rows", 0) for r in batch_rows) / n,
    }
    verdicts = [r["verdicts"] for r in rows if "verdicts" in r]
    if verdicts:
        for name in sorted({k for v in verdicts for k in v}):
            vals = [bool(v[name]) for v in verdicts if name in v]
            out[f"accuracy_{name}"] = sum(vals) / len(vals)
        out["unparsed"] = sum(not r.get("parsed_answer") for r in rows)
    if any("tpf_dinfer" in r for r in rows):
        tp = [r for r in rows if "tpf_dinfer" in r]
        out["tpf_dinfer_mean"] = sum(r["tpf_dinfer"] for r in tp) / len(tp)  # DMax reports the mean of per-sample TPF
        out["tpf_dinfer_pooled"] = sum(r["output_tokens"] for r in tp) / max(1, sum(r["nfe_dinfer"] for r in tp))
        out["mean_nfe_dinfer"] = sum(r["nfe_dinfer"] for r in tp) / len(tp)
        out["mask_argmax_fallbacks"] = sum(r["mask_argmax_fallbacks"] for r in tp)
        out["cache_used_frac"] = sum(bool(r["cache_used"]) for r in tp) / len(tp)
    if any("first_role_end" in r for r in rows):
        out["role_end_in_completion"] = sum(r.get("first_role_end") is not None for r in rows)
    return out
