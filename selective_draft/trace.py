"""CPU state recorder for rollouts (training buffer + diagnostics) and replay checks.
Levels: state_only (tokens, commits, confidences, guidance scalars) and diagnostic (+ source top-k)."""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch

SCHEMA_VERSION = 1


class StateRecorder:
    def __init__(self, level: str, mask_id: int, eos_id: int, prompt_len: int, row_meta: List[dict], topk: int = 8):
        if level not in ("state_only", "diagnostic"):
            raise ValueError(level)
        self.level, self.mask_id, self.eos_id, self.P, self.topk = level, int(mask_id), int(eos_id), int(prompt_len), int(topk)
        self.row_meta = row_meta
        self.steps: List[List[dict]] = [[] for _ in row_meta]

    @torch.no_grad()
    def record_step(self, *, step_id, block_id, block_start, block_end, x, attention_mask, eligible, commit,
                    cand, conf, active, guidance, logits=None):
        B = x.shape[0]
        for b in range(B):
            if not bool(active[b]) or not bool(eligible[b].any()):
                continue
            el = eligible[b].nonzero(as_tuple=False).flatten()
            cm = commit[b].nonzero(as_tuple=False).flatten()
            rec = {
                "step_id": int(step_id[b]), "block_id": int(block_id), "block_start": int(block_start),
                "block_end": int(block_end),
                "state": x[b, self.P:].to(torch.int32).cpu(),
                "eligible_positions": el.to(torch.int16).cpu(),
                "eligible_confidences": conf[b, el].float().cpu(),  # preserve threshold comparisons
                "eligible_candidates": cand[b, el].to(torch.int32).cpu(),
                "committed_positions": cm.to(torch.int16).cpu(),
                "committed_tokens": cand[b, cm].to(torch.int32).cpu(),
                "source_pos": -1, "source_confidence": 0.0, "gate_open": False, "skip_reason": "disabled",
            }
            if guidance is not None:
                rec["source_pos"] = int(guidance.source_pos[b])
                rec["source_confidence"] = float(guidance.source_confidence[b])
                rec["gate_open"] = bool(guidance.gate_open[b])
                rec["skip_reason"] = guidance.skip_reason[b]
                if guidance.source_argmax is not None:
                    rec["source_argmax"] = int(guidance.source_argmax[b])
                A = rec["source_pos"]
                if A >= 0:
                    rec["source_final_argmax"] = int(cand[b, A])
                    rec["source_was_committed"] = bool(commit[b, A])
                    rec["source_final_confidence"] = float(conf[b, A])
                if self.level == "diagnostic" and rec["gate_open"] and guidance.source_probs is not None:
                    p = guidance.source_probs[b]
                    tp, ti = p.topk(min(self.topk, p.shape[-1]))
                    rec["source_topk_ids"] = ti.to(torch.int32).cpu()
                    rec["source_topk_probs"] = tp.to(torch.float16).cpu()
                    rec["source_topk_mass"] = float(tp.sum())
                    rec["source_entropy"] = float(-(p * (p.clamp_min(1e-30)).log()).sum())
                if guidance.delta_norm is not None:
                    rec["delta_norm"] = float(guidance.delta_norm[b])
                if guidance.group_pos is not None:
                    valid = guidance.group_pos[b] >= 0
                    rec["group_positions"] = guidance.group_pos[b, valid].to(torch.int16).cpu()
                    gp = guidance.group_probs[b, valid]
                    rec["group_confidences"] = gp.max(-1).values.float().cpu()
                    rec["group_argmax"] = gp.argmax(-1).to(torch.int32).cpu()
            self.steps[b].append(rec)

    def finalize(self, result, extras: Optional[List[dict]] = None) -> List[dict]:
        out = []
        for b, meta in enumerate(self.row_meta):
            roll = {
                "schema_version": SCHEMA_VERSION,
                "meta": dict(meta, prompt_len=self.P, mask_id=self.mask_id, eos_id=self.eos_id),
                "steps": self.steps[b],
                "final_state": result.sequences[b, self.P:].to(torch.int32).cpu(),
                "completion_length": int(result.effective_lengths[b]),
                "computed_length": int(result.computed_lengths[b]),
                "stop_reason": result.stop_reasons[b],
                "row_steps": int(result.row_steps[b]),
            }
            if extras is not None:
                roll.update(extras[b])
            out.append(roll)
        return out


def save_rollouts(path: str, rollouts: List[dict]) -> None:
    torch.save({"schema_version": SCHEMA_VERSION, "rollouts": rollouts}, path)


def load_rollouts(path: str) -> List[dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {blob.get('schema_version')} != {SCHEMA_VERSION}")
    return blob["rollouts"]


def iter_states(rollouts: List[dict]) -> Iterator[Tuple[int, int, dict, dict]]:
    for ri, roll in enumerate(rollouts):
        for si, step in enumerate(roll["steps"]):
            yield ri, si, roll, step


def replay_check(roll: dict) -> None:
    """Consecutive recorded states differ exactly at the committed positions; the last transition
    reaches the final completion. Raises AssertionError with a message otherwise."""
    P = roll["meta"]["prompt_len"]
    steps = roll["steps"]
    for i, st in enumerate(steps):
        nxt = steps[i + 1]["state"] if i + 1 < len(steps) else roll["final_state"]
        cur = st["state"]
        n = min(cur.numel(), nxt.numel())  # growing windows: compare the overlap
        expected = cur[:n].clone()
        cp = st["committed_positions"].long() - P
        expected[cp] = st["committed_tokens"].to(expected.dtype)
        if not torch.equal(expected, nxt[:n]):
            diff = (expected != nxt[:n]).nonzero().flatten().tolist()
            raise AssertionError(f"step {i}: replay mismatch at completion positions {diff[:10]}")
        if i + 1 < len(steps):
            # the only change between consecutive recorded states is the commit set
            changed = (cur[:n] != nxt[:n]).nonzero().flatten()
            if not torch.equal(changed.sort().values, cp.sort().values):
                raise AssertionError(f"step {i}: changed {changed.tolist()} != committed {cp.tolist()}")


def summarize_rollouts(rollouts: List[dict]) -> Dict[str, float]:
    n_steps = sum(len(r["steps"]) for r in rollouts)
    gate = sum(1 for r in rollouts for s in r["steps"] if s["gate_open"])
    recv = sum(1 for r in rollouts for s in r["steps"] if len(s["eligible_positions"]) >= 2)
    correct = [r.get("is_correct") for r in rollouts if "is_correct" in r]
    return {
        "rollouts": len(rollouts), "steps": n_steps,
        "mean_steps": n_steps / max(1, len(rollouts)),
        "gate_open_frac_all": gate / max(1, n_steps),
        "gate_open_frac_receiver_steps": gate / max(1, recv),
        "accuracy": (sum(bool(c) for c in correct) / len(correct)) if correct else float("nan"),
        "mean_commit": sum(len(s["committed_positions"]) for r in rollouts for s in r["steps"]) / max(1, n_steps),
    }
