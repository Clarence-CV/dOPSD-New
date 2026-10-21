"""CPU state recorder for rollouts (training buffer, readout state banks) and replay checks."""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import torch

SCHEMA_VERSION = 2


class StateRecorder:
    def __init__(self, mask_id: int, eos_id: int, prompt_len: int, meta: dict):
        self.mask_id, self.eos_id, self.P = int(mask_id), int(eos_id), int(prompt_len)
        self.meta = meta
        self.steps: List[dict] = []

    @torch.no_grad()
    def record_step(self, *, step_id, block_id, block_start, block_end, x, eligible, commit, cand, conf, guidance):
        el = eligible[0].nonzero(as_tuple=False).flatten()
        cm = commit[0].nonzero(as_tuple=False).flatten()
        rec = {
            "step_id": int(step_id), "block_id": int(block_id), "block_start": int(block_start),
            "block_end": int(block_end),
            "state": x[0, self.P:].to(torch.int32).cpu(),
            "eligible_positions": el.to(torch.int32).cpu(),
            "eligible_confidences": conf[0, el].float().cpu(),
            "eligible_candidates": cand[0, el].to(torch.int32).cpu(),
            "committed_positions": cm.to(torch.int32).cpu(),
            "committed_tokens": cand[0, cm].to(torch.int32).cpu(),
        }
        if guidance is not None:
            valid = guidance.group_pos[0] >= 0
            rec["group_positions"] = guidance.group_pos[0, valid].to(torch.int32).cpu()
            gp = guidance.group_probs[0, valid]
            rec["group_confidences"] = gp.max(-1).values.float().cpu()
            rec["group_argmax"] = gp.argmax(-1).to(torch.int32).cpu()
            if guidance.delta_norm is not None:
                rec["delta_norm"] = float(guidance.delta_norm[0])
        self.steps.append(rec)

    def finalize(self, result, extras: dict) -> dict:
        roll = {
            "schema_version": SCHEMA_VERSION,
            "meta": dict(self.meta, prompt_len=self.P, mask_id=self.mask_id, eos_id=self.eos_id),
            "steps": self.steps,
            "final_state": result.sequences[0, self.P:].to(torch.int32).cpu(),
            "computed_length": int(result.computed_length),
            "eos_position": result.eos_position,
            "stop_reason": result.stop_reason,
            "steps_total": int(result.steps),
        }
        roll.update(extras)
        return roll


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
    """Consecutive recorded states differ exactly at the committed positions; the last transition reaches the final
    state. Raises AssertionError otherwise."""
    P = roll["meta"]["prompt_len"]
    steps = roll["steps"]
    for i, st in enumerate(steps):
        nxt = steps[i + 1]["state"] if i + 1 < len(steps) else roll["final_state"]
        cur = st["state"]
        n = min(cur.numel(), nxt.numel())  # growing window: compare the overlap
        expected = cur[:n].clone()
        cp = st["committed_positions"].long() - P
        expected[cp] = st["committed_tokens"].to(expected.dtype)
        if not torch.equal(expected, nxt[:n]):
            diff = (expected != nxt[:n]).nonzero().flatten().tolist()
            raise AssertionError(f"step {i}: replay mismatch at completion positions {diff[:10]}")
        changed = (cur[:n] != nxt[:n]).nonzero().flatten()
        if i + 1 < len(steps) and not torch.equal(changed.sort().values, cp.sort().values):
            raise AssertionError(f"step {i}: changed {changed.tolist()} != committed {cp.tolist()}")


def summarize_rollouts(rollouts: List[dict]) -> Dict[str, float]:
    n_steps = sum(len(r["steps"]) for r in rollouts)
    correct = [r.get("is_correct") for r in rollouts if "is_correct" in r]
    return {
        "rollouts": len(rollouts), "steps": n_steps, "mean_steps": n_steps / max(1, len(rollouts)),
        "accuracy": (sum(bool(c) for c in correct) / len(correct)) if correct else float("nan"),
        "mean_commit": sum(len(s["committed_positions"]) for r in rollouts for s in r["steps"]) / max(1, n_steps),
    }
