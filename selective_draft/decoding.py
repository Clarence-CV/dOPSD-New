"""Decoding primitives: candidate + raw confidence, dInfer's commit rule, dInfer's NFE accounting, per-step statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

import torch
import torch.nn.functional as F

DINFER_TIE_EPS = 1e-5


def choose_tokens(logits: torch.Tensor, temperature: float, rng: Optional[torch.Generator],
                  forbidden_token: Optional[int] = None):
    """Candidate per position and its raw probability under the un-noised distribution.
    temperature>0: Gumbel-max on logits + T*g (same argmax law as the legacy exp(logits)/(-log u)^T).
    temperature=0: argmax, no RNG consumed (LLaDA2.0's HF sampler samples even at T=0).
    forbidden_token cannot be selected, but confidence is always measured in the original model distribution,
    without renormalising away the forbidden probability."""
    lf = logits.float()
    probs = F.softmax(lf, dim=-1)
    if forbidden_token is not None:
        lf = lf.clone()
        lf[..., forbidden_token] = -math.inf
    if temperature > 0.0:
        u = torch.rand(lf.shape, generator=rng, device=lf.device, dtype=torch.float32)
        neg_log_u = -torch.log(u.clamp_min(1e-20))
        g = -torch.log(neg_log_u.clamp_min(1e-20))  # standard Gumbel(0, 1)
        cand = torch.argmax(lf + temperature * g, dim=-1)
    else:
        cand = torch.argmax(lf, dim=-1)
    conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
    return cand, conf


def dinfer_commits(eligible: torch.Tensor, conf: torch.Tensor, threshold: float) -> torch.Tensor:
    """dInfer ThresholdParallelDecoder (get_transfer_index_threshold, DMax commit 43fafda): per row commit every
    eligible position with confidence >= min(threshold, max_conf - 1e-5). `>=` keeps ties at the threshold; the best
    position always qualifies, and so does anything within 1e-5 of it. A row without any eligible position (or all
    -inf) commits nothing; the caller decides what to do then."""
    scores = conf.masked_fill(~eligible, -math.inf)
    best = scores.max(dim=1, keepdim=True).values
    actual = (best - DINFER_TIE_EPS).clamp(-1000.0, threshold)
    return eligible & (scores >= actual)


def dinfer_block_forwards(commits: List[int], initial_masks: int, expected_tpf: int = 4, max_unroll: int = 4) -> int:
    """Forward count dInfer's BlockDiffusionRunner.decode charges for one block, replayed from our per-step commit
    counts. It runs k = max(min(masks // expected_tpf, max_unroll), 1) forwards back to back before looking at the
    block again, so a block that finishes inside such a batch still pays for the remaining forwards (they commit
    nothing). Token output is unaffected; only the NFE/TPF accounting is."""
    remaining, forwards, i = int(initial_masks), 0, 0
    while remaining > 0 and (i < len(commits) or forwards == 0):
        k = max(min(remaining // expected_tpf, max_unroll), 1)
        for _ in range(k):
            forwards += 1
            if i < len(commits):
                remaining -= commits[i]
                i += 1
    return forwards  # a block cut short (commits exhausted with masks left) is charged only for what ran


class StepAccounting:
    """Per-forward counters reported in GenerationResult.stats."""

    def __init__(self):
        self.nfe = 0
        self.commit_counts: List[int] = []
        self.receiver_steps = 0  # steps with >= 2 eligible positions
        self.group_steps = 0  # steps in which a communication group (>= 2 high-confidence drafts) formed
        self.group_tokens = self.group_committed = self.group_changed = self.fallback_rows = 0
        self.high_conf_group_sizes: Counter = Counter()  # |{draft confidence > tau}| per live step
        self.skip_reasons: Counter = Counter()

    def step(self, out, ctx, eligible, cand, conf, commit, tau: float) -> None:
        self.nfe += 1
        live = eligible.any(1)
        self.fallback_rows += int((live & ~(eligible & (conf >= tau)).any(1)).sum())
        gi = out.guidance
        if gi is not None:
            for r in gi.skip_reason:
                self.skip_reasons[r] += 1
            self.receiver_steps += int(((gi.n_eligible >= 2) & ctx.active_rows).sum())
            self.group_steps += int(gi.gate_open.sum())
            if gi.n_high is not None:
                for b in ctx.active_rows.nonzero(as_tuple=False).flatten().tolist():
                    self.high_conf_group_sizes[int(gi.n_high[b])] += 1
            rr, ss = ((gi.group_pos >= 0) & gi.gate_open[:, None]).nonzero(as_tuple=True)
            pp = gi.group_pos[rr, ss]
            self.group_tokens += len(pp)
            self.group_committed += int(commit[rr, pp].sum())
            self.group_changed += int((cand[rr, pp] != gi.group_probs[rr, ss].argmax(-1)).sum())
        self.commit_counts.append(int(commit.sum()))

    def stats(self, **extra) -> dict:
        out = {
            "nfe": self.nfe, "commit_counts": self.commit_counts, "receiver_steps": self.receiver_steps,
            "group_steps": self.group_steps, "group_tokens": self.group_tokens,
            "group_committed": self.group_committed, "group_identity_changed": self.group_changed,
            "fallback_rows": self.fallback_rows, "skip_reasons": dict(self.skip_reasons),
            "high_conf_group_sizes": {str(k): v for k, v in sorted(self.high_conf_group_sizes.items())},
        }
        out.update(extra)
        return out
