"""Decoding primitives shared by every decoder (LLaDA fixed/growing window, LLaDA2 block-causal),
the guidance group rule and the same-state diagnostics: candidate + raw confidence, commit rule,
fixed-budget schedule, and the per-step statistics both decoders report."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

import torch
import torch.nn.functional as F


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """LLaDA's integer schedule: [B, steps]; remainder goes to the earliest steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = base.expand(-1, steps).clone()
    idx = torch.arange(steps, device=mask_index.device)
    out = out + (idx.unsqueeze(0) < remainder).long()
    return out.to(torch.int64)


def choose_tokens(logits: torch.Tensor, temperature: float, rng: Optional[torch.Generator],
                  forbidden_token: Optional[int] = None):
    """Candidate per position and its raw probability under the un-noised distribution.
    temperature>0: Gumbel-max on logits + T*g (same argmax law as the legacy exp(logits)/(-log u)^T).
    temperature=0: argmax, no RNG consumed (LLaDA2.0's HF sampler samples even at T=0; see plan §5.2).
    forbidden_token (usually MASK) cannot be selected, but confidence is always measured
    in the original model distribution, without renormalising away forbidden probability."""
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


def select_commits(eligible: torch.Tensor, conf: torch.Tensor, mode: str, threshold: float,
                   k_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Shared commit rule for generation and same-state counterfactual diagnostics."""
    scores = conf.masked_fill(~eligible, -math.inf)
    live = eligible.any(dim=1)
    commit = torch.zeros_like(eligible)
    if mode == "threshold":
        commit = eligible & (scores > threshold)
        for b in (live & ~commit.any(dim=1)).nonzero(as_tuple=False).flatten().tolist():
            commit[b, int(scores[b].argmax())] = True
    elif mode == "fixed_budget":
        if k_t is None:
            raise ValueError("fixed_budget requires k_t")
        for b in live.nonzero(as_tuple=False).flatten().tolist():
            k = min(int(k_t[b]), int(eligible[b].sum()))
            if k > 0:
                commit[b, torch.topk(scores[b], k=k).indices] = True
    else:
        raise ValueError(f"unknown commit mode {mode}")
    return commit


DINFER_TIE_EPS = 1e-5


def dinfer_commits(eligible: torch.Tensor, conf: torch.Tensor, threshold: float) -> torch.Tensor:
    """dInfer ThresholdParallelDecoder (get_transfer_index_threshold, DMax commit
    43fafda): per row commit every eligible position with confidence >= min(threshold, max_conf - 1e-5).
    `>=` keeps ties at the threshold; the best position always qualifies, and so does anything within 1e-5 of it.
    A row without any eligible position (or all -inf) commits nothing; callers decide what to do then."""
    scores = conf.masked_fill(~eligible, -math.inf)
    best = scores.max(dim=1, keepdim=True).values
    actual = (best - DINFER_TIE_EPS).clamp(-1000.0, threshold)
    return eligible & (scores >= actual)


def dinfer_block_forwards(commits: List[int], initial_masks: int, expected_tpf: int = 4, max_unroll: int = 4) -> int:
    """Forward count dInfer's BlockDiffusionRunner.decode charges for one block, replayed from our per-step commit
    counts. It runs k = max(min(masks // expected_tpf, max_unroll), 1) forwards back to back before looking at
    the block again, so a block that finishes inside such a batch still pays for the remaining forwards (they
    commit nothing). Token output is unaffected; only the NFE/TPF accounting is."""
    remaining, forwards, i = int(initial_masks), 0, 0
    while remaining > 0:
        k = max(min(remaining // expected_tpf, max_unroll), 1)
        for _ in range(k):
            forwards += 1
            if i < len(commits):
                remaining -= commits[i]
                i += 1
    return forwards


class StepAccounting:
    """Per-forward counters reported in GenerationResult.stats (identical keys for both decoders)."""

    def __init__(self):
        self.nfe = 0
        self.commit_counts: List[int] = []
        self.gate_open_steps = 0
        self.receiver_steps = 0  # steps where some active row had >= 2 eligible positions
        self.skip_reasons: Counter = Counter()
        self.empty_budget_steps = 0
        self.gate_source_committed = 0
        self.gate_source_changed = 0
        self.group_tokens = self.group_committed = self.group_changed = self.fallback_rows = 0
        self.high_conf_group_sizes: Counter = Counter()  # threshold group rule: |{c > tau}| per live row/step

    def step(self, out, ctx, eligible, rows_with_masks, cand, conf, commit, mode: str, tau: float,
             ge: bool = False) -> None:
        self.nfe += 1
        if mode == "threshold":
            above = (conf >= tau) if ge else (conf > tau)
            self.fallback_rows += int((rows_with_masks & ~(eligible & above).any(1)).sum())
        gi = out.guidance
        if gi is not None:
            for r in gi.skip_reason:
                self.skip_reasons[r] += 1
            self.receiver_steps += int(((gi.n_eligible >= 2) & ctx.active_rows).sum())
            n_open = int(gi.gate_open.sum())
            self.gate_open_steps += n_open
            if gi.n_high is not None:
                for b in ctx.active_rows.nonzero(as_tuple=False).flatten().tolist():
                    self.high_conf_group_sizes[int(gi.n_high[b])] += 1
            if gi.group_pos is not None:
                rr, ss = ((gi.group_pos >= 0) & gi.gate_open[:, None]).nonzero(as_tuple=True)
                pp = gi.group_pos[rr, ss]
                self.group_tokens += len(pp)
                self.group_committed += int(commit[rr, pp].sum())
                self.group_changed += int((cand[rr, pp] != gi.group_probs[rr, ss].argmax(-1)).sum())
            if n_open:
                rows = gi.gate_open.nonzero(as_tuple=False).flatten()
                pos = gi.source_pos[rows]
                self.gate_source_committed += int(commit[rows, pos].sum())
                if gi.source_argmax is not None:
                    self.gate_source_changed += int((cand[rows, pos] != gi.source_argmax[rows]).sum())
        self.commit_counts.append(int(commit.sum()))

    def stats(self, **extra) -> dict:
        out = {
            "nfe": self.nfe, "commit_counts": self.commit_counts, "gate_open_steps": self.gate_open_steps,
            "receiver_steps": self.receiver_steps, "skip_reasons": dict(self.skip_reasons),
            "empty_budget_steps": self.empty_budget_steps, "gate_source_committed": self.gate_source_committed,
            "gate_source_identity_changed": self.gate_source_changed,
            "group_tokens": self.group_tokens, "group_committed": self.group_committed,
            "group_identity_changed": self.group_changed, "fallback_rows": self.fallback_rows,
            "high_conf_group_sizes": {str(k): v for k, v in sorted(self.high_conf_group_sizes.items())},
        }
        out.update(extra)
        return out
