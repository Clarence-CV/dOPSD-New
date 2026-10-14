"""Full-vocabulary forward KL in fp32 plus the documented normalisation
(mean over receivers/positions within a state, then mean over valid states)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def forward_kl(log_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) per row. log_q: [n, V] fp32 log-probs (target, no grad); logits_p: [n, V]."""
    if log_q.shape != logits_p.shape:
        raise ValueError(f"shape mismatch {tuple(log_q.shape)} vs {tuple(logits_p.shape)}")
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(dim=-1)


def kl_between_logprobs(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)


def mean_over_states(per_state: List[torch.Tensor]) -> torch.Tensor:
    """Each entry is already the per-position mean of one state; empty list -> error."""
    if not per_state:
        raise ValueError("no valid states")
    return torch.stack(per_state).mean()
