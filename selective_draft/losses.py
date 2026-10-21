"""Full-vocabulary forward KL in fp32."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl(log_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) per row. log_q: [n, V] fp32 log-probs (target, no grad); logits_p: [n, V]."""
    if log_q.shape != logits_p.shape:
        raise ValueError(f"shape mismatch {tuple(log_q.shape)} vs {tuple(logits_p.shape)}")
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)
