"""The two trainable modules. Both are rank-bottleneck MLPs with a zero-initialised output
projection, so at initialisation the readout equals the native logit lens and the injector
contributes nothing (exact parity with the base model)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyReadoutAdapter(nn.Module):
    """g(h) = h + U silu(V h);  V random, U zero  ->  identity at init."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, std=d_model ** -0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(F.silu(self.down(h.to(self.down.weight.dtype)))).to(h.dtype)


class MessageInjector(nn.Module):
    """J(h_A, m_A) = U silu(V [h_A; m_A]);  V random, U zero  ->  zero message at init."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(2 * d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, std=(2 * d_model) ** -0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, h_source: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        dt = self.down.weight.dtype
        z = torch.cat([h_source.to(dt), message.to(dt)], dim=-1)
        return self.up(F.silu(self.down(z)))


def soft_message(probs: torch.Tensor, embedding_weight: torch.Tensor, vocab_chunk: int = 16384) -> torch.Tensor:
    """m = sum_v p(v) E(v) with FP32 accumulation over vocabulary chunks.
    probs: [..., V] (any float dtype), embedding_weight: [V, d]. Returns [..., d] fp32."""
    V, d = embedding_weight.shape
    if probs.shape[-1] != V:
        raise ValueError(f"probs vocab {probs.shape[-1]} != embedding rows {V}")
    out = torch.zeros(probs.shape[:-1] + (d,), dtype=torch.float32, device=probs.device)
    wdt = embedding_weight.dtype
    for s in range(0, V, vocab_chunk):
        e = min(V, s + vocab_chunk)
        if wdt == torch.float32:
            out = out + probs[..., s:e].float() @ embedding_weight[s:e]
        else:
            # bf16/fp16 table: matmul in the table dtype (fp32 accumulation inside the kernel), sum chunks in
            # fp32. Avoids materialising a [V, d] fp32 copy every denoising step (plan §4.2).
            out = out + (probs[..., s:e].to(wdt) @ embedding_weight[s:e]).float()
    return out
