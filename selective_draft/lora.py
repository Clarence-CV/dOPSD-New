"""LoRA on the frozen base, attached through forward hooks so the adapters live OUTSIDE `base` (the frozen-base
checks and checksums keep covering exactly the pretrained weights).

For every target nn.Linear y = W x (+ b) the hook returns y + scale * B(A(x)), A: [r, in], B: [out, r], scale = alpha/r.
A ~ N(0, 1/r) and B = 0, so a fresh adapter leaves the model bit-identical. Adapters are fp32; the update is cast to
the layer's output dtype. `enabled=False` (or the `disabled()` context) turns every hook into a no-op: that is the
frozen Teacher / reference path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("attention.query_key_value", "attention.dense")


def _key(name: str) -> str:
    return name.replace(".", "__")


class LoRAAdapters(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, targets=DEFAULT_TARGETS):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.rank, self.alpha, self.scale = int(rank), float(alpha), float(alpha) / float(rank)
        self.targets = tuple(targets)
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.enabled = True
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.names: List[str] = []
        device = next(base.parameters()).device
        g = torch.Generator().manual_seed(1234)
        for name, mod in base.named_modules():
            if not isinstance(mod, nn.Linear) or not any(name.endswith(t) for t in self.targets):
                continue
            k = _key(name)
            a = torch.randn(self.rank, mod.in_features, generator=g) / self.rank ** 0.5
            self.A[k] = nn.Parameter(a.to(device=device, dtype=torch.float32))
            self.B[k] = nn.Parameter(torch.zeros(mod.out_features, self.rank, device=device, dtype=torch.float32))
            self._handles.append(mod.register_forward_hook(self._make_hook(k)))
            self.names.append(name)
        if not self.names:
            raise ValueError(f"no nn.Linear in the base matches the LoRA targets {self.targets}")

    def _make_hook(self, k: str):
        def hook(module, inputs, output):
            if not self.enabled:
                return None
            x = inputs[0]
            upd = (x.float() @ self.A[k].t()) @ self.B[k].t()
            return output + (upd * self.scale).to(output.dtype)
        return hook

    @contextmanager
    def disabled(self):
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def describe(self) -> Dict[str, object]:
        return {"rank": self.rank, "alpha": self.alpha, "targets": list(self.targets), "n_layers": len(self.names),
                "params": sum(p.numel() for p in self.parameters())}
