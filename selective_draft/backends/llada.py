"""LLaDA-8B (modeling_llada.py, checked at revision 08b83a6f):
  base.model.transformer.{wte, blocks, ln_f, ff_out}; block -> (hidden, cache);
  logits = ff_out(ln_f(x)) or F.linear(ln_f(x), wte.weight) when weight_tying; optional scale_logits.
The CPU fake model in selective_draft/testing.py mirrors this layout, so this backend is what the
tests exercise."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Backend


class LLaDABackend(Backend):
    name = "llada"

    def __init__(self, model: nn.Module, mask_token_id: Optional[int] = None, eos_token_id: Optional[int] = None):
        super().__init__(model)
        inner = getattr(model, "model", None)
        transformer = getattr(inner, "transformer", None)
        if transformer is None:
            raise TypeError(f"{type(model).__name__} has no .model.transformer; not a LLaDAModelLM")
        cfg = model.config
        if getattr(cfg, "block_group_size", 1) != 1:
            raise ValueError("block_group_size != 1 is not supported (blocks live in block_groups)")
        if "blocks" not in transformer:
            raise ValueError("transformer has no 'blocks' ModuleList")
        self.transformer = transformer
        self.blocks = transformer["blocks"]
        self.final_norm = transformer["ln_f"]
        self.embedding = transformer["wte"]
        self.weight_tying = bool(getattr(cfg, "weight_tying", False))
        self.scale_logits = bool(getattr(cfg, "scale_logits", False))
        if not self.weight_tying and "ff_out" not in transformer:
            raise ValueError("weight_tying=False but transformer has no ff_out")
        self.ff_out = None if self.weight_tying else transformer["ff_out"]
        self.n_layers = len(self.blocks)
        if self.n_layers != int(cfg.n_layers):
            raise ValueError(f"len(blocks)={self.n_layers} != config.n_layers={cfg.n_layers}")
        self.d_model = int(cfg.d_model)
        self.vocab_size = int(self.embedding.weight.shape[0])
        mid = mask_token_id if mask_token_id is not None else getattr(cfg, "mask_token_id", None)
        eid = eos_token_id if eos_token_id is not None else getattr(cfg, "eos_token_id", None)
        if mid is None or eid is None:
            raise ValueError("mask_token_id / eos_token_id missing; set model.mask_token_id / eos_token_id")
        self.mask_token_id, self.eos_token_id = int(mid), int(eid)
        self.pad_token_id = getattr(cfg, "pad_token_id", None)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.final_norm(hidden)
        if self.weight_tying:
            logits = F.linear(x, self.embedding.weight, None)
        else:
            logits = self.ff_out(x)
        if self.scale_logits:
            logits = logits * (1.0 / math.sqrt(self.d_model))
        return logits

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    def describe(self):
        d = super().describe()
        d.update({"weight_tying": self.weight_tying, "scale_logits": self.scale_logits})
        return d
