"""iLLaDA-8B (modeling_illada.py, checked at revision 5769f049):
  base.model.{embed_tokens, layers, norm}; base.lm_head (tied to embed_tokens); layer -> (hidden, [attn]).
Static check only: this backend has not been run against the real checkpoint yet (plan P6).
The config has no mask token id; pass model.mask_token_id explicitly (official usage: 5)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import Backend


class ILLaDABackend(Backend):
    name = "illada"

    def __init__(self, model: nn.Module, mask_token_id: Optional[int] = None, eos_token_id: Optional[int] = None):
        super().__init__(model)
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "norm"):
            raise TypeError(f"{type(model).__name__} does not look like ILLaDAForCausalLM")
        if not hasattr(model, "lm_head"):
            raise TypeError("ILLaDAForCausalLM without lm_head")
        cfg = model.config
        self.blocks = inner.layers
        self.final_norm = inner.norm
        self.embedding = inner.embed_tokens
        self.lm_head = model.lm_head
        self.n_layers = len(self.blocks)
        self.d_model = int(cfg.hidden_size)
        self.vocab_size = int(self.embedding.weight.shape[0])
        if mask_token_id is None:
            raise ValueError("iLLaDA config has no mask_token_id; set model.mask_token_id (official usage: 5)")
        eid = eos_token_id if eos_token_id is not None else getattr(cfg, "eos_token_id", None)
        if eid is None:
            raise ValueError("eos_token_id missing")
        self.mask_token_id, self.eos_token_id = int(mask_token_id), int(eid)
        self.pad_token_id = getattr(cfg, "pad_token_id", None)
        self.tied = bool(getattr(cfg, "tie_word_embeddings", False))
        if self.tied and self.lm_head.weight.data_ptr() != self.embedding.weight.data_ptr():
            raise ValueError("config says tie_word_embeddings but lm_head and embed_tokens are not the same storage")

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def describe(self):
        d = super().describe()
        d.update({"tie_word_embeddings": self.tied})
        return d
