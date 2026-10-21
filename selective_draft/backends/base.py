"""Backend contract: where the blocks / final norm / embedding / LM projection of a HF model
live, how a block's output tuple carries the residual hidden state, and how to run the
native forward. Nothing here changes the model's computation path."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class Backend:
    name: str = "base"

    def __init__(self, model: nn.Module):
        self.model = model

    # ---- structure (subclasses set these in __init__) ----
    blocks: nn.ModuleList
    final_norm: nn.Module
    embedding: nn.Embedding
    n_layers: int
    d_model: int
    vocab_size: int
    mask_token_id: int
    eos_token_id: int
    pad_token_id: Optional[int]

    # ---- contract ----
    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """final norm + LM projection exactly as the native forward does it (norm applied once)."""
        raise NotImplementedError

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def block_hidden(output: Any) -> torch.Tensor:
        """Residual hidden state inside a block's return value."""
        if isinstance(output, torch.Tensor):
            return output
        return output[0]

    @staticmethod
    def with_block_hidden(output: Any, hidden: torch.Tensor) -> Any:
        """Same container as `output`, with only the hidden replaced (cache / attentions kept)."""
        if isinstance(output, torch.Tensor):
            return hidden
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        if isinstance(output, list):
            return [hidden] + list(output[1:])
        raise TypeError(f"unsupported block output type {type(output)}")

    def boundary_module(self, early_layer: int) -> nn.Module:
        if not (1 <= early_layer <= self.n_layers):
            raise ValueError(f"early_layer={early_layer} outside 1..{self.n_layers}")
        return self.blocks[early_layer - 1]

    def embedding_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model_class": type(self.model).__name__,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "embedding_rows": tuple(self.embedding.weight.shape),
        }


def resolve_backend(name: str, model: nn.Module, mask_token_id: Optional[int] = None,
                    eos_token_id: Optional[int] = None, block_length: Optional[int] = None) -> Backend:
    if name == "llada":
        from .llada import LLaDABackend
        return LLaDABackend(model, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    if name == "illada":
        from .illada import ILLaDABackend
        return ILLaDABackend(model, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    if name in ("llada2_mini", "fake_llada2"):
        from .llada2_mini import LLaDA2MiniBackend
        if block_length is None:
            raise ValueError("llada2_mini needs the attention block length (generation.block_length)")
        return LLaDA2MiniBackend(model, block_length, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    if name == "fake_llada":
        from .llada import LLaDABackend
        return LLaDABackend(model, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    raise ValueError(f"unknown backend '{name}'")
