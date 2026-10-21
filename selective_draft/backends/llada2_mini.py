"""LLaDA2.0-mini (modeling_llada2_moe.py, revision dad945cac317da394b390f82c7b40691d8a881ed):
  base.model.{word_embeddings, layers, norm}; base.lm_head (untied); layer -> (hidden, ...).
  logits = lm_head(norm(h)).float(), predicted at the SAME position as the MASK (no AR shift).

Attention is block-causal over absolute 32-position cells counted from position 0: a query in block j sees
keys in blocks <= j, bidirectionally inside its own block. The HF forward only accepts that as a 4D additive
mask of shape [B, 1, N, N]; this backend builds it from positions (official generate builds the same tensor
as log(tril(ones)) in bf16). Inputs are batch-1 without padding: left padding would move the block grid.

MoE: in eval mode the official SparseMoeBlock calls `moe_infer`, which is decorated with @torch.no_grad, so
routed experts contribute no input gradient (plan §7.4). Every MoE layer gets an instance-level
`moe_infer` with the same arithmetic minus no_grad and in-place autograd hazards. Under no_grad it produces
bit-identical outputs (checked in tests), so the Teacher, rollouts and the frozen model are unchanged.
"""

from __future__ import annotations

import types
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import Backend

MINI_MASK_ID = 156895  # <|mask|>
MINI_EOS_ID = 156892  # <|endoftext|> (also pad)
MINI_ROLE_END_ID = 156900  # <|role_end|>; recorded, not a stop token


def _moe_infer_autograd(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
    """Official LLaDA2MoeSparseMoeBlock.moe_infer without @torch.no_grad: same expert order, same dtype casts,
    same summation; the scatter `new_x[idxs] = outs` becomes a gather by the inverse permutation and the
    in-place mul_ becomes out-of-place (identical values)."""
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        num_tokens = int(num_tokens)
        if num_tokens == 0:
            continue
        end_idx = start_idx + num_tokens
        outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx]).to(x.device))
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    inverse = torch.empty_like(idxs)
    inverse[idxs] = torch.arange(idxs.numel(), device=idxs.device)
    new_x = outs[inverse]
    return ((new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1).type(new_x.dtype))


class LLaDA2MiniBackend(Backend):
    name = "llada2_mini"
    attention = "block_causal"

    def __init__(self, model: nn.Module, block_length: int, mask_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None, role_end_token_id: Optional[int] = None):
        super().__init__(model)
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "word_embeddings"):
            raise TypeError(f"{type(model).__name__} does not look like LLaDA2MoeModelLM "
                            "(load with AutoModelForCausalLM; AutoModel maps to the headless backbone)")
        if not hasattr(model, "lm_head"):
            raise TypeError("LLaDA2MoeModelLM without lm_head")
        cfg = model.config
        self.blocks = inner.layers
        self.final_norm = inner.norm
        self.embedding = inner.word_embeddings
        self.lm_head = model.lm_head
        self.n_layers = len(self.blocks)
        if self.n_layers != int(cfg.num_hidden_layers):
            raise ValueError(f"len(layers)={self.n_layers} != num_hidden_layers={cfg.num_hidden_layers}")
        self.d_model = int(cfg.hidden_size)
        self.vocab_size = int(self.embedding.weight.shape[0])
        if int(block_length) < 1:
            raise ValueError("block_length must be positive")
        self.block_length = int(block_length)
        self.mask_token_id = int(MINI_MASK_ID if mask_token_id is None else mask_token_id)
        self.eos_token_id = int(MINI_EOS_ID if eos_token_id is None else eos_token_id)
        self.role_end_token_id = int(MINI_ROLE_END_ID if role_end_token_id is None else role_end_token_id)
        self.pad_token_id = getattr(cfg, "pad_token_id", None)
        self.tied = bool(getattr(cfg, "tie_word_embeddings", False))
        self.attn_implementation = getattr(cfg, "_attn_implementation", None)
        self._mask_cache: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
        self.moe_layers = []
        for i, layer in enumerate(self.blocks):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "moe_infer"):
                mlp.moe_infer = types.MethodType(_moe_infer_autograd, mlp)
                self.moe_layers.append(i)

    # ------------------------------------------------------------------ contract
    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden))

    def block_mask(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """[1, 1, n, n] additive mask: 0 where block(key) <= block(query), -inf elsewhere."""
        key = (n, str(device), dtype)
        m = self._mask_cache.get(key)
        if m is None:
            blk = torch.arange(n, device=device) // self.block_length
            allowed = blk[None, :] <= blk[:, None]
            m = torch.zeros(n, n, dtype=dtype, device=device).masked_fill(~allowed, float("-inf"))[None, None]
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N = input_ids.shape
        if attention_mask is not None and not bool((attention_mask != 0).all()):
            raise ValueError("llada2_mini inputs must be unpadded (padding would shift the absolute block grid)")
        dtype = self.embedding.weight.dtype
        mask = self.block_mask(N, input_ids.device, dtype).expand(B, 1, N, N)
        pos = torch.arange(N, device=input_ids.device)[None].expand(B, N)
        return self.model(input_ids=input_ids, attention_mask=mask, position_ids=pos).logits

    # ------------------------------------------------------------------ prefix KV cache (dInfer-style)
    # The HF forward rejects q_len != kv_len, so the decoder layers are called directly with a DynamicCache.
    # Cache invariant used by block_generation: it holds K/V of positions [0, keep) computed from CLEAN tokens;
    # every window forward appends its own K/V and is then cropped back to `keep`.
    def new_cache(self):
        from transformers.cache_utils import DynamicCache
        return DynamicCache()

    def forward_window(self, input_ids: torch.Tensor, start: int, cache, keep: int,
                       attention_mask: Optional[torch.Tensor] = None, last_n: Optional[int] = None,
                       return_logits: bool = True) -> Optional[torch.Tensor]:
        """Run positions [start, start+q) against the cached prefix; return fp32 logits of the last `last_n` rows
        (all rows if None). attention_mask: additive [1,1,q,cached+q] or None (= full attention, correct for a
        block that sees its whole prefix and itself). The cache is cropped to `keep` positions afterwards."""
        inner = self.model.model
        B, q = input_ids.shape
        pos = torch.arange(start, start + q, device=input_ids.device)[None].expand(B, q)
        h = inner.word_embeddings(input_ids)
        pe = inner.rotary_emb(h, pos)
        for layer in self.blocks:
            h = layer(h, attention_mask=attention_mask, position_ids=pos, past_key_value=cache,
                      position_embeddings=pe, use_cache=True)[0]
        cache.crop(keep)
        if not return_logits:
            return None
        h = self.final_norm(h)
        if last_n is not None:
            h = h[:, -last_n:]
        return self.lm_head(h).float()

    def describe(self):
        d = super().describe()
        d.update({"tie_word_embeddings": self.tied, "block_length": self.block_length,
                  "attention": "block_causal_absolute_zero", "attn_implementation": self.attn_implementation,
                  "role_end_token_id": self.role_end_token_id, "moe_layers": self.moe_layers,
                  "moe_infer": "autograd_equivalent (official arithmetic without no_grad)"})
        return d
