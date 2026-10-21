"""CPU stand-ins so the real backend, hook, decoder and trainers run end-to-end without a pretrained model:
a tiny randomly initialised LLaDA2MoeModelLM built from the vendored official remote code (real block-causal mask
handling, real MoE routing) plus a word-level hashing tokenizer. Not a language model."""

from __future__ import annotations

import hashlib
from typing import List, Optional

import torch


class FakeTokenizer:
    """Word-level hashing tokenizer with a trivial chat template."""

    def __init__(self, vocab_size=64, mask_token_id=3, eos_token_id=2, pad_token_id=0):
        self.vocab_size = vocab_size
        self.mask_token_id, self.eos_token_id, self.pad_token_id = mask_token_id, eos_token_id, pad_token_id
        self.reserved = 4
        self._id2word = {}

    def _wid(self, w: str) -> int:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        wid = self.reserved + h % (self.vocab_size - self.reserved)
        self._id2word.setdefault(wid, w)
        return wid

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def encode(self, text: str) -> List[int]:
        return [self._wid(w) for w in text.split()]

    def __call__(self, text, **kw):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for i in ids:
            i = int(i)
            if i == self.eos_token_id:
                words.append("" if skip_special_tokens else "<eos>")
                if skip_special_tokens:
                    break
            elif i < self.reserved:
                if not skip_special_tokens:
                    words.append(f"<{i}>")
            else:
                words.append(self._id2word.get(i, f"w{i}"))
        return " ".join(w for w in words if w)


# ----------------------------------------------------------------------------- LLaDA2.0-mini stand-in
def make_fake_llada2(kwargs: Optional[dict] = None):
    """Tiny randomly initialised LLaDA2MoeModelLM built from the vendored official remote code (real block
    mask handling, real MoE routing incl. the no_grad moe_infer), plus the word-level FakeTokenizer.
    Needs the Transformers version the remote code targets (4.57)."""
    from .vendor.llada2_mini.configuration_llada2_moe import LLaDA2MoeConfig
    from .vendor.llada2_mini.modeling_llada2_moe import LLaDA2MoeModelLM
    kw = dict(kwargs or {})
    seed = kw.pop("seed", 0)
    logit_scale = kw.pop("logit_scale", 30.0)
    attn = kw.pop("attn_implementation", "sdpa")
    vocab = kw.pop("vocab_size", 64)
    mask_id, eos_id, pad_id = kw.pop("mask_token_id", 3), kw.pop("eos_token_id", 2), kw.pop("pad_token_id", 0)
    params = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, moe_intermediate_size=16,
                  num_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1, num_shared_experts=1,
                  first_k_dense_replace=1, partial_rotary_factor=0.5, rotary_dim=4, max_position_embeddings=1024,
                  pad_token_id=pad_id, use_bias=False, use_qkv_bias=False, embedding_dropout=0.0,
                  attention_dropout=0.0, output_dropout=0.0, rope_theta=10000.0, routed_scaling_factor=2.5,
                  tie_word_embeddings=False, use_cache=False)
    params.update(kw)
    cfg = LLaDA2MoeConfig(**params)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = LLaDA2MoeModelLM(cfg).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(logit_scale)  # peaked enough that threshold decoding commits several tokens
        model.model.word_embeddings.weight.mul_(10.0)
    tok = FakeTokenizer(vocab_size=vocab, mask_token_id=mask_id, eos_token_id=eos_id, pad_token_id=pad_id)
    return model, tok
