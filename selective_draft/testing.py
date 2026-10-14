"""CPU-only stand-ins with the LLaDA module layout, so the real LLaDABackend, hook, decoder,
teacher and trainers run end-to-end without a pretrained model. Not a language model."""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeConfig:
    def __init__(self, n_layers=4, d_model=32, vocab_size=64, mask_token_id=3, eos_token_id=2, pad_token_id=0,
                 weight_tying=False, scale_logits=False, max_positions=512, logit_scale=6.0, abs_pos=True):
        self.n_layers, self.d_model, self.vocab_size = n_layers, d_model, vocab_size
        self.mask_token_id, self.eos_token_id, self.pad_token_id = mask_token_id, eos_token_id, pad_token_id
        self.weight_tying, self.scale_logits, self.block_group_size = weight_tying, scale_logits, 1
        self.max_positions, self.logit_scale, self.abs_pos = max_positions, logit_scale, abs_pos


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class FakeBlock(nn.Module):
    """Pre-norm bidirectional attention + SiLU MLP; returns (hidden, None) like LLaDABlock."""

    def __init__(self, d):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ff_in = nn.Linear(d, 4 * d, bias=False)
        self.ff_out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False):
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        att = q @ k.transpose(1, 2) / math.sqrt(x.shape[-1])
        if attention_bias is not None:
            att = att + attention_bias
        x = x + self.out(torch.softmax(att.float(), dim=-1).to(x.dtype) @ v)
        x = x + self.ff_out(F.silu(self.ff_in(self.ff_norm(x))))
        return x, None


class FakeInner(nn.Module):
    def __init__(self, cfg: FakeConfig):
        super().__init__()
        d, V = cfg.d_model, cfg.vocab_size
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(V, d),
            "wpe": nn.Embedding(cfg.max_positions, d),
            "blocks": nn.ModuleList([FakeBlock(d) for _ in range(cfg.n_layers)]),
            "ln_f": RMSNorm(d),
        })
        if not cfg.weight_tying:
            self.transformer["ff_out"] = nn.Linear(d, V, bias=False)
            with torch.no_grad():  # sharpness lives in the head weights, so project() == native logits
                self.transformer["ff_out"].weight.mul_(cfg.logit_scale)
        else:
            with torch.no_grad():
                self.transformer["wte"].weight.mul_(cfg.logit_scale ** 0.5)

    def forward(self, input_ids, attention_mask=None):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device)
        x = self.transformer["wte"](input_ids)
        if self.cfg.abs_pos:
            x = x + self.transformer["wpe"](pos)[None]
        bias = None
        if attention_mask is not None and bool((attention_mask == 0).any()):
            bias = (1.0 - attention_mask.float())[:, None, :] * torch.finfo(torch.float32).min
        for blk in self.transformer["blocks"]:
            x, _ = blk(x, attention_bias=bias)
        x = self.transformer["ln_f"](x)
        if self.cfg.weight_tying:
            logits = F.linear(x, self.transformer["wte"].weight)
        else:
            logits = self.transformer["ff_out"](x)
        if self.cfg.scale_logits:
            logits = logits * (1 / math.sqrt(self.cfg.d_model))
        return logits


class FakeLLaDA(nn.Module):
    def __init__(self, cfg: FakeConfig, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = cfg
        self.model = FakeInner(cfg)
        self.dtype = torch.float32

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return SimpleNamespace(logits=self.model(input_ids, attention_mask))


class FakeTokenizer:
    """Word-level hashing tokenizer with left padding and a trivial chat template."""

    def __init__(self, cfg: FakeConfig):
        self.vocab_size = cfg.vocab_size
        self.mask_token_id, self.eos_token_id, self.pad_token_id = cfg.mask_token_id, cfg.eos_token_id, cfg.pad_token_id
        self.reserved = 4
        self.padding_side = "left"
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

    def __call__(self, texts, padding=True, return_tensors="pt", **kw):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        input_ids = torch.full((len(ids), L), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(ids), L), dtype=torch.long)
        for r, seq in enumerate(ids):
            input_ids[r, L - len(seq):] = torch.tensor(seq)
            attn[r, L - len(seq):] = 1
        return {"input_ids": input_ids, "attention_mask": attn}

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


def make_fake(kwargs: Optional[dict] = None):
    cfg = FakeConfig(**(kwargs or {}))
    return FakeLLaDA(cfg), FakeTokenizer(cfg)
