"""Build (GuidedModel, tokenizer) from a Config: the real LLaDA2.0-mini checkpoint or the CPU toy model."""

from __future__ import annotations

import torch

from .backend import LLaDA2MiniBackend
from .config import Config
from .model import GuidedModel

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _special_ids(tok, cfg: Config):
    """mask/eos from the tokenizer, cross-checked against explicit config values."""
    mask = tok.convert_tokens_to_ids("<|mask|>")
    eos = tok.eos_token_id
    for name, got, want in (("mask_token_id", mask, cfg.model.mask_token_id), ("eos_token_id", eos, cfg.model.eos_token_id)):
        if got is None or (isinstance(got, int) and got < 0):
            raise ValueError(f"tokenizer has no {name}")
        if want is not None and int(want) != int(got):
            raise ValueError(f"config {name}={want} but the tokenizer says {got}")
    return int(mask), int(eos)


def build_model_and_tokenizer(cfg: Config, device: torch.device):
    torch.manual_seed(cfg.training.seed)
    m = cfg.model
    if m.backend == "fake_llada2":
        from .testing import make_fake_llada2
        base, tok = make_fake_llada2(m.fake_kwargs)
        base = base.to(device)
        mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rev = {"revision": m.revision} if m.revision else {}
        tok = AutoTokenizer.from_pretrained(m.path, trust_remote_code=m.trust_remote_code, **rev)
        mask_id, eos_id = _special_ids(tok, cfg)
        # AutoModel maps to the headless backbone in this repo's config.json: the LM head needs ForCausalLM.
        base = AutoModelForCausalLM.from_pretrained(
            m.path, trust_remote_code=m.trust_remote_code, torch_dtype=DTYPES[m.dtype],
            attn_implementation=m.attn_implementation or "sdpa", **rev).to(device).eval()
    backend = LLaDA2MiniBackend(base, cfg.generation.block_length, mask_token_id=mask_id, eos_token_id=eos_id)
    return GuidedModel(base, backend, cfg.guidance), tok
