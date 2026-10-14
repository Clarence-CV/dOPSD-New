"""Build (GuidedModel, tokenizer) from a Config: real HF checkpoint or the CPU fake."""

from __future__ import annotations

import torch

from .backends import resolve_backend
from .config import Config
from .model import GuidedModel

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model_and_tokenizer(cfg: Config, device: torch.device):
    torch.manual_seed(cfg.training.seed)
    m = cfg.model
    if m.backend == "fake_llada":
        from .testing import make_fake
        base, tok = make_fake(m.fake_kwargs)
        base = base.to(device)
    else:
        from transformers import AutoModel, AutoTokenizer
        kw = dict(trust_remote_code=m.trust_remote_code, torch_dtype=DTYPES[m.dtype])
        if m.revision:
            kw["revision"] = m.revision
        base = AutoModel.from_pretrained(m.path, **kw).to(device).eval()
        tok = AutoTokenizer.from_pretrained(m.path, trust_remote_code=m.trust_remote_code,
                                            **({"revision": m.revision} if m.revision else {}))
    backend = resolve_backend(m.backend, base, mask_token_id=m.mask_token_id, eos_token_id=m.eos_token_id)
    gm = GuidedModel(base, backend, cfg.guidance)
    return gm, tok
