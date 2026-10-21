"""Shared helpers for the CPU tests: a tiny LLaDA2MoeModelLM built from the vendored official remote code
(needs Transformers 4.57), a controlled-logits variant, and a plain-script runner."""
import math
import os
import sys
import types
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selective_draft.config import Config  # noqa: E402
from selective_draft.loading import build_model_and_tokenizer  # noqa: E402

MASK, EOS, V, BL = 3, 2, 64, 8


def mini_config(**over) -> Config:
    """Overrides as {"section.field": value}; string values must be JSON (e.g. '"group_soft"')."""
    cfg = Config.load(os.path.join(ROOT, "configs", "fake_llada2_cpu.json"))
    return cfg.apply_overrides([f"{k}={v}" for k, v in over.items()]).validate()


def build(cfg=None, **fake_kw):
    cfg = cfg or mini_config()
    cfg.model.fake_kwargs = dict(cfg.model.fake_kwargs, **fake_kw)
    gm, tok = build_model_and_tokenizer(cfg, torch.device("cpu"))
    return gm, tok, cfg


def controlled(fn, cfg=None):
    """Real backend/decoder around static logits fn(input_ids) -> [B, N, V]. Returns (gm, tok, cfg, calls)."""
    gm, tok, cfg = build(cfg)
    calls = []

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        calls.append(input_ids.clone())
        return SimpleNamespace(logits=fn(input_ids))
    gm.base.forward = types.MethodType(forward, gm.base)
    return gm, tok, cfg, calls


def conf_logits(plan, default=(10, 0.99)):
    """Static logits: absolute position -> (token, confidence). Others share the remaining mass; MASK is impossible."""
    def fn(ids):
        B, N = ids.shape
        logits = torch.zeros(B, N, V)
        for pos in range(N):
            tok, c = plan.get(pos, default)
            logits[:, pos, tok] = math.log(c * (V - 2) / (1 - c))
        logits[..., MASK] = -30.0
        return logits
    return fn


def prompt(P, seed=0):
    torch.manual_seed(seed)
    return torch.randint(4, V, (1, P))


def run_all(mod_globals):
    fails = 0
    for name, fn in sorted(mod_globals.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print(f"  FAIL {name}: {e}", flush=True)
                traceback.print_exc()
    print("FAILED" if fails else "ALL PASSED", f"({fails} failures)")
    sys.exit(1 if fails else 0)
