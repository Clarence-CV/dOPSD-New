"""Shared helpers for the CPU tests (fake LLaDA-shaped model, controlled-logit model)."""
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selective_draft.backends import resolve_backend  # noqa: E402
from selective_draft.config import Config  # noqa: E402
from selective_draft.model import GuidedModel, StepContext  # noqa: E402
from selective_draft.testing import FakeConfig, FakeLLaDA, FakeTokenizer, make_fake  # noqa: E402

MASK, EOS, PAD, V, D, L = 3, 2, 0, 64, 32, 4


def fake_config(**over) -> Config:
    cfg = Config.load(os.path.join(ROOT, "configs", "selective_draft", "fake_cpu.json"))
    return cfg.apply_overrides([f"{k}={v}" for k, v in over.items()]).validate()


def build(cfg: Config = None, seed=0, **fake_kw):
    cfg = cfg or fake_config()
    kw = dict(cfg.model.fake_kwargs)
    kw.update(fake_kw)
    base, tok = make_fake(kw)
    torch.manual_seed(seed)
    backend = resolve_backend("fake_llada", base)
    return GuidedModel(base, backend, cfg.guidance), tok, cfg


class ControlledFake(FakeLLaDA):
    """Returns logits from a user-supplied function f(input_ids) -> [B, N, V] (mode='off' paths only)."""

    def __init__(self, cfg, fn):
        super().__init__(cfg)
        self.fn = fn
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, **kw):
        self.calls.append((input_ids.clone(), None if attention_mask is None else attention_mask.clone()))
        return SimpleNamespace(logits=self.fn(input_ids))


def controlled(fn, cfg: Config = None):
    cfg = cfg or fake_config()
    fc = FakeConfig(**cfg.model.fake_kwargs)
    base = ControlledFake(fc, fn)
    return GuidedModel(base, resolve_backend("fake_llada", base), cfg.guidance), FakeTokenizer(fc), cfg


def peaked_logits(probs_fn):
    """probs_fn(input_ids) -> [B, N] desired top-1 probability for token (pos % (V-4)) + 4; returns logits."""
    def fn(ids):
        B, N = ids.shape
        p = probs_fn(ids).clamp(1e-4, 1 - 1e-4)
        tok = (torch.arange(N) % (V - 4)) + 4
        logits = torch.zeros(B, N, V)
        # top-1 prob p, the rest spread uniformly over the V-2 other non-MASK tokens; MASK excluded so the
        # decoder's forbid_mask_candidate renormalisation is a no-op and p is exact
        logits += torch.log((1 - p) / (V - 2)).unsqueeze(-1)
        logits[torch.arange(B)[:, None], torch.arange(N)[None], tok[None]] = torch.log(p)
        logits[..., MASK] = -float("inf")
        return logits
    return fn


def make_ctx(x, bs, be, mask_id=MASK, active=None):
    B, N = x.shape
    elig = torch.zeros(B, N, dtype=torch.bool)
    elig[:, bs:be] = x[:, bs:be] == mask_id
    act = torch.ones(B, dtype=torch.bool) if active is None else active
    return StepContext(elig, act, torch.full((B,), bs), torch.full((B,), be), torch.zeros(B, dtype=torch.long))


def run_all(mod_globals):
    fails = 0
    for name, fn in sorted(mod_globals.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print(f"  FAIL {name}: {e}")
                traceback.print_exc()
    print("FAILED" if fails else "ALL PASSED", f"({fails} failures)")
    sys.exit(1 if fails else 0)
