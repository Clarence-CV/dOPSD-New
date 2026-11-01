"""Check moe_impl=fused (DMax/VeOmni fused MoE) against the HF expert loop on the real LLaDA2.0-mini.

Three measurements on one loaded model, switching the implementation in place (backend.set_moe_impl):
  logits   fixed recorded states: logit differences, KL(hf || fused), argmax agreement on the current block's MASK rows;
  gradient d loss / d delta for a zero perturbation added after the early layer at the block rows (the path the
           injector trains through): cosine and relative error between the two implementations;
  decode   greedy guidance-off generation of the same questions: identical tokens or first divergence, correctness,
           denoising steps, seconds per step; plus the time of one forward at a few sequence lengths.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .data import Example, tokenize_prompt
from .generation import generate
from .model import GuidedModel
from .scoring import grade
from .training import state_tensors

IMPLS = ("hf_loop", "fused")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def logits_parity(gm: GuidedModel, picks: List[Tuple[dict, dict]]) -> Dict[str, object]:
    be, rows = gm.backend, []
    for roll, step in picks:
        x, _, pos = state_tensors(roll, step, gm.device)
        out = {}
        for impl in IMPLS:
            be.set_moe_impl(impl)
            out[impl] = be.forward_logits(x)[0, pos].float()
        a, b = out["hf_loop"], out["fused"]
        la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
        kl = (la.exp() * (la - lb)).sum(-1)
        rows.append({"example_id": roll["meta"].get("example_id"), "n_positions": len(pos),
                     "max_abs_logit_diff": float((a - b).abs().max()), "mean_abs_logit_diff": float((a - b).abs().mean()),
                     "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                     "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
                     "max_abs_conf_diff": float((la.exp().max(-1).values - lb.exp().max(-1).values).abs().max())})
    n = max(1, len(rows))
    summary = {k: sum(r[k] for r in rows) / n for k in ("mean_abs_logit_diff", "kl_mean", "argmax_agreement")}
    summary.update(max_abs_logit_diff=max(r["max_abs_logit_diff"] for r in rows), kl_max=max(r["kl_max"] for r in rows),
                   max_abs_conf_diff=max(r["max_abs_conf_diff"] for r in rows), n_states=len(rows))
    return {"summary": summary, "rows": rows}


def gradient_parity(gm: GuidedModel, picks: List[Tuple[dict, dict]], layer: int, seed: int = 0) -> Dict[str, object]:
    be, rows = gm.backend, []
    g = torch.Generator().manual_seed(seed)
    for roll, step in picks:
        x, _, pos = state_tensors(roll, step, gm.device)
        target = torch.softmax(torch.randn(len(pos), be.vocab_size, generator=g) * 3.0, -1).to(gm.device)
        grads = {}
        for impl in IMPLS:
            be.set_moe_impl(impl)
            delta = torch.zeros(len(pos), be.d_model, device=gm.device, dtype=torch.float32, requires_grad=True)

            def hook(module, inputs, output):
                h = be.block_hidden(output)
                add = torch.zeros_like(h).index_put((torch.zeros_like(pos), pos), delta.to(h.dtype))
                return be.with_block_hidden(output, h + add)

            handle = be.boundary_module(layer).register_forward_hook(hook)
            try:
                logits = be.forward_logits(x)[0, pos].float()
                loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
                loss.backward()
            finally:
                handle.remove()
            grads[impl] = delta.grad.detach().clone()
        a, b = grads["hf_loop"], grads["fused"]
        rows.append({"example_id": roll["meta"].get("example_id"),
                     "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                     "rel_err": float((a - b).norm() / a.norm().clamp_min(1e-30)), "hf_norm": float(a.norm()),
                     "fused_norm": float(b.norm())})
    n = max(1, len(rows))
    return {"summary": {"layer": layer, "n_states": len(rows), "cosine_min": min(r["cosine"] for r in rows),
                        "cosine_mean": sum(r["cosine"] for r in rows) / n,
                        "rel_err_max": max(r["rel_err"] for r in rows),
                        "rel_err_mean": sum(r["rel_err"] for r in rows) / n},
            "rows": rows}


@torch.no_grad()
def forward_timing(gm: GuidedModel, lengths: List[int], repeats: int = 3) -> Dict[str, Dict[int, float]]:
    be, out = gm.backend, {}
    for impl in IMPLS:
        be.set_moe_impl(impl)
        out[impl] = {}
        for n in lengths:
            x = torch.randint(1000, 100000, (1, n), device=gm.device)
            be.forward_logits(x)  # warm-up (Triton compilation on the first fused call)
            _sync()
            t0 = time.perf_counter()
            for _ in range(repeats):
                be.forward_logits(x)
            _sync()
            out[impl][n] = (time.perf_counter() - t0) / repeats
    return out


def decode_parity(gm: GuidedModel, tok, examples: List[Example], cfg: Config) -> Dict[str, object]:
    be, rows = gm.backend, []
    for ex in examples:
        ids = tokenize_prompt(tok, ex.prompt_text, gm.device)
        res = {}
        for impl in IMPLS:
            be.set_moe_impl(impl)
            _sync()
            t0 = time.perf_counter()
            r = generate(gm, ids, cfg.generation, guidance_mode="off")
            _sync()
            sec = time.perf_counter() - t0
            text = tok.decode(r.completion, skip_special_tokens=True)
            res[impl] = {"tokens": r.completion, "steps": r.steps, "seconds": sec, "s_per_step": sec / max(1, r.steps),
                         "correct": bool(grade(ex.answer_value, text, ex.task)["is_correct"]),
                         "stop_reason": r.stop_reason}
        a, b = res["hf_loop"]["tokens"], res["fused"]["tokens"]
        first = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        row = {"example_id": ex.example_id, "identical": a == b, "first_divergence": first,
               "len_hf": len(a), "len_fused": len(b)}
        for impl in IMPLS:
            row.update({f"{impl}_{k}": v for k, v in res[impl].items() if k != "tokens"})
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    n = max(1, len(rows))
    summary = {"n": len(rows), "identical": sum(r["identical"] for r in rows),
               "correct_hf": sum(r["hf_loop_correct"] for r in rows), "correct_fused": sum(r["fused_correct"] for r in rows),
               "steps_hf": sum(r["hf_loop_steps"] for r in rows) / n, "steps_fused": sum(r["fused_steps"] for r in rows) / n,
               "s_per_step_hf": sum(r["hf_loop_seconds"] for r in rows) / max(1, sum(r["hf_loop_steps"] for r in rows)),
               "s_per_step_fused": sum(r["fused_seconds"] for r in rows) / max(1, sum(r["fused_steps"] for r in rows))}
    summary["speedup_per_step"] = summary["s_per_step_hf"] / max(1e-9, summary["s_per_step_fused"])
    return {"summary": summary, "rows": rows}
