"""Frozen-teacher targets for receivers B given a source A.

reveal_view only changes the token at A (prompt, other MASKs, attention mask, positions untouched).
Modes (see doc/Selective_Draft_OPSD.md §4 for the single-source default and its limits):
  realized  q_B^{a}  with a = the token A actually receives in the rollout    (1 forward)
  argmax    q_B^{v*} with v* = argmax r_A                                      (1 forward)
  mc        sum_j w_j q_B^{v_j},  v_j ~ r_A with replacement, w_j = counts/K    (<= K forwards)
  topk      sum_{v in topK} r_A(v)/mass q_B^{v}                                 (K forwards)
Probabilities are mixed, never logits. A candidate equal to MASK leaves the view unchanged
(mathematically 'not revealed') and is counted in info['mask_candidate_events']."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import TeacherConfig
from .model import GuidanceSpec, GuidedModel, StepContext


@dataclass
class TeacherTarget:
    log_q: torch.Tensor  # [n_receivers, V] fp32 log-probs
    info: Dict[str, object] = field(default_factory=dict)


def reveal_view(state_row: torch.Tensor, pos: int, token: int) -> torch.Tensor:
    view = state_row.clone()
    view[pos] = int(token)
    return view


@torch.no_grad()
def log_probs_at(gm: GuidedModel, views: torch.Tensor, attn: torch.Tensor, receivers: torch.Tensor,
                 batch_size: int = 1) -> torch.Tensor:
    """views [k, N] -> [k, n, V] fp32 log-probs at receiver positions (native base forward)."""
    outs = []
    for s in range(0, views.shape[0], batch_size):
        v = views[s:s + batch_size]
        logits = gm.base_forward(v, attn[: v.shape[0]])
        outs.append(F.log_softmax(logits[:, receivers].float(), dim=-1))
    return torch.cat(outs, dim=0)


def candidate_set(mode: str, r_A: Optional[torch.Tensor], realized_token: Optional[int], cfg: TeacherConfig,
                  rng: Optional[torch.Generator]) -> Tuple[List[int], torch.Tensor, Dict[str, object]]:
    """-> (candidates, weights fp32 summing to 1, info)."""
    info: Dict[str, object] = {"mode": mode}
    if mode == "realized":
        if realized_token is None:
            raise ValueError("teacher.mode=realized needs the realized token at A")
        return [int(realized_token)], torch.ones(1), info
    if r_A is None:
        raise ValueError(f"teacher.mode={mode} needs r_A")
    r = r_A.float().cpu()
    if mode == "argmax":
        return [int(r.argmax())], torch.ones(1), info
    if mode == "mc":
        draws = torch.multinomial(r, cfg.mc_samples, replacement=True, generator=rng)
        uniq, counts = torch.unique(draws, return_counts=True)
        info.update({"mc_samples": cfg.mc_samples, "unique_candidates": int(uniq.numel())})
        return uniq.tolist(), counts.float() / cfg.mc_samples, info
    if mode == "topk":
        k = min(cfg.topk, r.numel())
        p, idx = r.topk(k)
        mass = float(p.sum())
        info.update({"topk": k, "retained_mass": mass, "truncation_error_proxy": 1.0 - mass})
        return idx.tolist(), p / p.sum(), info
    raise ValueError(mode)


@torch.no_grad()
def teacher_targets(gm: GuidedModel, state_row: torch.Tensor, attn_row: torch.Tensor, source_pos: int,
                    receivers: torch.Tensor, r_A: Optional[torch.Tensor], realized_token: Optional[int],
                    cfg: TeacherConfig, rng: Optional[torch.Generator] = None, mode: Optional[str] = None) -> TeacherTarget:
    mode = cfg.mode if mode is None else mode
    cands, weights, info = candidate_set(mode, r_A, realized_token, cfg, rng)
    mask_id = gm.backend.mask_token_id
    info["mask_candidate_events"] = sum(1 for c in cands if c == mask_id)
    info["candidates"] = cands
    info["weights"] = weights.tolist()
    views = torch.stack([reveal_view(state_row, source_pos, c) for c in cands], dim=0)
    attn = attn_row.unsqueeze(0).expand(len(cands), -1)
    log_qv = log_probs_at(gm, views, attn, receivers, cfg.candidate_batch_size)  # [k, n, V]
    info["teacher_forwards"] = len(cands)
    log_w = weights.log().to(log_qv.device).view(-1, 1, 1)
    log_q = torch.logsumexp(log_w + log_qv, dim=0)  # probability mixture
    return TeacherTarget(log_q, info)


@torch.no_grad()
def unconditional_log_probs(gm: GuidedModel, state_row: torch.Tensor, attn_row: torch.Tensor,
                            receivers: torch.Tensor) -> torch.Tensor:
    return log_probs_at(gm, state_row.unsqueeze(0), attn_row.unsqueeze(0), receivers)[0]


@torch.no_grad()
def oracle_patch_log_probs(gm: GuidedModel, state_row: torch.Tensor, attn_row: torch.Tensor, ctx: StepContext,
                           source_pos: int, token: int, receivers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Activation-patch oracle: replace only h_A^ell by its value under the revealed view, run the native
    tail. Returns (log-probs at receivers [n, V], delta [d])."""
    x = state_row.unsqueeze(0)
    a = attn_row.unsqueeze(0)
    h_mask = gm(x, a, context=ctx, mode="off", capture_hidden=True).hidden_early[0, source_pos]
    xr = reveal_view(state_row, source_pos, token).unsqueeze(0)
    h_rev = gm(xr, a, context=ctx, mode="off", capture_hidden=True).hidden_early[0, source_pos]
    delta = (h_rev - h_mask).unsqueeze(0)
    spec = GuidanceSpec(source_pos=torch.tensor([source_pos], device=x.device),
                        gate_open=torch.tensor([True], device=x.device), override_delta=delta)
    out = gm(x, a, context=ctx, mode="single_soft", spec=spec)
    return F.log_softmax(out.logits[0, receivers].float(), dim=-1), delta[0]
