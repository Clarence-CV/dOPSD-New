"""One decoding core for collection, training rollouts and evaluation.
Threshold mode (adaptive steps) and fixed-budget mode share the loop; only the commit rule differs.
Left-padded prompts; attention_mask 0 = padding only (MASK tokens stay attendable)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import GenerationConfig
from .model import GuidedModel, StepContext


@dataclass
class GenerationResult:
    sequences: torch.Tensor  # [B, N_final] (prompt + generated window)
    prompt_len: int
    computed_lengths: List[int]  # generated positions actually decoded (block granularity)
    effective_lengths: List[int]  # up to (excluding) the first EOS in the generated region
    stop_reasons: List[str]  # eos | length
    row_steps: List[int]  # forwards in which this row had eligible positions
    stats: Dict[str, object] = field(default_factory=dict)

    def completion_ids(self, b: int, effective: bool = True) -> List[int]:
        n = self.effective_lengths[b] if effective else self.computed_lengths[b]
        return self.sequences[b, self.prompt_len:self.prompt_len + n].tolist()


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """LLaDA's integer schedule: [B, steps]; remainder goes to the earliest steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = base.expand(-1, steps).clone()
    idx = torch.arange(steps, device=mask_index.device)
    out = out + (idx.unsqueeze(0) < remainder).long()
    return out.to(torch.int64)


def choose_tokens(logits: torch.Tensor, temperature: float, rng: Optional[torch.Generator],
                  forbidden_token: Optional[int] = None):
    """Candidate per position and its raw probability under the un-noised distribution.
    temperature>0: Gumbel-max on logits + T*g (same argmax law as the legacy exp(logits)/(-log u)^T).
    forbidden_token (usually MASK) cannot be selected, but confidence is always measured
    in the original model distribution, without renormalising away forbidden probability."""
    lf = logits.float()
    probs = F.softmax(lf, dim=-1)
    if forbidden_token is not None:
        lf = lf.clone()
        lf[..., forbidden_token] = -math.inf
    if temperature > 0.0:
        u = torch.rand(lf.shape, generator=rng, device=lf.device, dtype=torch.float32)
        neg_log_u = -torch.log(u.clamp_min(1e-20))
        g = -torch.log(neg_log_u.clamp_min(1e-20))  # standard Gumbel(0, 1)
        cand = torch.argmax(lf + temperature * g, dim=-1)
    else:
        cand = torch.argmax(lf, dim=-1)
    conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
    return cand, conf


def select_commits(eligible: torch.Tensor, conf: torch.Tensor, mode: str, threshold: float,
                   k_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Shared commit rule for generation and same-state counterfactual diagnostics."""
    scores = conf.masked_fill(~eligible, -math.inf)
    live = eligible.any(dim=1)
    commit = torch.zeros_like(eligible)
    if mode == "threshold":
        commit = eligible & (scores > threshold)
        for b in (live & ~commit.any(dim=1)).nonzero(as_tuple=False).flatten().tolist():
            commit[b, int(scores[b].argmax())] = True
    elif mode == "fixed_budget":
        if k_t is None:
            raise ValueError("fixed_budget requires k_t")
        for b in live.nonzero(as_tuple=False).flatten().tolist():
            k = min(int(k_t[b]), int(eligible[b].sum()))
            if k > 0:
                commit[b, torch.topk(scores[b], k=k).indices] = True
    else:
        raise ValueError(f"unknown commit mode {mode}")
    return commit


@torch.no_grad()
def baseline_block_commitments(gm: GuidedModel, x: torch.Tensor, attention_mask: torch.Tensor,
                               context: StepContext, cfg: GenerationConfig, positions: torch.Tensor,
                               threshold: float, initial_logits: Optional[torch.Tensor] = None) -> dict:
    """Continue a single recorded state with guidance off until the requested positions commit.

    Only the current block is needed: EOS stopping happens after it is fully decoded. This
    diagnostic is deterministic; sampled continuations need an explicit coupling of RNG states.
    Returns absolute position -> (committed token, number of steps after the recorded step).
    """
    if cfg.mode != "threshold" or cfg.temperature != 0:
        raise ValueError("baseline_block_commitments requires greedy threshold decoding")
    if x.shape[0] != 1:
        raise ValueError("counterfactual continuation expects one state")
    x = x.clone()
    remaining = set(positions.tolist())
    result = {}
    for step in range(int(context.eligible_mask.sum())):
        eligible = context.eligible_mask & (x == gm.backend.mask_token_id)
        if step == 0 and initial_logits is not None:
            logits = initial_logits
        else:
            with _autocast(cfg, x.device):
                logits = gm(x, attention_mask, mode="off").logits
        cand, conf = choose_tokens(logits, 0.0, None,
                                   gm.backend.mask_token_id if cfg.forbid_mask_candidate else None)
        commit = select_commits(eligible, conf, "threshold", threshold)
        if bool((cand[commit] == gm.backend.mask_token_id).any()) or not bool(torch.isfinite(conf[commit]).all()):
            raise RuntimeError("invalid candidate in baseline continuation")
        for pos in commit[0].nonzero(as_tuple=False).flatten().tolist():
            if pos in remaining:
                result[pos] = (int(cand[0, pos]), step)
                remaining.remove(pos)
        x = torch.where(commit, cand, x)
        if not remaining:
            return result
    raise RuntimeError("baseline continuation did not commit all requested positions")


def _autocast(cfg: GenerationConfig, device: torch.device):
    if cfg.autocast == "none" or device.type != "cuda":
        return torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else _NullCtx()
    dt = torch.bfloat16 if cfg.autocast == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dt)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@torch.no_grad()
def generate(gm: GuidedModel, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor,
             cfg: GenerationConfig, guidance_mode: Optional[str] = None, recorder=None,
             sample_rng: Optional[torch.Generator] = None, commit_threshold: Optional[float] = None,
             steps: Optional[int] = None, initial_state: Optional[torch.Tensor] = None,
             start_block: int = 0, initial_step: int = 0) -> GenerationResult:
    """Decode from a prompt, or continue an actual threshold-decoding state without remasking known tokens."""
    backend = gm.backend
    device = prompt_ids.device
    MASK, EOS = backend.mask_token_id, backend.eos_token_id
    B, P = prompt_ids.shape
    G, BL = cfg.max_new_tokens, cfg.block_length
    num_blocks = G // BL
    tau = cfg.commit_threshold if commit_threshold is None else float(commit_threshold)
    total_steps = cfg.steps if steps is None else int(steps)
    if cfg.mode == "fixed_budget":
        if total_steps % num_blocks:
            raise ValueError(f"steps={total_steps} not a multiple of num_blocks={num_blocks}")
        spb = total_steps // num_blocks
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt_attention_mask must match prompt_ids")

    if not 0 <= start_block < num_blocks:
        raise ValueError("start_block outside generation window")
    init_len = G if cfg.window_mode == "fixed" else (start_block + 1) * BL
    if initial_state is None:
        if start_block != 0 or initial_step != 0:
            raise ValueError("resumed block/step requires initial_state")
        x = torch.cat([prompt_ids, torch.full((B, init_len), MASK, dtype=torch.long, device=device)], dim=1)
    else:
        if cfg.mode != "threshold":
            raise ValueError("state continuation currently requires threshold decoding")
        if initial_state.shape != (B, P + init_len) or not torch.equal(initial_state[:, :P], prompt_ids):
            raise ValueError("initial_state must preserve prompt and match the configured window")
        if bool((initial_state[:, P:P + start_block * BL] == MASK).any()):
            raise ValueError("completed blocks in initial_state must not contain MASK")
        if cfg.stop_mode == "block_end" and bool((initial_state[:, P:P + start_block * BL] == EOS).any()):
            raise ValueError("cannot continue beyond an already completed EOS block")
        x = initial_state.to(device=device, dtype=torch.long).clone()
    attn = torch.cat([prompt_attention_mask.to(torch.long),
                      torch.ones(B, init_len, dtype=torch.long, device=device)], dim=1)
    active = torch.ones(B, dtype=torch.bool, device=device)
    stop_reasons = ["length"] * B
    computed = [0] * B
    row_steps = [0] * B
    nfe = 0
    commit_counts: List[int] = []
    gate_open_steps = 0
    receiver_steps = 0  # steps where some active row had >= 2 eligible positions
    skip_reasons: Counter = Counter()
    empty_budget_steps = 0
    gate_source_committed = 0
    gate_source_changed = 0
    group_tokens = group_committed = group_changed = fallback_rows = 0
    step_counter = torch.full((B,), initial_step, dtype=torch.long, device=device)

    for blk in range(start_block, num_blocks):
        bs, be = P + blk * BL, P + (blk + 1) * BL
        if cfg.window_mode == "growing" and blk > start_block:
            x = torch.cat([x, torch.full((B, BL), MASK, dtype=torch.long, device=device)], dim=1)
            attn = torch.cat([attn, torch.ones(B, BL, dtype=torch.long, device=device)], dim=1)
        if cfg.mode == "fixed_budget":
            block_masks = (x[:, bs:be] == MASK) & active[:, None]
            k_sched = get_num_transfer_tokens(block_masks, spb)
        step_in_block = 0
        while True:
            eligible = torch.zeros_like(x, dtype=torch.bool)
            eligible[:, bs:be] = x[:, bs:be] == MASK
            eligible &= active[:, None]
            rows_with_masks = eligible.any(dim=1)
            if cfg.mode == "threshold":
                if not bool(rows_with_masks.any()):
                    break
                if step_in_block >= BL:
                    raise RuntimeError("threshold decoding exceeded block_length steps in one block (invariant violated)")
                k_t = None
            else:
                if step_in_block >= spb:
                    break
                k_t = k_sched[:, step_in_block]
                if cfg.skip_empty_budget_steps and not bool((k_t[rows_with_masks] > 0).any()):
                    empty_budget_steps += 1
                    step_in_block += 1
                    continue
            ctx = StepContext(
                eligible_mask=eligible, active_rows=active & rows_with_masks,
                block_start=torch.full((B,), bs, dtype=torch.long, device=device),
                block_end=torch.full((B,), be, dtype=torch.long, device=device),
                step_id=step_counter.clone(),
            )
            with _autocast(cfg, device):
                out = gm(x, attn, context=ctx, mode=guidance_mode)
            nfe += 1
            cand, conf = choose_tokens(out.logits, cfg.temperature, sample_rng,
                                       MASK if cfg.forbid_mask_candidate else None)
            conf = torch.where(eligible, conf, torch.full_like(conf, -math.inf))
            commit = select_commits(eligible, conf, cfg.mode, tau, k_t)
            if cfg.mode == "threshold":
                fallback_rows += int((rows_with_masks & ~(eligible & (conf > tau)).any(1)).sum())
            if bool((cand[commit] == MASK).any()):
                raise RuntimeError("sampler produced MASK as a committed candidate")
            if not bool(torch.isfinite(conf[commit]).all()):
                raise RuntimeError("non-finite confidence at a committed position")
            if out.guidance is not None:
                gi = out.guidance
                for r in gi.skip_reason:
                    skip_reasons[r] += 1
                receiver_steps += int(((gi.n_eligible >= 2) & ctx.active_rows).sum())
                n_open = int(gi.gate_open.sum())
                gate_open_steps += n_open
                if gi.group_pos is not None:
                    rr, ss = ((gi.group_pos >= 0) & gi.gate_open[:, None]).nonzero(as_tuple=True)
                    pp = gi.group_pos[rr, ss]
                    group_tokens += len(pp)
                    group_committed += int(commit[rr, pp].sum())
                    group_changed += int((cand[rr, pp] != gi.group_probs[rr, ss].argmax(-1)).sum())
                if n_open:
                    rows = gi.gate_open.nonzero(as_tuple=False).flatten()
                    pos = gi.source_pos[rows]
                    gate_source_committed += int(commit[rows, pos].sum())
                    if gi.source_argmax is not None:
                        gate_source_changed += int((cand[rows, pos] != gi.source_argmax[rows]).sum())
            if recorder is not None:
                recorder.record_step(step_id=step_counter, block_id=blk, block_start=bs, block_end=be, x=x,
                                     attention_mask=attn, eligible=eligible, commit=commit, cand=cand,
                                     conf=conf, active=ctx.active_rows, guidance=out.guidance, logits=out.logits)
            x = torch.where(commit, cand, x)
            commit_counts.append(int(commit.sum()))
            for b in rows_with_masks.nonzero(as_tuple=False).flatten().tolist():
                row_steps[b] += 1
            step_counter += rows_with_masks.long()
            step_in_block += 1
        for b in range(B):
            if bool(active[b]):
                computed[b] = be - P
        if cfg.stop_mode == "block_end":
            for b in active.nonzero(as_tuple=False).flatten().tolist():
                if bool((x[b, P:be] == EOS).any()):
                    active[b] = False
                    stop_reasons[b] = "eos"
        if not bool(active.any()):
            break

    effective = []
    for b in range(B):
        gen = x[b, P:P + computed[b]]
        hits = (gen == EOS).nonzero(as_tuple=False).flatten()
        effective.append(int(hits[0]) if hits.numel() else computed[b])
    stats = {
        "nfe": nfe, "commit_counts": commit_counts, "gate_open_steps": gate_open_steps,
        "receiver_steps": receiver_steps, "skip_reasons": dict(skip_reasons),
        "empty_budget_steps": empty_budget_steps, "gate_source_committed": gate_source_committed,
        "gate_source_identity_changed": gate_source_changed, "mode": cfg.mode,
        "commit_threshold": tau if cfg.mode == "threshold" else None,
        "steps": total_steps if cfg.mode == "fixed_budget" else None,
        "group_tokens": group_tokens, "group_committed": group_committed,
        "group_identity_changed": group_changed, "fallback_rows": fallback_rows,
    }
    return GenerationResult(x, P, computed, effective, stop_reasons, row_steps, stats)
