"""One decoding core for collection, training rollouts and evaluation.
Threshold mode (adaptive steps) and fixed-budget mode share the loop; only the commit rule differs.
Left-padded prompts; attention_mask 0 = padding only (MASK tokens stay attendable)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import GenerationConfig
from .decoding import StepAccounting, choose_tokens, get_num_transfer_tokens, select_commits  # noqa: F401
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
    # output_view=drop_eos (dInfer): the completion is every generated non-EOS token of the decoded region, not the
    # text up to the first EOS. effective_lengths keeps the cut-at-first-EOS meaning either way.
    completions: Optional[List[List[int]]] = None

    def completion_ids(self, b: int, effective: bool = True) -> List[int]:
        if self.completions is not None and effective:
            return list(self.completions[b])
        n = self.effective_lengths[b] if effective else self.computed_lengths[b]
        return self.sequences[b, self.prompt_len:self.prompt_len + n].tolist()


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
    if cfg.block_origin == "sequence_zero":
        from .block_generation import generate_block_causal
        if initial_state is not None or start_block or initial_step:
            raise ValueError("state continuation is not part of the block-causal (root_success_v2) protocol")
        return generate_block_causal(gm, prompt_ids, prompt_attention_mask, cfg, guidance_mode=guidance_mode,
                                     recorder=recorder, sample_rng=sample_rng, commit_threshold=commit_threshold)
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
    acct = StepAccounting()
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
                    acct.empty_budget_steps += 1
                    step_in_block += 1
                    continue
            ctx = StepContext(
                eligible_mask=eligible, active_rows=active & rows_with_masks,
                block_start=torch.full((B,), bs, dtype=torch.long, device=device),
                block_end=torch.full((B,), be, dtype=torch.long, device=device),
                step_id=step_counter.clone(),
                commit_threshold=tau if cfg.mode == "threshold" else None,
            )
            with _autocast(cfg, device):
                out = gm(x, attn, context=ctx, mode=guidance_mode)
            cand, conf = choose_tokens(out.logits, cfg.temperature, sample_rng,
                                       MASK if cfg.forbid_mask_candidate else None)
            conf = torch.where(eligible, conf, torch.full_like(conf, -math.inf))
            commit = select_commits(eligible, conf, cfg.mode, tau, k_t)
            if bool((cand[commit] == MASK).any()):
                raise RuntimeError("sampler produced MASK as a committed candidate")
            if not bool(torch.isfinite(conf[commit]).all()):
                raise RuntimeError("non-finite confidence at a committed position")
            acct.step(out, ctx, eligible, rows_with_masks, cand, conf, commit, cfg.mode, tau)
            if recorder is not None:
                recorder.record_step(step_id=step_counter, block_id=blk, block_start=bs, block_end=be, x=x,
                                     attention_mask=attn, eligible=eligible, commit=commit, cand=cand,
                                     conf=conf, active=ctx.active_rows, guidance=out.guidance, logits=out.logits)
            x = torch.where(commit, cand, x)
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
    stats = acct.stats(mode=cfg.mode, commit_threshold=tau if cfg.mode == "threshold" else None,
                       steps=total_steps if cfg.mode == "fixed_budget" else None)
    return GenerationResult(x, P, computed, effective, stop_reasons, row_steps, stats)
