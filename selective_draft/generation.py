"""Block-causal threshold decoder for LLaDA2.0-mini with DMax's dInfer evaluation semantics.

Official LLaDA2.0-mini decoding (HF `generate`, model card / paper: block 32, threshold 0.95, T=0):
  * blocks are absolute cells [k*B, (k+1)*B) counted from position 0; decoding starts in block P // B, whose head may
    still be prompt (known positions, never eligible);
  * each forward sees only the prefix up to the end of the current block (block-causal attention hides the future);
  * at most block_length steps per block: commit the confident positions, at least one per step;
  * early_stop: stop after the block that contains an EOS (official outer loop and dInfer).
Exactly as DMax's evaluation (dInfer ThresholdParallelDecoder + eval_dinfer_sglang.py, DMax commit 43fafda):
  * commit every eligible position with confidence >= min(tau, best - 1e-5); a position whose argmax is MASK is not
    eligible. If every eligible argmax is MASK dInfer would spin forever; here the best non-MASK token is committed
    and counted (stats.mask_argmax_fallbacks);
  * total length floor((P + max_new_tokens) / B) * B (the harness buckets it down to a multiple of 32);
  * completion = every generated non-EOS token (TokenArray.get_generated_tokens), not the text up to the first EOS;
  * NFE like the eval script: 1 prefill + per block the forwards its unrolled loop charges
    (decoding.dinfer_block_forwards); tpf = completion tokens / NFE.
  * T=0 is argmax and consumes no RNG; confidence is the raw model probability (fp32 softmax).
cache_mode=prefix: dInfer-style prefix KV cache. Prefill the full prompt blocks, decode a block against the cache,
seal a finished block with its clean tokens in the next block's first forward (a 2B window [previous | current],
previous rows cannot see the current block). The same forward as recomputing the prefix; guidance off only.
Communication groups never include positions past the floored end: StepContext.guidance_mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import GenerationConfig
from .decoding import StepAccounting, choose_tokens, dinfer_block_forwards, dinfer_commits
from .model import GuidedModel, StepContext, StepOutput


@dataclass
class GenerationResult:
    sequences: torch.Tensor  # [1, N] prompt + decoded window
    prompt_len: int
    completion: List[int]  # every generated non-EOS token (DMax/dInfer)
    computed_length: int  # generated positions actually decoded (block granularity)
    eos_position: Optional[int]  # first EOS relative to the prompt, None if none
    stop_reason: str  # eos | length
    steps: int  # denoising forwards
    stats: Dict[str, object] = field(default_factory=dict)


@torch.no_grad()
def generate(gm: GuidedModel, prompt_ids: torch.Tensor, cfg: GenerationConfig, guidance_mode: Optional[str] = None,
             recorder=None, sample_rng: Optional[torch.Generator] = None,
             commit_threshold: Optional[float] = None) -> GenerationResult:
    backend = gm.backend
    device = prompt_ids.device
    MASK, EOS = backend.mask_token_id, backend.eos_token_id
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("block-causal decoding is batch-1 (padding would shift the absolute block grid)")
    P = prompt_ids.shape[1]
    BL = cfg.block_length
    tau = cfg.commit_threshold if commit_threshold is None else float(commit_threshold)
    end = (P + cfg.max_new_tokens) // BL * BL
    if end <= P:
        raise ValueError(f"a prompt of {P} tokens leaves no room for generation (max_new_tokens={cfg.max_new_tokens})")
    first_block, n_blocks = P // BL, end // BL
    positions = torch.arange(end, device=device)
    mode = gm.cfg.mode if guidance_mode is None else guidance_mode
    use_cache = cfg.cache_mode == "prefix" and mode == "off" and P >= BL
    mask_dtype = backend.embedding.weight.dtype

    x = prompt_ids.to(torch.long).clone()
    acct = StepAccounting()
    stop_reason, eos_pos, computed, steps, fallbacks, block_forwards = "length", None, 0, 0, 0, 0
    active = torch.ones(1, dtype=torch.bool, device=device)
    cache, need_cross = None, False
    if use_cache:
        cache = backend.new_cache()
        pl = first_block * BL  # full prompt blocks; the partial prompt tail belongs to the first decode block
        backend.forward_window(x[:, :pl], 0, cache, keep=pl, attention_mask=backend.block_mask(pl, device, mask_dtype),
                               return_logits=False)
    for blk in range(first_block, n_blocks):
        bs, be = blk * BL, (blk + 1) * BL
        x = torch.cat([x, torch.full((1, be - x.shape[1]), MASK, dtype=torch.long, device=device)], dim=1)
        m0 = int((x[0, bs:be] == MASK).sum())
        block_commits: List[int] = []
        while True:
            eligible = torch.zeros_like(x, dtype=torch.bool)
            eligible[:, bs:be] = x[:, bs:be] == MASK
            if not bool(eligible.any()):
                break
            if len(block_commits) >= BL:
                raise RuntimeError("threshold decoding exceeded block_length steps in one block (invariant violated)")
            ctx = StepContext(eligible_mask=eligible, active_rows=active.clone(),
                              block_start=torch.tensor([bs], device=device), block_end=torch.tensor([be], device=device),
                              step_id=torch.tensor([steps], device=device), commit_threshold=tau,
                              guidance_mask=eligible & (positions[:be] < end)[None])
            if use_cache and need_cross:  # seal the previous block with its clean tokens, decode this block
                cross = torch.zeros(1, 1, 2 * BL, bs + BL, dtype=mask_dtype, device=device)
                cross[..., :BL, bs:bs + BL] = -math.inf
                window = backend.forward_window(x[:, bs - BL:be], bs - BL, cache, keep=bs, attention_mask=cross,
                                                last_n=BL)
                need_cross = False
                out = StepOutput(None, None, None)
            elif use_cache:
                window = backend.forward_window(x[:, bs:be], bs, cache, keep=bs, last_n=BL)
                out = StepOutput(None, None, None)
            else:
                out = gm(x, context=ctx, mode=guidance_mode)
                window = out.logits[:, bs:be]
            # Only the current block can commit: score that window (an N x V softmax would cost GBs at N ~ 2.6k).
            wc, wf = choose_tokens(window, cfg.temperature, sample_rng)
            cand = torch.full_like(x, MASK)
            cand[:, bs:be] = wc
            conf = torch.full(x.shape, -math.inf, dtype=wf.dtype, device=device)
            conf[:, bs:be] = wf
            ok = eligible & (cand != MASK)
            conf = torch.where(ok, conf, torch.full_like(conf, -math.inf))
            commit = dinfer_commits(ok, conf, tau)
            if not bool(commit.any()):  # every eligible argmax is MASK: dInfer would never finish this block
                wc, wf = choose_tokens(window, cfg.temperature, sample_rng, MASK)
                cand[:, bs:be] = wc
                conf = torch.full(x.shape, -math.inf, dtype=wf.dtype, device=device)
                conf[:, bs:be] = wf
                conf = torch.where(eligible, conf, torch.full_like(conf, -math.inf))
                commit = dinfer_commits(eligible, conf, tau)
                fallbacks += 1
            if bool((cand[commit] == MASK).any()) or not bool(torch.isfinite(conf[commit]).all()):
                raise RuntimeError("invalid committed candidate (MASK or non-finite confidence)")
            acct.step(out, ctx, eligible, cand, conf, commit, tau)
            if recorder is not None:
                recorder.record_step(step_id=steps, block_id=blk, block_start=bs, block_end=be, x=x,
                                     eligible=eligible, commit=commit, cand=cand, conf=conf, guidance=out.guidance)
            block_commits.append(int(commit.sum()))
            x = torch.where(commit, cand, x)
            steps += 1
        computed = be - P
        block_forwards += dinfer_block_forwards(block_commits, m0, cfg.dinfer_expected_tpf, cfg.dinfer_unroll)
        need_cross = use_cache
        hits = (x[0, P:be] == EOS).nonzero(as_tuple=False).flatten()
        if hits.numel():
            eos_pos = int(hits[0])
            if cfg.early_stop:
                stop_reason = "eos"
                break

    completion = [t for t in x[0, P:P + computed].tolist() if t != EOS]
    nfe_dinfer = 1 + block_forwards  # +1: dInfer's prefill forward
    stats = acct.stats(threshold=tau, prompt_len=P, end=end, first_block=first_block,
                       blocks_decoded=(P + computed) // BL - first_block, cache_used=use_cache,
                       max_new_tokens_effective=end - P, mask_argmax_fallbacks=fallbacks,
                       nfe_dinfer=nfe_dinfer, output_tokens=len(completion), tpf_dinfer=len(completion) / nfe_dinfer)
    return GenerationResult(x, P, completion, computed, eos_pos, stop_reason, steps, stats)
