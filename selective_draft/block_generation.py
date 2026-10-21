"""Block-causal threshold decoder for LLaDA2.0-mini (plan §5), with DMax's dInfer evaluation semantics available.

Common to every setting (taken from the official HF `generate`):
  * blocks are absolute cells [k*B, (k+1)*B) counted from position 0; decoding starts in block P // B, whose
    head may still be prompt (known positions, never eligible);
  * each forward sees only the prefix up to the end of the current block (block-causal attention hides the
    future anyway);
  * steps_per_block == block_length: at least one commit per step, threshold decoding is adaptive.
  * T=0 is argmax and consumes no RNG (HF 2.0 samples with multinomial even at T=0); confidence is the raw
    model probability of the chosen token.

Switches (GenerationConfig), defaults = HF behaviour:
  commit_rule    hf      conf > tau, else the single best; MASK is never committed (next-best token is used)
                 dinfer  DMax/dInfer ThresholdParallelDecoder: conf >= min(tau, max_conf - 1e-5); a position whose
                         argmax is MASK is not eligible. If every eligible argmax is MASK dInfer would spin forever;
                         here the best non-MASK token is committed and counted (stats.mask_argmax_fallbacks).
  length_policy  cap             sem_end = P + max_new_tokens, last block rounded up physically
                 dinfer_floor32  sem_end = floor((P + G) / B) * B (the eval harness buckets the total length down)
  stop_mode      block_end            stop after the block that contains an EOS (official HF outer loop, dInfer)
                 prefix_complete_eos  stop mid-block once the first EOS has no MASK on its left (a pure compute
                                      saving here: committed tokens never change, so the text before the EOS is the same)
  output_view    cut_at_eos  completion = tokens before the first EOS
                 drop_eos    completion = every generated non-EOS token (dInfer's TokenArray.get_generated_tokens)
  cache_mode     prefix  dInfer-style prefix KV cache: prefill the full prompt blocks, decode a block against the
                         cache, and seal the finished block with its clean tokens in the next block's first forward
                         (64-token window [previous block | current block], previous rows cannot see the current
                         block). Mathematically the same forward as recomputing the prefix. Guidance off only.

NFE accounting like dInfer's eval script: nfe_dinfer = 1 (prefill) + per block the forwards its unrolled loop
charges (decoding.dinfer_block_forwards), tpf_dinfer = non-EOS generated tokens / nfe_dinfer.
Communication groups never include a rounded-up tail: StepContext.guidance_mask = eligible & (pos < sem_end).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .config import GenerationConfig
from .decoding import StepAccounting, choose_tokens, dinfer_block_forwards, dinfer_commits, select_commits
from .generation import GenerationResult, _autocast
from .model import GuidedModel, StepContext, StepOutput


def _first_eos(x: torch.Tensor, eos: int, start: int, end: int) -> Optional[int]:
    hits = (x[0, start:end] == eos).nonzero(as_tuple=False).flatten()
    return start + int(hits[0]) if hits.numel() else None


@torch.no_grad()
def generate_block_causal(gm: GuidedModel, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor,
                          cfg: GenerationConfig, guidance_mode: Optional[str] = None, recorder=None,
                          sample_rng: Optional[torch.Generator] = None,
                          commit_threshold: Optional[float] = None) -> GenerationResult:
    backend = gm.backend
    device = prompt_ids.device
    MASK, EOS = backend.mask_token_id, backend.eos_token_id
    B, P = prompt_ids.shape
    if B != 1:
        raise ValueError("block-causal decoding is batch-1 (left padding would shift the absolute block grid)")
    if prompt_attention_mask.shape != prompt_ids.shape or not bool((prompt_attention_mask != 0).all()):
        raise ValueError("block-causal decoding expects an unpadded prompt")
    if cfg.mode != "threshold" or cfg.steps_per_block != cfg.block_length:
        raise ValueError("block-causal decoding implements threshold mode with steps_per_block == block_length")
    BL, G = cfg.block_length, cfg.max_new_tokens
    tau = cfg.commit_threshold if commit_threshold is None else float(commit_threshold)
    if cfg.length_policy == "dinfer_floor32":
        sem_end = (P + G) // BL * BL
        if sem_end <= P:
            raise ValueError(f"prompt of {P} tokens leaves no room under length_policy=dinfer_floor32 (G={G})")
    else:
        sem_end = P + G
    phys_end = -(-sem_end // BL) * BL
    first_block, n_blocks = P // BL, phys_end // BL
    positions = torch.arange(phys_end, device=device)
    dinfer_rule = cfg.commit_rule == "dinfer"
    mode = gm.cfg.mode if guidance_mode is None else guidance_mode
    use_cache = cfg.cache_mode == "prefix" and mode == "off" and P >= BL and hasattr(backend, "forward_window")
    mask_dtype = backend.embedding.weight.dtype

    x = prompt_ids.to(torch.long).clone()
    acct = StepAccounting()
    stop_reason, eos_pos, computed, row_steps, step_counter, fallbacks = "length", None, 0, 0, 0, 0
    block_forwards = 0
    active = torch.ones(1, dtype=torch.bool, device=device)
    cache, need_cross = None, False
    if use_cache:
        cache = backend.new_cache()
        pl = first_block * BL  # full prompt blocks only; the partial prompt tail belongs to the first decode block
        with _autocast(cfg, device):
            backend.forward_window(x[:, :pl], 0, cache, keep=pl, attention_mask=backend.block_mask(pl, device, mask_dtype),
                                   return_logits=False)
    for blk in range(first_block, n_blocks):
        bs, be = blk * BL, (blk + 1) * BL
        x = torch.cat([x, torch.full((1, be - x.shape[1]), MASK, dtype=torch.long, device=device)], dim=1)
        attn = torch.ones_like(x)
        m0 = int((x[0, bs:be] == MASK).sum())
        block_commits: List[int] = []
        step_in_block = 0
        while True:
            eligible = torch.zeros_like(x, dtype=torch.bool)
            eligible[:, bs:be] = x[:, bs:be] == MASK
            if not bool(eligible.any()):
                break
            if step_in_block >= BL:
                raise RuntimeError("threshold decoding exceeded block_length steps in one block (invariant violated)")
            ctx = StepContext(eligible_mask=eligible, active_rows=active.clone(),
                              block_start=torch.tensor([bs], device=device), block_end=torch.tensor([be], device=device),
                              step_id=torch.tensor([step_counter], device=device), commit_threshold=tau,
                              guidance_mask=eligible & (positions[:be] < sem_end)[None])
            with _autocast(cfg, device):
                if use_cache and need_cross:
                    # seal the previous block with its clean tokens and decode this block in the same forward
                    cross = torch.zeros(1, 1, 2 * BL, bs + BL, dtype=mask_dtype, device=device)
                    cross[..., :BL, bs:bs + BL] = -math.inf
                    window = backend.forward_window(x[:, bs - BL:be], bs - BL, cache, keep=bs, attention_mask=cross,
                                                    last_n=BL)
                    need_cross = False
                    out = StepOutput(window, None, None)
                elif use_cache:
                    window = backend.forward_window(x[:, bs:be], bs, cache, keep=bs, last_n=BL)
                    out = StepOutput(window, None, None)
                else:
                    out = gm(x, attn, context=ctx, mode=guidance_mode)
                    window = out.logits[:, bs:be]
            # Only the current block can commit: sample/score that window (N x V softmax would cost GBs at N~2.6k).
            # The RNG draw shape therefore covers the window only; T=0 is unaffected.
            forbid = None if dinfer_rule else (MASK if cfg.forbid_mask_candidate else None)
            wc, wf = choose_tokens(window, cfg.temperature, sample_rng, forbid)
            cand = torch.full_like(x, MASK)
            cand[:, bs:be] = wc
            conf = torch.full(x.shape, -math.inf, dtype=wf.dtype, device=device)
            conf[:, bs:be] = wf
            if dinfer_rule:
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
            else:
                conf = torch.where(eligible, conf, torch.full_like(conf, -math.inf))
                commit = select_commits(eligible, conf, "threshold", tau)
            if bool((cand[commit] == MASK).any()):
                raise RuntimeError("sampler produced MASK as a committed candidate")
            if not bool(torch.isfinite(conf[commit]).all()):
                raise RuntimeError("non-finite confidence at a committed position")
            acct.step(out, ctx, eligible, eligible.any(dim=1), cand, conf, commit, "threshold", tau, ge=dinfer_rule)
            if recorder is not None:
                recorder.record_step(step_id=ctx.step_id, block_id=blk, block_start=bs, block_end=be, x=x,
                                     attention_mask=attn, eligible=eligible, commit=commit, cand=cand,
                                     conf=conf, active=ctx.active_rows, guidance=out.guidance,
                                     logits=None if use_cache else out.logits)
            block_commits.append(int(commit.sum()))
            x = torch.where(commit, cand, x)
            row_steps += 1
            step_counter += 1
            step_in_block += 1
            if cfg.stop_mode == "prefix_complete_eos":
                e = _first_eos(x, EOS, P, min(be, sem_end))
                if e is not None and not bool((x[0, P:e] == MASK).any()):
                    stop_reason, eos_pos = "eos", e
                    break
        computed = be - P
        block_forwards += dinfer_block_forwards(block_commits, m0, cfg.dinfer_expected_tpf, cfg.dinfer_unroll)
        need_cross = use_cache
        if stop_reason == "eos":
            break
        if cfg.stop_mode in ("block_end", "prefix_complete_eos"):
            e = _first_eos(x, EOS, P, min(be, sem_end))
            if e is not None:
                stop_reason, eos_pos = "eos", e
                break

    effective = (eos_pos - P) if eos_pos is not None else min(computed, sem_end - P)
    completions = None
    output_tokens = effective
    if cfg.output_view == "drop_eos":
        region = x[0, P:min(P + computed, sem_end)].tolist()
        completions = [[t for t in region if t != EOS]]
        output_tokens = len(completions[0])
    role_end = getattr(backend, "role_end_token_id", None)
    first_role_end = None
    if role_end is not None:
        hits = (x[0, P:P + effective] == role_end).nonzero(as_tuple=False).flatten()
        first_role_end = int(hits[0]) if hits.numel() else None
    nfe_dinfer = 1 + block_forwards  # +1: dInfer's prefill forward
    stats = acct.stats(mode="threshold", commit_threshold=tau, steps=None, prompt_len=P,
                       physical_end=phys_end, first_block=first_block, blocks_decoded=(P + computed) // BL - first_block,
                       eos_position=None if eos_pos is None else eos_pos - P, first_role_end=first_role_end,
                       stop_mode=cfg.stop_mode, commit_rule=cfg.commit_rule, length_policy=cfg.length_policy,
                       output_view=cfg.output_view, cache_used=use_cache, max_new_tokens_effective=sem_end - P,
                       mask_argmax_fallbacks=fallbacks, nfe_decode=acct.nfe, nfe_dinfer=nfe_dinfer,
                       output_tokens=output_tokens, tpf_dinfer=output_tokens / nfe_dinfer)
    return GenerationResult(x, P, [computed], [effective], [stop_reason], [row_steps], stats, completions)
