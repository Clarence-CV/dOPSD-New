"""End-to-end smoke test for the dLLM OPSD pipeline.

On-policy generation + masked-prediction loss (LLaDA / Dream-style):

    student_prompt  ─►  model.diffusion_generate()  ─►  completion (concrete)
                                                              │
                                              random mask over valid positions
                                                              │
                                                  noisy_completion (with <mask>)
                                                              │
                       ┌──────────────────────────────────────┤
                       │                                      │
                       ▼                                      ▼
   [s_prompt | noisy_completion]            [t_prompt | noisy_completion]
                       │                                      │
                  forward (grad)                       forward (no_grad)
                       │                                      │
                  student_logits                        teacher_logits
                       │                                      │
                       └──── JSD on MASKED positions only ────┘

Notes specific to Dream:
  * Generation uses `model.diffusion_generate(...)`, not `model.generate(...)`.
  * Dream inherits Qwen2.5's AR head, so logits are right-shifted by 1
    (`[:, :1] + [:, :-1]`) to align logit[i] with token[i].
  * `modeling_dream.py` forwards `attention_mask` straight to SDPA without
    `_prepare_4d_attention_mask`, so we expand 2D (B, L) → 4D (B, 1, 1, L) bool
    before every model() call.

Usage:
    python test_opsd_dllm.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --num_samples 4 --max_new_tokens 128 --diffusion_steps 128
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from data_collator_dllm import SelfDistillationDLLMDataCollator
from opsd_dllm_trainer import OPSDDLLMTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_answer_length", type=int, default=256)
    # Generation (diffusion_generate)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=128)
    p.add_argument("--gen_temperature", type=float, default=0.2)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_alg", default="entropy")
    p.add_argument("--gen_alg_temp", type=float, default=0.0)
    # JSD loss
    p.add_argument("--beta", type=float, default=0.5,
                   help="0=forward KL, 1=reverse KL, in (0,1)=JSD mixture")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--token_clip", type=float, default=0.0, help="0 = no clip")
    p.add_argument("--sampling_eps", type=float, default=1e-3,
                   help="Lower bound on the antithetic per-example mask rate")
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_backward", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--instruction_field", default="instruction")
    p.add_argument("--response_field", default="response")
    p.add_argument("--context_field", default="context",
                   help="Pass '' to disable")
    return p.parse_args()


def section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")


def _expand_mask_4d(attention_mask):
    """Dream's modeling_dream.py forwards attention_mask straight to SDPA;
    the 2D (B, L) HF convention must be reshaped to (B, 1, 1, L) bool first."""
    return attention_mask[:, None, None, :].bool()


def _shifted_forward(model, input_ids, attention_mask):
    """Forward + Dream's right-shift logits convention (Qwen2.5-AR-init fix)."""
    mask = _expand_mask_4d(attention_mask)
    out = model(input_ids=input_ids, attention_mask=mask)
    return torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)


def _build_completion_mask(completion_ids, eos_id, pad_id):
    """Mark *valid* (real) positions in a generated completion (1=valid, 0=ignore).

    A position is invalid if it is a pad token, or if it sits *strictly after*
    the first EOS in the same sequence. The EOS itself is kept valid so the
    model is rewarded for terminating in the right place.
    """
    B, L = completion_ids.shape
    mask = torch.ones_like(completion_ids)
    for i in range(B):
        seq = completion_ids[i]
        if eos_id is not None:
            eos_positions = (seq == eos_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first = int(eos_positions[0].item())
                mask[i, first + 1:] = 0
    if pad_id is not None:
        mask[completion_ids == pad_id] = 0
    return mask


def _sample_mask(B, L, valid, sampling_eps, device):
    """Antithetic per-example mask rate, restricted to valid positions.

    Args:
        B, L: batch size and sequence length
        valid: [B, L] bool/int — 1 where a position is allowed to be masked
               (i.e. real generated tokens, not pad / post-EOS)
        sampling_eps: lower clamp on the per-example mask rate

    Returns:
        mask:           [B, L] bool — True at randomly masked positions
        p_mask_sample:  [B] float  — the per-example mask rate (for diagnostics)
    """
    valid_b = valid.bool()
    u0 = torch.rand(1, device=device, dtype=torch.float32)
    idx = torch.arange(B, device=device, dtype=torch.float32)
    t = (u0 + idx / B) % 1
    p_mask_sample = (1 - sampling_eps) * t + sampling_eps  # [B]
    p_mask_grid = p_mask_sample[:, None].expand(B, L)
    rand = torch.rand((B, L), device=device)
    mask = (rand < p_mask_grid) & valid_b

    # Guarantee at least one masked position per example that has valid tokens —
    # otherwise the JSD reduction divides by zero on that row.
    for i in range(B):
        if valid_b[i].any() and not mask[i].any():
            valid_idx = valid_b[i].nonzero(as_tuple=False).flatten()
            j = int(torch.randint(0, valid_idx.numel(), (1,), device=device).item())
            mask[i, int(valid_idx[j].item())] = True
    return mask, p_mask_sample


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    context_field = args.context_field or None

    section(f"1. Load tokenizer + model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token_id  = {tokenizer.pad_token_id}")
    print(f"  eos_token_id  = {tokenizer.eos_token_id}")
    print(f"  mask_token_id = {tokenizer.mask_token_id}")

    model = AutoModel.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  model class = {type(model).__name__}")
    print(f"  model dtype = {next(model.parameters()).dtype}")
    print(f"  vocab size  = {model.config.vocab_size}")

    section(f"2. Sample {args.num_samples} examples from {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))
    print(f"  columns: {ds.column_names}")
    for i in range(args.num_samples):
        row = ds[i]
        cat = row.get("category", "?")
        ctx = row.get(context_field) if context_field else ""
        print(f"  [{i:2d}] cat={cat}  ctx_len={len(ctx) if ctx else 0}")
        print(f"       instr: {row[args.instruction_field][:140]!r}")

    section("3. Build prompts via SelfDistillationDLLMDataCollator")
    collator = SelfDistillationDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        instruction_field=args.instruction_field,
        response_field=args.response_field,
        context_field=context_field,
    )
    batch = collator([ds[i] for i in range(args.num_samples)])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    s_prompt_ids = batch["student_prompt_input_ids"]
    s_prompt_mask = batch["student_prompt_attention_mask"]
    t_prompt_ids = batch["teacher_prompt_input_ids"]
    t_prompt_mask = batch["teacher_prompt_attention_mask"]
    s_prompt_len = s_prompt_ids.shape[1]
    t_prompt_len = t_prompt_ids.shape[1]
    print(f"  student_prompt_ids = {tuple(s_prompt_ids.shape)}")
    print(f"  teacher_prompt_ids = {tuple(t_prompt_ids.shape)}")
    for i in range(args.num_samples):
        print(f"    [{i}] real student_prompt_len={int(s_prompt_mask[i].sum())}  "
              f"real teacher_prompt_len={int(t_prompt_mask[i].sum())}")

    section(f"4. Student rollout via diffusion_generate "
            f"(steps={args.diffusion_steps}, max_new_tokens={args.max_new_tokens})")
    with torch.no_grad():
        gen_out = model.diffusion_generate(
            s_prompt_ids,
            attention_mask=s_prompt_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=args.diffusion_steps,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            alg=args.gen_alg,
            alg_temp=args.gen_alg_temp,
        )
    full_seq = gen_out.sequences
    completion_ids = full_seq[:, s_prompt_len:].contiguous()
    B, L_c = completion_ids.shape
    print(f"  completion_ids = {tuple(completion_ids.shape)}")

    completion_mask = _build_completion_mask(
        completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id
    )
    real_lens = completion_mask.sum(dim=1).tolist()
    print(f"  effective completion lengths (post-EOS trim): {real_lens}")

    print("\n  decoded sample[0] completion (truncated):")
    text0 = tokenizer.decode(
        completion_ids[0][completion_mask[0].bool()], skip_special_tokens=False
    )
    print(f"    {text0[:400]!r}")

    section("5. Sample mask pattern + build noisy completion")
    mask_pattern, p_mask_sample = _sample_mask(
        B, L_c, completion_mask, args.sampling_eps, device
    )
    print(f"  per-example mask rate = "
          f"{[round(float(x), 4) for x in p_mask_sample.tolist()]}")
    print(f"  total masked / total valid completion tokens = "
          f"{int(mask_pattern.sum())} / {int(completion_mask.sum())}")
    # Sanity: masked positions must be a subset of valid positions.
    assert int((mask_pattern & ~completion_mask.bool()).sum()) == 0

    noisy_completion = torch.where(
        mask_pattern,
        torch.full_like(completion_ids, tokenizer.mask_token_id),
        completion_ids,
    )

    student_full_ids = torch.cat([s_prompt_ids, noisy_completion], dim=1)
    student_full_mask = torch.cat([s_prompt_mask, completion_mask], dim=1)
    teacher_full_ids = torch.cat([t_prompt_ids, noisy_completion], dim=1)
    teacher_full_mask = torch.cat([t_prompt_mask, completion_mask], dim=1)
    print(f"  student_full_ids = {tuple(student_full_ids.shape)}  "
          f"(s_prompt_len={s_prompt_len}, completion_len={L_c})")
    print(f"  teacher_full_ids = {tuple(teacher_full_ids.shape)}  "
          f"(t_prompt_len={t_prompt_len}, completion_len={L_c})")

    section("6. Forward + JSD on MASKED positions only (no_grad sanity)")
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            student_logits = _shifted_forward(model, student_full_ids, student_full_mask)
            teacher_logits = _shifted_forward(model, teacher_full_ids, teacher_full_mask)
    student_completion_logits = student_logits[:, s_prompt_len: s_prompt_len + L_c, :]
    teacher_completion_logits = teacher_logits[:, t_prompt_len: t_prompt_len + L_c, :]

    s_flat = student_completion_logits[mask_pattern].float()
    t_flat = teacher_completion_logits[mask_pattern].float()
    print(f"  gathered logits over MASKED completion tokens: "
          f"{tuple(s_flat.shape)} (N, V)")

    per_token_jsd = OPSDDLLMTrainer.generalized_jsd_loss(
        s_flat, t_flat,
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="none",
    )
    print(f"  per_token_jsd  shape  = {tuple(per_token_jsd.shape)}")
    print(f"  per_token_jsd  mean   = {float(per_token_jsd.mean()):.6f}")
    print(f"  per_token_jsd  max    = {float(per_token_jsd.max()):.6f}")
    print(f"  per_token_jsd  median = {float(per_token_jsd.median()):.6f}")
    assert torch.isfinite(per_token_jsd).all(), "non-finite per-token JSD"
    neg = per_token_jsd[per_token_jsd < 0]
    if neg.numel() > 0:
        worst = float(neg.min())
        print(f"  note: {neg.numel()}/{per_token_jsd.numel()} tokens slightly "
              f"negative (worst={worst:.2e}) — float noise")
        assert worst > -1e-3, f"meaningfully negative JSD ({worst})"
    print("  ✓ per-token JSD is finite")

    # Per-position top-5 sanity at the first MASKED token of sample[0].
    if int(mask_pattern[0].sum()) > 0:
        first_pos = int(mask_pattern[0].nonzero(as_tuple=False)[0].item())
        true_id = int(completion_ids[0, first_pos])  # the token the model has to recover
        s_top = student_completion_logits[0, first_pos].topk(5)
        t_top = teacher_completion_logits[0, first_pos].topk(5)
        print(f"\n  sample[0] masked completion-pos {first_pos}: "
              f"true id={true_id} ({tokenizer.decode([true_id])!r})")
        print(f"    student top-5: {[tokenizer.decode([i]) for i in s_top.indices.tolist()]}")
        print(f"    teacher top-5: {[tokenizer.decode([i]) for i in t_top.indices.tolist()]}")

    if args.no_backward:
        print("\nSkipping backward (--no_backward).")
        return

    section("7. Backward: gradient flows through student forward only")
    model.train()
    with torch.amp.autocast("cuda", dtype=dtype):
        s_logits_grad = _shifted_forward(model, student_full_ids, student_full_mask)
    s_completion_logits = s_logits_grad[:, s_prompt_len: s_prompt_len + L_c, :]
    s_completion_flat = s_completion_logits[mask_pattern].float()

    loss = OPSDDLLMTrainer.generalized_jsd_loss(
        s_completion_flat,
        t_flat.detach(),
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="batchmean",
    )
    print(f"  scalar loss = {float(loss):.6f}")
    loss.backward()

    grad_norms = [p.grad.detach().float().norm().item()
                  for p in model.parameters() if p.grad is not None]
    if not grad_norms:
        raise RuntimeError("No parameters received gradients!")
    gn = torch.tensor(grad_norms)
    print(f"  num params with grad = {len(grad_norms)}")
    print(f"  grad norm mean / max = {gn.mean():.4f} / {gn.max():.4f}")
    assert torch.isfinite(gn).all(), "non-finite grad norms"
    assert gn.max() > 0, "all grads are zero"
    print("  ✓ gradient flows through the student forward")

    section("All checks passed ✓")


if __name__ == "__main__":
    main()
