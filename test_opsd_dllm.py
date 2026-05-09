"""End-to-end smoke test for the dLLM OPSD pipeline on the Dolly-15k dataset.

Samples 10 Dolly examples and runs:
  1. Data collator (instruction / context / response columns) → check shapes / pad layout.
  2. Trainer's mask sampler → check mask is restricted to the response span.
  3. Student forward + (no-grad) teacher forward through Dream-7B.
  4. Token-level JSD on masked positions only.
  5. (Optional) backward pass on the student to confirm gradient flow.

Usage:
    python test_opsd_dllm.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --num_samples 10
"""

import argparse
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from data_collator_dllm import SelfDistillationDLLMDataCollator
from opsd_dllm_trainer import OPSDDLLMTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_answer_length", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.0, help="0=forward KL, 1=reverse KL, in (0,1)=JSD mixture")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling_eps", type=float, default=1e-3)
    p.add_argument("--token_clip", type=float, default=0.0, help="0 = no clip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_backward", action="store_true", help="Skip the backward pass test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--instruction_field", default="instruction")
    p.add_argument("--response_field", default="response")
    p.add_argument("--context_field", default="context", help="Pass an empty string '' to disable")
    return p.parse_args()


def section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")


def _expand_mask_4d(attention_mask):
    """Dream's modeling_dream.py passes attention_mask straight to SDPA without
    _prepare_4d_attention_mask, so callers must expand (B, L) -> (B, 1, 1, L) bool."""
    return attention_mask[:, None, None, :].bool()


@torch.no_grad()
def _shifted_forward(model, input_ids, attention_mask):
    """No-grad forward + Dream's right-shift logits convention; mirrors the trainer."""
    mask = _expand_mask_4d(attention_mask)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=mask)
    return torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    context_field = args.context_field or None

    section(f"1. Loading tokenizer + model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token_id  = {tokenizer.pad_token_id}")
    print(f"  mask_token_id = {tokenizer.mask_token_id}")

    model = AutoModel.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"  model class = {type(model).__name__}")
    print(f"  model dtype = {next(model.parameters()).dtype}")
    print(f"  vocab size  = {model.config.vocab_size}")

    section(f"2. Sampling {args.num_samples} examples from {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  total rows: {len(ds)}, columns: {ds.column_names}")
    # Mix categories so the smoke test exercises both context-bearing (closed_qa,
    # information_extraction, summarization) and context-free (open_qa, brainstorming,
    # creative_writing, classification) examples.
    if "category" in ds.column_names:
        cats = sorted(set(ds["category"]))
        print(f"  categories present: {cats}")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    print("\n  sampled examples (first 160 chars per field):")
    for i in range(args.num_samples):
        row = ds[i]
        ctx = row.get(context_field) if context_field else None
        cat = row.get("category", "?")
        print(f"    [{i:2d}] cat={cat:20s} ctx_len={len(ctx) if ctx else 0:4d}")
        print(f"         instr: {row[args.instruction_field][:160]!r}")
        print(f"         resp : {row[args.response_field][:160]!r}")

    section("3. Data collator → batch tensors")
    collator = SelfDistillationDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        instruction_field=args.instruction_field,
        response_field=args.response_field,
        context_field=context_field,
    )
    batch = collator([ds[i] for i in range(args.num_samples)])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:38s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:38s} = {v}")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    s_attn = batch["student_prompt_attention_mask"]
    t_attn = batch["teacher_prompt_attention_mask"]
    a_attn = batch["answer_attention_mask"]
    print("\n  per-example actual prompt/answer token lengths:")
    for i in range(args.num_samples):
        print(
            f"    [{i:2d}] student_prompt={int(s_attn[i].sum()):4d}  "
            f"teacher_prompt={int(t_attn[i].sum()):4d}  answer={int(a_attn[i].sum()):4d}"
        )

    # Quick visual sanity check: decode the first sample's student / teacher prompts.
    print("\n  decoded sample[0] student prompt (truncated):")
    sp_text = tokenizer.decode(
        batch["student_prompt_input_ids"][0][batch["student_prompt_attention_mask"][0].bool()],
        skip_special_tokens=False,
    )
    print(f"    {sp_text[:400]!r}")
    print("\n  decoded sample[0] teacher prompt (truncated):")
    tp_text = tokenizer.decode(
        batch["teacher_prompt_input_ids"][0][batch["teacher_prompt_attention_mask"][0].bool()],
        skip_special_tokens=False,
    )
    print(f"    {tp_text[:400]!r}")

    section("4. Mask sampler → restricted to response span")
    helper = SimpleNamespace(sampling_eps=args.sampling_eps, mask_token_id=tokenizer.mask_token_id)
    B, A = batch["answer_input_ids"].shape
    mask_pattern, p_mask_sample = OPSDDLLMTrainer._sample_mask(
        helper, B, A, batch["answer_lengths"], device
    )
    print(f"  mask_pattern shape   = {tuple(mask_pattern.shape)}")
    print(f"  per-sample mask rate = {[round(float(x), 4) for x in p_mask_sample.tolist()]}")
    print(f"  total masked tokens  = {int(mask_pattern.sum())}")
    print(f"  total answer tokens  = {int(batch['answer_lengths'].sum())}")

    positions = torch.arange(A, device=device)[None, :].expand(B, -1)
    valid = positions < batch["answer_lengths"][:, None]
    assert (mask_pattern & ~valid).sum() == 0, "mask leaked outside the response span!"
    print("  ✓ all masked positions lie within the response span")

    section("5. Building noisy answer + concat full sequences")
    noisy_answer = torch.where(
        mask_pattern,
        torch.full_like(batch["answer_input_ids"], tokenizer.mask_token_id),
        batch["answer_input_ids"],
    )
    student_input_ids = torch.cat([batch["student_prompt_input_ids"], noisy_answer], dim=1)
    student_attn_mask = torch.cat(
        [batch["student_prompt_attention_mask"], batch["answer_attention_mask"]], dim=1
    )
    teacher_input_ids = torch.cat([batch["teacher_prompt_input_ids"], noisy_answer], dim=1)
    teacher_attn_mask = torch.cat(
        [batch["teacher_prompt_attention_mask"], batch["answer_attention_mask"]], dim=1
    )
    s_prompt_len = batch["student_prompt_input_ids"].shape[1]
    t_prompt_len = batch["teacher_prompt_input_ids"].shape[1]
    print(f"  student_input_ids = {tuple(student_input_ids.shape)}")
    print(f"  teacher_input_ids = {tuple(teacher_input_ids.shape)}")
    print(f"  s_prompt_len={s_prompt_len}, t_prompt_len={t_prompt_len}, A={A}")

    first_idx = mask_pattern[0].nonzero(as_tuple=False)[0].item() if mask_pattern[0].any() else None
    if first_idx is not None:
        true_id = int(batch["answer_input_ids"][0, first_idx])
        print(
            f"  sample[0] first masked answer-pos {first_idx}: "
            f"true token = {true_id} ({tokenizer.decode([true_id])!r})"
        )

    section("6. Forward pass + token-level JSD (no_grad)")
    with torch.no_grad():
        student_logits = _shifted_forward(model, student_input_ids, student_attn_mask)
        teacher_logits = _shifted_forward(model, teacher_input_ids, teacher_attn_mask)
    print(f"  student_logits = {tuple(student_logits.shape)}")
    print(f"  teacher_logits = {tuple(teacher_logits.shape)}")

    student_answer_logits = student_logits[:, s_prompt_len : s_prompt_len + A, :]
    teacher_answer_logits = teacher_logits[:, t_prompt_len : t_prompt_len + A, :]
    student_masked = student_answer_logits[mask_pattern].float()
    teacher_masked = teacher_answer_logits[mask_pattern].float()
    print(f"  gathered masked logits = {tuple(student_masked.shape)} (N, V)")

    per_token_jsd = OPSDDLLMTrainer.generalized_jsd_loss(
        student_masked,
        teacher_masked,
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="none",
    )
    print(f"  per_token_jsd shape       = {tuple(per_token_jsd.shape)}")
    print(f"  per_token_jsd  mean       = {float(per_token_jsd.mean()):.6f}")
    print(f"  per_token_jsd  max        = {float(per_token_jsd.max()):.6f}")
    print(f"  per_token_jsd  median     = {float(per_token_jsd.median()):.6f}")
    print(f"  per_token_jsd  >0 fraction= {float((per_token_jsd > 0).float().mean()):.4f}")
    assert torch.isfinite(per_token_jsd).all(), "per-token JSD has non-finite values"
    neg = per_token_jsd[per_token_jsd < 0]
    if neg.numel() > 0:
        worst = float(neg.min())
        print(f"  note: {neg.numel()}/{per_token_jsd.numel()} tokens slightly negative (worst={worst:.2e}) — float noise")
        assert worst > -1e-3, f"per-token JSD has a meaningfully negative value ({worst})"
    print("  ✓ per-token JSD is finite (negatives within float-noise tolerance)")

    if first_idx is not None:
        true_id = int(batch["answer_input_ids"][0, first_idx])
        t_topk = teacher_answer_logits[0, first_idx].topk(5)
        s_topk = student_answer_logits[0, first_idx].topk(5)
        print(f"\n  sample[0] masked pos {first_idx}, true id={true_id} ({tokenizer.decode([true_id])!r})")
        print(f"    teacher top-5: {[tokenizer.decode([i]) for i in t_topk.indices.tolist()]}")
        print(f"    student top-5: {[tokenizer.decode([i]) for i in s_topk.indices.tolist()]}")

    if args.no_backward:
        print("\nSkipping backward test (--no_backward).")
        return

    section("7. Backward pass: gradient flows through student forward only")
    model.train()
    student_attn_4d = _expand_mask_4d(student_attn_mask)
    out_s = model(input_ids=student_input_ids, attention_mask=student_attn_4d)
    s_logits = torch.cat([out_s.logits[:, :1], out_s.logits[:, :-1]], dim=1)
    s_answer = s_logits[:, s_prompt_len : s_prompt_len + A, :]
    s_masked = s_answer[mask_pattern].float()
    loss = OPSDDLLMTrainer.generalized_jsd_loss(
        s_masked,
        teacher_masked.detach(),
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="batchmean",
    )
    print(f"  scalar loss = {float(loss):.6f}")
    loss.backward()

    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.detach().float().norm().item())
    if not grad_norms:
        raise RuntimeError("No parameters received gradients!")
    gn = torch.tensor(grad_norms)
    print(f"  num params with grad = {len(grad_norms)}")
    print(f"  grad norm mean / max = {gn.mean():.4f} / {gn.max():.4f}")
    assert torch.isfinite(gn).all(), "non-finite grad norms"
    assert gn.max() > 0, "all grads are zero — gradient is not flowing"
    print("  ✓ gradient flows through the student forward")

    section("All checks passed ✓")


if __name__ == "__main__":
    main()
