"""End-to-end smoke test for the dLLM OPSD pipeline.

Samples 10 (problem, solution) examples and runs:
  1. Data collator → check shapes / pad-and-concat layout.
  2. Trainer's mask sampler → check mask is restricted to answer span.
  3. Student forward + (no-grad) teacher forward through Dream-7B.
  4. Token-level JSD on masked positions only.
  5. (Optional) backward pass on the student to confirm gradient flow.

Usage:
    python test_opsd_dllm.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B \
        --num_samples 10

Set NUM_SAMPLES=N or pass --num_samples to vary; default is 10.
"""

import argparse
import os
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from data_collator_dllm import SelfDistillationDLLMDataCollator
from opsd_dllm_trainer import OPSDDLLMTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
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
    return p.parse_args()


def section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")


@torch.no_grad()
def _basic_forward(model, input_ids, attention_mask):
    """Forward + Dream's logits-shift, mirroring the trainer."""
    # Dream's modeling_dream.py passes attention_mask straight to SDPA without
    # _prepare_4d_attention_mask, so we expand (B, L) -> (B, 1, 1, L) here.
    mask = attention_mask[:, None, None, :].bool()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=mask)
    return torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    section(f"1. Loading tokenizer + model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  pad_token_id = {tokenizer.pad_token_id}")
    print(f"  mask_token_id = {tokenizer.mask_token_id}")

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"  model class    = {type(model).__name__}")
    print(f"  model dtype    = {next(model.parameters()).dtype}")
    print(f"  vocab size     = {model.config.vocab_size}")

    section(f"2. Sampling {args.num_samples} examples from {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))
    print(f"  columns: {ds.column_names}")
    print(f"  example[0].problem (first 200 chars):  {ds[0]['problem'][:200]!r}")
    print(f"  example[0].solution (first 200 chars): {ds[0]['solution'][:200]!r}")

    section("3. Data collator → batch tensors")
    collator = SelfDistillationDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
    )
    batch = collator([ds[i] for i in range(args.num_samples)])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:38s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:38s} = {v}")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Sanity: prompt portion uses left-pad (pads only at the start) and answer is right-pad
    s_attn = batch["student_prompt_attention_mask"]
    t_attn = batch["teacher_prompt_attention_mask"]
    a_attn = batch["answer_attention_mask"]
    print("  per-example actual prompt/answer lengths:")
    for i in range(args.num_samples):
        print(
            f"    sample[{i}] student_prompt={int(s_attn[i].sum())} "
            f"teacher_prompt={int(t_attn[i].sum())} answer={int(a_attn[i].sum())}"
        )

    section("4. Building a temporary trainer instance to reuse mask sampler + JSD")
    # Build a trainer-like helper without invoking SFTTrainer.__init__ (which would
    # require GOLDConfig args). Just use OPSDDLLMTrainer's static / pure methods.
    helper = SimpleNamespace(
        sampling_eps=args.sampling_eps,
        mask_token_id=tokenizer.mask_token_id,
    )

    B, A = batch["answer_input_ids"].shape
    mask_pattern, p_mask_sample = OPSDDLLMTrainer._sample_mask(
        helper, B, A, batch["answer_lengths"], device
    )
    print(f"  mask_pattern shape         = {tuple(mask_pattern.shape)}")
    print(f"  per-sample mask rate       = {[round(float(x), 4) for x in p_mask_sample.tolist()]}")
    print(f"  total masked tokens        = {int(mask_pattern.sum())}")
    print(f"  total answer tokens        = {int(batch['answer_lengths'].sum())}")

    # Verify: every masked position is inside the actual answer span
    positions = torch.arange(A, device=device)[None, :].expand(B, -1)
    valid = positions < batch["answer_lengths"][:, None]
    assert (mask_pattern & ~valid).sum() == 0, "mask leaked outside the answer span!"
    print("  ✓ all masked positions lie within the answer span")

    section("5. Building noisy answer + concat full sequences")
    noisy_answer = torch.where(
        mask_pattern,
        torch.full_like(batch["answer_input_ids"], tokenizer.mask_token_id),
        batch["answer_input_ids"],
    )
    student_input_ids = torch.cat([batch["student_prompt_input_ids"], noisy_answer], dim=1)
    student_attn_mask = torch.cat([batch["student_prompt_attention_mask"], batch["answer_attention_mask"]], dim=1)
    teacher_input_ids = torch.cat([batch["teacher_prompt_input_ids"], noisy_answer], dim=1)
    teacher_attn_mask = torch.cat([batch["teacher_prompt_attention_mask"], batch["answer_attention_mask"]], dim=1)
    s_prompt_len = batch["student_prompt_input_ids"].shape[1]
    t_prompt_len = batch["teacher_prompt_input_ids"].shape[1]
    print(f"  student_input_ids shape    = {tuple(student_input_ids.shape)}")
    print(f"  teacher_input_ids shape    = {tuple(teacher_input_ids.shape)}")
    print(f"  s_prompt_len={s_prompt_len}, t_prompt_len={t_prompt_len}, A={A}")

    # Print one sample (first masked token in sample 0) decoded for sanity
    first_idx = mask_pattern[0].nonzero(as_tuple=False)[0].item() if mask_pattern[0].any() else None
    if first_idx is not None:
        true_id = int(batch["answer_input_ids"][0, first_idx])
        print(
            f"  sample 0, masked answer-pos {first_idx}: "
            f"true token = {true_id} ({tokenizer.decode([true_id])!r})"
        )

    section("6. Student & teacher forward + token-level JSD (no_grad sanity)")
    with torch.no_grad():
        student_logits = _basic_forward(model, student_input_ids, student_attn_mask)
        teacher_logits = _basic_forward(model, teacher_input_ids, teacher_attn_mask)
    print(f"  student_logits shape       = {tuple(student_logits.shape)}")
    print(f"  teacher_logits shape       = {tuple(teacher_logits.shape)}")

    student_answer_logits = student_logits[:, s_prompt_len : s_prompt_len + A, :]
    teacher_answer_logits = teacher_logits[:, t_prompt_len : t_prompt_len + A, :]
    print(f"  answer-slice student       = {tuple(student_answer_logits.shape)}")
    print(f"  answer-slice teacher       = {tuple(teacher_answer_logits.shape)}")

    student_masked = student_answer_logits[mask_pattern].float()  # [N, V]
    teacher_masked = teacher_answer_logits[mask_pattern].float()  # [N, V]
    print(f"  gathered masked logits     = {tuple(student_masked.shape)} (N, V)")

    per_token_jsd = OPSDDLLMTrainer.generalized_jsd_loss(
        student_masked,
        teacher_masked,
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="none",
    )
    print(f"  per_token_jsd shape        = {tuple(per_token_jsd.shape)}")
    print(f"  per_token_jsd  mean        = {float(per_token_jsd.mean()):.6f}")
    print(f"  per_token_jsd  max         = {float(per_token_jsd.max()):.6f}")
    print(f"  per_token_jsd  median      = {float(per_token_jsd.median()):.6f}")
    print(f"  per_token_jsd  >0 fraction = {float((per_token_jsd > 0).float().mean()):.4f}")
    assert torch.isfinite(per_token_jsd).all(), "per-token JSD has non-finite values"
    assert (per_token_jsd >= 0).all(), "per-token JSD must be non-negative (KL/JSD ≥ 0)"
    print("  ✓ per-token JSD is finite and non-negative")

    # Top-3 teacher predictions for the first masked position of sample 0 — should match the
    # ground-truth answer reasonably often since the teacher sees the privileged solution.
    if first_idx is not None:
        true_id = int(batch["answer_input_ids"][0, first_idx])
        t_topk = teacher_answer_logits[0, first_idx].topk(5)
        s_topk = student_answer_logits[0, first_idx].topk(5)
        print(f"\n  sample 0, masked pos {first_idx}, true id = {true_id} ({tokenizer.decode([true_id])!r})")
        print(f"    teacher top-5: ids={t_topk.indices.tolist()}  decoded={[tokenizer.decode([i]) for i in t_topk.indices.tolist()]}")
        print(f"    student top-5: ids={s_topk.indices.tolist()}  decoded={[tokenizer.decode([i]) for i in s_topk.indices.tolist()]}")

    if args.no_backward:
        print("\nSkipping backward test (--no_backward).")
        return

    section("7. Backward pass: gradient flows through student forward only")
    model.train()
    # Only the student forward must require grad; the teacher forward stays no-grad.
    out_s = model(input_ids=student_input_ids, attention_mask=student_attn_mask)
    s_logits = torch.cat([out_s.logits[:, :1], out_s.logits[:, :-1]], dim=1)
    s_answer = s_logits[:, s_prompt_len : s_prompt_len + A, :]
    s_masked = s_answer[mask_pattern].float()
    # Reuse the no-grad teacher logits from step 6.
    loss = OPSDDLLMTrainer.generalized_jsd_loss(
        s_masked,
        teacher_masked.detach(),
        beta=args.beta,
        temperature=args.temperature,
        token_clip=args.token_clip if args.token_clip > 0 else None,
        reduction="batchmean",
    )
    print(f"  scalar loss                = {float(loss):.6f}")
    loss.backward()

    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.detach().float().norm().item())
    if grad_norms:
        gn = torch.tensor(grad_norms)
        print(f"  num params with grad       = {len(grad_norms)}")
        print(f"  grad norm mean / max       = {gn.mean():.4f} / {gn.max():.4f}")
        assert torch.isfinite(gn).all(), "non-finite grad norms"
        assert gn.max() > 0, "all grads are zero — gradient is not flowing"
        print("  ✓ gradient flows through the student forward")
    else:
        raise RuntimeError("No parameters received gradients!")

    section("All checks passed ✓")


if __name__ == "__main__":
    main()
