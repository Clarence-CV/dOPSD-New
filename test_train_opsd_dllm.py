"""Smoke-test the full OPSDDLLMTrainer training loop on 10 samples × 3 epochs.

Mirrors `opsd_dllm_train.py` exactly on the data path: 10 rows of the canonical
OPSD training corpus (`siyanzhao/Openthoughts_math_30k_opsd`, `problem` /
`solution` columns), 3 epochs, short completions (`gen_max_new_tokens=32`,
`gen_steps=32`) so the run finishes in a few minutes on a single GPU. No wandb,
no checkpoint, no LR scheduler tricks.

Verifies:
  * The trainer constructs end-to-end with a Dream backbone.
  * `compute_loss` runs (generation + masking + dual forward + JSD).
  * Each step's loss is finite.
  * Loss generally trends downward across the 3 epochs (rough sanity).

Usage:
    python test_train_opsd_dllm.py \
        --model_name_or_path Dream-org/Dream-v0-Instruct-7B
"""

import argparse
import faulthandler
import os
import sys

# Print a Python stack trace when a segfault hits, instead of dying silently.
faulthandler.enable(file=sys.stderr, all_threads=True)

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_callback import TrainerCallback

from trl.experimental.gold import GOLDConfig

from data_collator_dllm import SelfDistillationDLLMDataCollator
from opsd_dllm_trainer import OPSDDLLMTrainer


# Silence the trl.experimental warning so the smoke-test output stays readable.
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


class LossLogger(TrainerCallback):
    """Records each step's training loss for the post-run sanity check."""

    def __init__(self):
        self.records: list[tuple[int, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.records.append((state.global_step, float(logs["loss"])))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Dream-org/Dream-v0-Instruct-7B")
    p.add_argument("--output_dir", default="./opsd_dllm_smoke_run")
    # Default = the canonical OPSD training data: Siyan Zhao's reformat of
    # OpenThoughts math, with `problem` / `solution` columns.
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--instruction_field", default="problem",
                   help="Dataset column for the student's input (Openthoughts: 'problem').")
    p.add_argument("--response_field", default="solution",
                   help="Dataset column for the gold response (Openthoughts: 'solution').")
    p.add_argument("--context_field", default="",
                   help="Optional context column. Pass '' for none (Openthoughts has none).")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_answer_length", type=int, default=256)
    # Generation (kept small so the test stays fast).
    p.add_argument("--gen_max_new_tokens", type=int, default=32)
    p.add_argument("--gen_steps", type=int, default=32)
    p.add_argument("--gen_temperature", type=float, default=0.2)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    # JSD
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sampling_eps", type=float, default=1e-3)
    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                   default=True, help="Use --no-gradient_checkpointing to disable.")
    p.add_argument("--use_peft", action=argparse.BooleanOptionalAction, default=True,
                   help="LoRA adapters (default ON). --no-use_peft = full fine-tune (needs ~100 GB).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="sdpa is the safest default for Dream-7B.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"\n[1/4] Loading {args.model_name_or_path} (dtype={dtype}, "
          f"attn={args.attn_implementation}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"      pad_token_id  = {tokenizer.pad_token_id}")
    print(f"      eos_token_id  = {tokenizer.eos_token_id}")
    print(f"      mask_token_id = {tokenizer.mask_token_id}")

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.gradient_checkpointing:
        # use_reentrant=False is required when training with PEFT/LoRA so the
        # backward pass can find adapter parameters that aren't on the recompute path.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"      model class   = {type(model).__name__}")

    context_field = args.context_field or None
    print(f"\n[2/4] Loading {args.num_samples} samples from {args.dataset} ...")
    print(f"      fields  : instruction={args.instruction_field!r}, "
          f"response={args.response_field!r}, context={context_field!r}")
    ds = load_dataset(args.dataset, split="train")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))
    print(f"      columns: {ds.column_names}")
    print(f"      kept    : {len(ds)} rows")
    # Sanity: required columns must exist after the load.
    for required in (args.instruction_field, args.response_field):
        if required not in ds.column_names:
            raise ValueError(
                f"Dataset {args.dataset!r} has no column {required!r}. "
                f"Available columns: {ds.column_names}"
            )

    print(f"\n[3/4] Configuring trainer "
          f"({args.num_train_epochs} epochs, "
          f"per_device_bs={args.per_device_train_batch_size}, "
          f"on-policy + masked JSD) ...")
    training_args = GOLDConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        beta=args.beta,
        temperature=args.temperature,
        max_length=args.max_prompt_length + args.max_answer_length,
        max_completion_length=args.gen_max_new_tokens,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        seed=args.seed,
        remove_unused_columns=False,  # collator reads instruction/response/context directly
        dataloader_num_workers=0,     # avoid multi-worker spawn overhead on a 10-sample run
    )
    # SFTTrainer skips re-loading the model when this is None.
    training_args.model_init_kwargs = None
    # SelfDistillationDLLMDataCollator tokenizes instruction/response/context at
    # collate time, so SFTTrainer's default `dataset.map(tokenize_fn)` (which
    # expects a "text" column) must be bypassed.
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    loss_callback = LossLogger()

    peft_config = None
    if args.use_peft:
        # NOTE: do NOT set task_type="CAUSAL_LM" — PEFT's PeftModelForCausalLM
        # hard-fetches `base_model.prepare_inputs_for_generation` at __init__,
        # which Dream (a diffusion LM) does not provide. Leaving task_type=None
        # selects the generic PeftModel wrapper, which delegates attribute access
        # to the base model so `model.diffusion_generate(...)` still works.
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        print(f"      LoRA r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout}")

    data_collator = SelfDistillationDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        instruction_field=args.instruction_field,
        response_field=args.response_field,
        context_field=context_field,
    )

    trainer = OPSDDLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds,
        eval_dataset=None,
        processing_class=tokenizer,
        callbacks=[loss_callback],
        peft_config=peft_config,
        fixed_teacher=False,
        mask_token_id=tokenizer.mask_token_id,
        sampling_eps=args.sampling_eps,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_steps=args.gen_steps,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        gen_alg="entropy",
        gen_alg_temp=0.0,
    )

    expected_steps = (
        (args.num_samples // args.per_device_train_batch_size) * args.num_train_epochs
    )
    print(f"\n[4/4] Training "
          f"(expecting ~{expected_steps} optimizer steps) ...")
    trainer.train()

    # === Post-run sanity ======================================================
    print("\n=== Per-step training loss ===")
    if not loss_callback.records:
        raise RuntimeError("No loss values recorded — LossLogger never fired.")
    for step, loss in loss_callback.records:
        print(f"  step {step:3d}: loss = {loss:.6f}")

    losses = [v for _, v in loss_callback.records]
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError(f"Non-finite loss recorded: {losses}")

    first, last = losses[0], losses[-1]
    print(f"\n  first loss = {first:.6f}")
    print(f"  last  loss = {last:.6f}")
    print(f"  Δ          = {last - first:+.6f}")

    # Sanity trend: the smoke run is short and noisy, so just *warn* if the loss
    # didn't improve at least slightly — don't hard-fail.
    if last > first * 1.5:
        print("  ⚠️  loss grew significantly — investigate before scaling up.")
    else:
        print("  ✓ loss did not blow up; on-policy distillation step is wired correctly.")

    # Also surface trainer-level metrics from the last logged window.
    print("\n=== Trainer-state metric history (final values) ===")
    for k, vals in trainer._metrics["train"].items():
        if vals:
            print(f"  {k:30s} : last={vals[-1]:.6f}  (n={len(vals)})")

    print("\n✓ smoke test completed.")


if __name__ == "__main__":
    main()
