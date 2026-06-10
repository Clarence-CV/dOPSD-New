"""Entry point: OPSD self-distillation training for diffusion LLMs (Dream-7B or LLaDA-8B)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb

from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from data_collator_dllm import SelfDistillationDLLMDataCollator
from opsd_dllm_trainer import OPSDDLLMTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# === OPSD training data =======================================================
# Each row must yield a `problem` (what the student sees) and a `solution` (the
# privileged step-by-step reasoning the teacher sees in its prompt). Math is the
# right domain for the privileged-info paradigm: the gold solution is
# high-content (sharpens the teacher's distribution) and answers are verifiable.
#
# DATASET_REGISTRY maps a --dataset choice to its hub id and the source columns
# that play the problem/solution roles. The loader renames those two columns to
# the canonical `problem`/`solution` and drops the rest, so adding a dataset is
# just one entry here.
#   * zigeng   — Zigeng's dParallel reformat of OpenThoughts. `llm_response` is
#                the teacher's full CoT (final-answer-only fields are dropped).
#   * mixchain — horseee/MixChain-Z-PRM12K. `answer` is the reference \boxed{}
#                solution; the alternative solution_0..4 / token / correctness
#                columns are dropped.
DATASET_REGISTRY = {
    "zigeng": {
        "id": "Zigeng/dParallel_Dream_Distill_Data",
        "problem_field": "question",
        "solution_field": "llm_response",
    },
    "mixchain": {
        "id": "horseee/MixChain-Z-PRM12K",
        "problem_field": "question",
        "solution_field": "answer",
    },
}
DEFAULT_DATASET = "mixchain"
# Canonical column names the collator consumes after the rename below.
BASELINE_PROBLEM_FIELD = "problem"
BASELINE_SOLUTION_FIELD = "solution"


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extra args for dLLM OPSD training."""

    dataset: str = field(
        default=DEFAULT_DATASET,
        metadata={
            "help": "Training dataset key from DATASET_REGISTRY. "
            f"Choices: {sorted(DATASET_REGISTRY)}. 'zigeng' = dParallel OpenThoughts "
            "reformat (default); 'mixchain' = horseee/MixChain-Z-PRM12K "
            "(question -> problem, answer -> solution)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (base model w/o LoRA adapters) as a fixed teacher. "
            "Requires --use_peft."
        },
    )
    off_policy: bool = field(
        default=False,
        metadata={
            "help": "Off-policy mode: distill on the dataset's ground-truth answer instead "
            "of an on-policy student rollout. Student sees the masked GT; teacher sees the "
            "concrete GT. When set, --gen_* generation flags are unused."
        },
    )
    use_privileged_info: bool = field(
        default=False,
        metadata={
            "help": "Privileged-information teacher: embed the ground-truth solution in the "
            "teacher's prompt (student still sees the problem only). False (default) keeps the "
            "teacher prompt identical to the student's (no-PI baseline). This is the single knob "
            "the PI-vs-no-PI controlled experiment toggles."
        },
    )
    student_backend: str = field(
        default="dream",
        metadata={
            "help": "Student diffusion backend: 'dream' (Dream-7B, AutoModel) or "
            "'llada' (GSAI LLaDA, AutoModelForCausalLM). Selects the model class, "
            "the PEFT task type, and the logits/attention-mask path inside "
            "OPSDDLLMTrainer (Dream-shifted vs. standard CausalLM)."
        },
    )
    mask_token_id: int = field(
        default=-1,
        metadata={"help": "Mask token id; -1 = use tokenizer.mask_token_id."},
    )
    sampling_eps: float = field(
        default=1e-3,
        metadata={"help": "Lower bound on the antithetic per-example mask rate over the on-policy completion."},
    )
    mask_schedule: str = field(
        default="diffusion",
        metadata={
            "help": "Mask-sampling strategy over the JSD-valid completion positions: "
            "'diffusion' (antithetic per-example rate, i.i.d. Bernoulli per valid "
            "position — original OPSD behavior) or 'fixed' (exact-count "
            "k = round(n_valid * fixed_mask_ratio) per example, mirroring "
            "tabom_test_code.py's fixed schedule)."
        },
    )
    fixed_mask_ratio: str = field(
        default="0.75",
        metadata={
            "help": "Only used with --mask_schedule=fixed. Either a single float "
            "('0.75') or a 'lo:hi' range ('0.25:0.75') sampled uniformly per batch."
        },
    )
    diffusion_min_t: float = field(
        default=0.0,
        metadata={"help": "Only used with --mask_schedule=diffusion. Lower bound on per-example mask rate (before sampling_eps floor)."},
    )
    diffusion_max_t: float = field(
        default=1.0,
        metadata={"help": "Only used with --mask_schedule=diffusion. Upper bound on per-example mask rate."},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Max length (in tokens) of the chat-templated student/teacher prompt."},
    )
    max_answer_length: int = field(
        default=1024,
        metadata={"help": "Max length (in tokens) of the gold answer span (used by the collator only)."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD to top-k teacher tokens. 0 = full vocabulary."},
    )
    jsd_token_clip: float = field(
        default=0.0,
        metadata={"help": "Per-element JSD clip. 0 = disabled."},
    )
    # Generation hyperparameters for the on-policy student rollout (Dream's diffusion_generate).
    gen_max_new_tokens: int = field(
        default=256,
        metadata={"help": "Number of completion tokens generated by the student per example."},
    )
    gen_steps: int = field(
        default=256,
        metadata={"help": "Number of diffusion denoising steps used during generation."},
    )
    gen_temperature: float = field(
        default=0.2,
        metadata={"help": "Sampling temperature for diffusion_generate."},
    )
    gen_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p (nucleus) for diffusion_generate."},
    )
    gen_alg: str = field(
        default="entropy",
        metadata={"help": "Dream sampler algorithm (e.g. 'entropy')."},
    )
    gen_alg_temp: float = field(
        default=0.0,
        metadata={"help": "Temperature for the Dream sampler scoring fn (0 = greedy ordering)."},
    )
    run_config: str = field(
        default=None,
        metadata={"help": "Suffix for output_dir and W&B run name."},
    )


def _resolve_dtype(model_args):
    dtype_attr = getattr(model_args, "torch_dtype", None) or getattr(model_args, "dtype", None)
    if dtype_attr is None:
        return torch.bfloat16
    if isinstance(dtype_attr, str):
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_attr.lower(), torch.bfloat16)
    return dtype_attr


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.student_backend == "llada":
        # LLaDA is loaded as AutoModelForCausalLM, so the standard
        # PeftModelForCausalLM wrapper is correct: prepare_inputs_for_generation
        # exists on the base, and HF CausalLM forward is what the trainer's
        # LLaDA path calls.
        model_args.lora_task_type = "CAUSAL_LM"
    else:
        # Dream is a diffusion LM, not an AR causal LM. PeftModelForCausalLM
        # fetches `base_model.prepare_inputs_for_generation` at __init__, which
        # Dream does not expose — leading to AttributeError. Forcing task_type=None
        # selects PEFT's generic PeftModel wrapper instead, which delegates
        # attribute access to the base model so `diffusion_generate` still works.
        model_args.lora_task_type = None

    # === Run / output naming ==================================================
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_bs = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )

    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_bs}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        run_name = (
            f"opsd_dllm_{script_args.student_backend}_{model_name}"
            f"_lr{lr_str}_bs{effective_bs}_ans{script_args.max_answer_length}"
        )
        if script_args.fixed_teacher:
            run_name += "_fixteach"

    print(f"\n{'='*80}\nRUN CONFIGURATION\n{'='*80}")
    print(f"WandB run: {run_name}")
    print(f"Output dir: {training_args.output_dir}")
    print(f"{'='*80}\n")

    # === Validate flag combinations ===========================================
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True (LoRA-disable serves as the fixed teacher).")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=getattr(training_args, "wandb_entity", None),
            project=getattr(training_args, "wandb_project", None),
            name=run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_bs,
                "num_train_epochs": training_args.num_train_epochs,
                "max_prompt_length": script_args.max_prompt_length,
                "max_answer_length": script_args.max_answer_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "fixed_teacher": script_args.fixed_teacher,
                "off_policy": script_args.off_policy,
                "use_privileged_info": script_args.use_privileged_info,
                "student_backend": script_args.student_backend,
                "mask_schedule": script_args.mask_schedule,
                "fixed_mask_ratio": script_args.fixed_mask_ratio,
                "diffusion_min_t": script_args.diffusion_min_t,
                "diffusion_max_t": script_args.diffusion_max_t,
                "sampling_eps": script_args.sampling_eps,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
                "gen_max_new_tokens": script_args.gen_max_new_tokens,
                "gen_steps": script_args.gen_steps,
                "gen_temperature": script_args.gen_temperature,
                "gen_top_p": script_args.gen_top_p,
                "gen_alg": script_args.gen_alg,
                "gen_alg_temp": script_args.gen_alg_temp,
            },
        )

    # === Tokenizer ============================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mask_token_id = (
        script_args.mask_token_id if script_args.mask_token_id >= 0 else tokenizer.mask_token_id
    )
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; pass --mask_token_id explicitly.")
    print(f"[opsd_dllm_train] Using mask_token_id = {mask_token_id}")

    # === Model ================================================================
    # Dream-7B: diffusion LM exposed via AutoModel + trust_remote_code (no
    #           standard HF CausalLM interface; generation is `diffusion_generate`).
    # LLaDA-8B: standard HF CausalLM, loaded via AutoModelForCausalLM. The
    #           trainer's LLaDA forward path expects this class so logits keep
    #           the `logits[:, i] predicts token i` semantics (see
    #           `tabom_test_code.py` and `OPSDDLLMTrainer._forward`).
    model_dtype = _resolve_dtype(model_args)
    print(
        f"\n{'='*80}\nLoading {model_args.model_name_or_path} "
        f"(student_backend={script_args.student_backend}, dtype={model_dtype})\n{'='*80}\n"
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    if script_args.student_backend == "llada":
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if training_args.gradient_checkpointing:
        # use_reentrant=False is required when training with PEFT/LoRA: the
        # default reentrant mode needs at least one input tensor with
        # requires_grad=True, and integer input_ids / boolean attention_mask
        # carry no grad. Without this kwarg the backward through the LoRA
        # adapters can silently produce wrong gradients.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # SFTTrainer expects this attr; setting empty avoids it trying to re-load the model from str.
    training_args.model_init_kwargs = None

    # === Dataset ==============================================================
    if script_args.dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"--dataset must be one of {sorted(DATASET_REGISTRY)} (got {script_args.dataset!r})."
        )
    ds_cfg = DATASET_REGISTRY[script_args.dataset]
    src_problem, src_solution = ds_cfg["problem_field"], ds_cfg["solution_field"]
    print(f"\n[opsd_dllm_train] Loading dataset {script_args.dataset!r}: {ds_cfg['id']}")
    print(f"    {src_problem!r} -> {BASELINE_PROBLEM_FIELD!r}")
    print(f"    {src_solution!r} -> {BASELINE_SOLUTION_FIELD!r}")
    dataset = load_dataset(ds_cfg["id"])
    train_dataset = dataset["train"]

    # Rename the dataset's problem/solution columns to the canonical
    # `problem`/`solution` the collator consumes, then drop every other column
    # (final-answer-only fields, alternative solutions, token counts, ...).
    missing = [c for c in (src_problem, src_solution) if c not in train_dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset {ds_cfg['id']} is missing expected column(s) {missing}; "
            f"has {train_dataset.column_names}."
        )
    train_dataset = train_dataset.rename_columns(
        {src_problem: BASELINE_PROBLEM_FIELD, src_solution: BASELINE_SOLUTION_FIELD}
    )
    train_dataset = train_dataset.select_columns(
        [BASELINE_PROBLEM_FIELD, BASELINE_SOLUTION_FIELD]
    )
    print(f"[opsd_dllm_train] Columns -> {train_dataset.column_names}")

    # Build the collator with field names matching the dataset columns. This
    # also drives `_set_signature_columns_if_needed`, so `_remove_unused_columns`
    # keeps these columns through to collate time instead of dropping them.
    data_collator = SelfDistillationDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_answer_length=script_args.max_answer_length,
        problem_field=BASELINE_PROBLEM_FIELD,
        solution_field=BASELINE_SOLUTION_FIELD,
        use_privileged_info=script_args.use_privileged_info,
    )

    # === Trainer ==============================================================
    trainer = OPSDDLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        off_policy=script_args.off_policy,
        student_backend=script_args.student_backend,
        mask_token_id=mask_token_id,
        sampling_eps=script_args.sampling_eps,
        mask_schedule=script_args.mask_schedule,
        fixed_mask_ratio=script_args.fixed_mask_ratio,
        diffusion_min_t=script_args.diffusion_min_t,
        diffusion_max_t=script_args.diffusion_max_t,
        max_prompt_length=script_args.max_prompt_length,
        max_answer_length=script_args.max_answer_length,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        gen_max_new_tokens=script_args.gen_max_new_tokens,
        gen_steps=script_args.gen_steps,
        gen_temperature=script_args.gen_temperature,
        gen_top_p=script_args.gen_top_p,
        gen_alg=script_args.gen_alg,
        gen_alg_temp=script_args.gen_alg_temp,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
