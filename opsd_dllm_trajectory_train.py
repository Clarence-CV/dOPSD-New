"""Entry point: Trajectory-OPSD self-distillation for diffusion LLMs (Dream-7B or LLaDA-8B).

Same plumbing as `opsd_dllm_train.py`, but drives `OPSDDLLMTrajectoryTrainer`:
the student's noisy view is a real intermediate decoding step (least-masked
step with >`traj_mask_threshold` masked tokens) and the teacher's privileged
information is the concrete final rollout. The off-policy / synthetic-mask
flags (`--off_policy`, `--mask_schedule`, `--fixed_mask_ratio`,
`--diffusion_min_t/max_t`) do not apply here and are replaced by
`--traj_mask_threshold` and `--traj_step_select`.
"""

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

from data_collator_dllm_trajectory import SelfDistillationDLLMTrajectoryDataCollator
from opsd_dllm_trajectory_trainer import OPSDDLLMTrajectoryTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# Mirrors opsd_dllm_train.py's registry (problem/solution source columns).
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
BASELINE_PROBLEM_FIELD = "problem"
BASELINE_SOLUTION_FIELD = "solution"


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extra args for trajectory-OPSD dLLM training."""

    dataset: str = field(
        default=DEFAULT_DATASET,
        metadata={
            "help": "Training dataset key from DATASET_REGISTRY. "
            f"Choices: {sorted(DATASET_REGISTRY)}."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (base model w/o LoRA adapters) as a fixed teacher. "
            "Requires --use_peft."
        },
    )
    student_backend: str = field(
        default="dream",
        metadata={
            "help": "Student diffusion backend: 'dream' (Dream-7B, AutoModel) or "
            "'llada' (GSAI LLaDA, AutoModelForCausalLM)."
        },
    )
    mask_token_id: int = field(
        default=-1,
        metadata={"help": "Mask token id; -1 = use tokenizer.mask_token_id."},
    )
    sampling_eps: float = field(
        default=1e-3,
        metadata={"help": "Lower bound on mask rate (kept for parity; unused by trajectory masking)."},
    )
    # === Trajectory-mode knobs ===============================================
    traj_mask_threshold: float = field(
        default=0.5,
        metadata={
            "help": "A decoding step is eligible as the student's noisy view only "
            "if its masked fraction over the JSD-valid region exceeds this value "
            "(default 0.5 = '>50% masked')."
        },
    )
    traj_step_select: str = field(
        default="least",
        metadata={
            "help": "Which eligible decoding step to use: 'least' (smallest masked "
            "fraction still above threshold; default), 'most' (noisiest), or "
            "'random' (uniform among eligible)."
        },
    )
    traj_teacher_view: str = field(
        default="snapshot",
        metadata={
            "help": "Teacher target construction. 'snapshot' (default) = a single "
            "teacher forward on one state (see --traj_teacher_gap). 'all_future' = "
            "average the teacher's predictive distribution over ALL remaining steps "
            "k+1->final where each scored position is still masked (one teacher "
            "forward per remaining step — EXPENSIVE, ~10-50x slower)."
        },
    )
    traj_teacher_gap: int = field(
        default=-1,
        metadata={
            "help": "Only used with --traj_teacher_view=snapshot. -1 (default) = the "
            "concrete final rollout (trajectory endpoint; teacher sees the answer at "
            "every scored position). n >= 0 = the trajectory state n steps after the "
            "student's step (history[k+n], clamped to the final state) — the teacher "
            "gains extra context but positions still masked at k+n stay predictive."
        },
    )
    verify_gold_pi: bool = field(
        default=True,
        metadata={
            "help": "Verify the student's final rollout against the gold solution: "
            "CORRECT rollouts keep the on-policy trajectory PI; WRONG rollouts get the "
            "FULL gold solution as the teacher's privileged context (the teacher re-scores "
            "the student's own noisy rollout at the masked positions). Set False to disable "
            "(pure trajectory PI for every example)."
        },
    )
    gold_pi_max_solution_tokens: int = field(
        default=-1,
        metadata={
            "help": "Only used with --verify_gold_pi. Truncate the gold solution to at most "
            "this many tokens when it is prepended as the teacher's privileged context. "
            "-1 (default) = no extra cap (the collator already caps it at --max_answer_length)."
        },
    )
    decode_intro_prompt: str = field(
        default="",
        metadata={
            "help": "Framing text inserted in the teacher prompt BEFORE the decoding "
            "step. Default '' = OFF (recommended) — the teacher prompt equals the "
            "student prompt, so the teacher input stays in the model's native decode "
            "format and the privilege comes only from the completion tokens. Pass "
            "explicit text to enable framing (out-of-distribution; for ablations)."
        },
    )
    transition_prompt: str = field(
        default="",
        metadata={
            "help": "Transition text appended to the teacher input AFTER the decoding "
            "step. Default '' = DISABLED (no suffix appended) — recommended, since a "
            "teacher-only suffix the student never sees adds out-of-distribution "
            "context. Pass explicit text to enable; the collator also has a built-in "
            "default available by editing DEFAULT_TRANSITION_PROMPT."
        },
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
    # Generation hyperparameters for the on-policy student rollout.
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
        model_args.lora_task_type = "CAUSAL_LM"
    else:
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
            f"opsd_dllm_traj_{script_args.student_backend}_{model_name}"
            f"_lr{lr_str}_bs{effective_bs}_ans{script_args.max_answer_length}"
        )
        if script_args.fixed_teacher:
            run_name += "_fixteach"

    print(f"\n{'='*80}\nRUN CONFIGURATION (TRAJECTORY-OPSD)\n{'='*80}")
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
                "student_backend": script_args.student_backend,
                "method": "trajectory",
                "traj_mask_threshold": script_args.traj_mask_threshold,
                "traj_step_select": script_args.traj_step_select,
                "traj_teacher_view": script_args.traj_teacher_view,
                "traj_teacher_gap": (
                    None if script_args.traj_teacher_gap < 0 else script_args.traj_teacher_gap
                ),
                "verify_gold_pi": script_args.verify_gold_pi,
                "gold_pi_max_solution_tokens": (
                    None if script_args.gold_pi_max_solution_tokens < 0 else script_args.gold_pi_max_solution_tokens
                ),
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
    print(f"[opsd_dllm_trajectory_train] Using mask_token_id = {mask_token_id}")

    # === Model ================================================================
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
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    training_args.model_init_kwargs = None

    # === Dataset ==============================================================
    if script_args.dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"--dataset must be one of {sorted(DATASET_REGISTRY)} (got {script_args.dataset!r})."
        )
    ds_cfg = DATASET_REGISTRY[script_args.dataset]
    src_problem, src_solution = ds_cfg["problem_field"], ds_cfg["solution_field"]
    print(f"\n[opsd_dllm_trajectory_train] Loading dataset {script_args.dataset!r}: {ds_cfg['id']}")
    dataset = load_dataset(ds_cfg["id"])
    train_dataset = dataset["train"]

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
    print(f"[opsd_dllm_trajectory_train] Columns -> {train_dataset.column_names}")

    data_collator = SelfDistillationDLLMTrajectoryDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_answer_length=script_args.max_answer_length,
        problem_field=BASELINE_PROBLEM_FIELD,
        solution_field=BASELINE_SOLUTION_FIELD,
        decode_intro_prompt=script_args.decode_intro_prompt,
        transition_prompt=script_args.transition_prompt,
    )

    # === Trainer ==============================================================
    trainer = OPSDDLLMTrajectoryTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        student_backend=script_args.student_backend,
        mask_token_id=mask_token_id,
        sampling_eps=script_args.sampling_eps,
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
        traj_mask_threshold=script_args.traj_mask_threshold,
        traj_step_select=script_args.traj_step_select,
        traj_teacher_view=script_args.traj_teacher_view,
        traj_teacher_gap=(None if script_args.traj_teacher_gap < 0 else script_args.traj_teacher_gap),
        verify_gold_pi=script_args.verify_gold_pi,
        gold_pi_max_solution_tokens=(
            None if script_args.gold_pi_max_solution_tokens < 0 else script_args.gold_pi_max_solution_tokens
        ),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
