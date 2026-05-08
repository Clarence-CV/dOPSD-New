"""Entry point: OPSD self-distillation training for diffusion LLMs (Dream-7B)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from opsd_dllm_trainer import OPSDDLLMTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extra args for dLLM OPSD training."""

    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (base model w/o LoRA adapters) as a fixed teacher. "
            "Requires --use_peft."
        },
    )
    sampling_eps: float = field(
        default=1e-3,
        metadata={"help": "Lower bound on the per-example mask rate (LLaDA / Dream-style)."},
    )
    weight_loss_by_p_mask: bool = field(
        default=False,
        metadata={
            "help": "Reweight the per-token JSD by 1/p_mask (LLaDA-style unbiased estimator). "
            "Off by default — set True only if you know you want this scaling."
        },
    )
    mask_token_id: int = field(
        default=-1,
        metadata={"help": "Mask token id; -1 = use tokenizer.mask_token_id."},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Max length (in tokens) of the chat-templated student/teacher prompt."},
    )
    max_answer_length: int = field(
        default=1024,
        metadata={"help": "Max length (in tokens) of the answer span — the region that gets masked."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD to top-k teacher tokens. 0 = full vocabulary."},
    )
    jsd_token_clip: float = field(
        default=0.0,
        metadata={"help": "Per-element JSD clip. 0 = disabled."},
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
        run_name = f"opsd_dllm_{model_name}_lr{lr_str}_bs{effective_bs}_ans{script_args.max_answer_length}"
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
                "sampling_eps": script_args.sampling_eps,
                "weight_loss_by_p_mask": script_args.weight_loss_by_p_mask,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "jsd_token_clip": script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
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

    # === Model (Dream-7B is a diffusion LM — use AutoModel + trust_remote_code) ===
    model_dtype = _resolve_dtype(model_args)
    print(f"\n{'='*80}\nLoading {model_args.model_name_or_path} with dtype={model_dtype}\n{'='*80}\n")

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

    model = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # SFTTrainer expects this attr; setting empty avoids it trying to re-load the model from str.
    training_args.model_init_kwargs = None

    # === Dataset ==============================================================
    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"]

    # === Trainer ==============================================================
    trainer = OPSDDLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        sampling_eps=script_args.sampling_eps,
        mask_token_id=mask_token_id,
        max_prompt_length=script_args.max_prompt_length,
        max_answer_length=script_args.max_answer_length,
        weight_loss_by_p_mask=script_args.weight_loss_by_p_mask,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
