import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, TrainerCallback
from trl import TrlParser, ModelConfig
from peft import LoraConfig
from transformers.trainer_utils import get_last_checkpoint
import os
import warnings

# Custom imports
from d_opsd_trainer import dOPSDTrainer
from d_opsd_config import dOPSDConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_math_questions,
)
from utils import set_random_seed


class AdapterSnapshotCallback(TrainerCallback):
    """Every `every` steps, save a bf16 LoRA-adapter-only snapshot (no optimizer state).

    Cheap evaluation checkpoints; full resumable checkpoints are left to save_steps /
    save_total_limit. ZeRO-2 keeps full parameters on every rank, so rank 0 saves alone.
    """

    def __init__(self, output_dir, every):
        self.output_dir = output_dir
        self.every = every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every != 0 or not state.is_world_process_zero:
            return
        peft_model = getattr(model, "module", model)
        # named_parameters, not state_dict(): the latter re-serializes every 4-bit base weight.
        lora_sd = {k: v.detach().to(torch.bfloat16) for k, v in peft_model.named_parameters() if "lora_" in k}
        path = os.path.join(self.output_dir, "adapters", f"step-{state.global_step}")
        peft_model.save_pretrained(path, state_dict=lora_sd)
        print(f"[adapter snapshot] step {state.global_step} -> {path}")


def main(opsd_config, model_config):
    # Set seed for reproducibility
    set_random_seed(opsd_config.seed)

    # Load dataset based on configuration
    if opsd_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions(split="train", add_ref=opsd_config.add_ref)
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif opsd_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif opsd_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif opsd_config.dataset == "math":
        dataset = get_math_questions("train", add_ref=opsd_config.add_ref)
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=opsd_config.seed)

    # Split dataset if needed
    if opsd_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        opsd_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(opsd_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    if opsd_config.activation_checkpointing:
        # LLaDA's own (non-reentrant) checkpointing; the strategy enum is a StrEnum.
        model.model.set_activation_checkpointing(opsd_config.activation_checkpointing)
        print(f"LLaDA activation checkpointing: {opsd_config.activation_checkpointing}")

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = dOPSDTrainer(
        args=opsd_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )
    if opsd_config.adapter_save_steps > 0:
        trainer.add_callback(AdapterSnapshotCallback(opsd_config.output_dir, opsd_config.adapter_save_steps))

    if opsd_config.save_steps % opsd_config.num_iterations != 0:
        warnings.warn(
            f"save_steps ({opsd_config.save_steps}) is not divisible by num_iterations ({opsd_config.num_iterations}). If resuming training from a checkpoint, you might need to manually specify the checkpoint where the training step is divisible by {opsd_config.num_iterations}."
        )

    # resume_from_checkpoint: a checkpoint dir, or "auto" = latest checkpoint-* in output_dir
    # (fresh start if none). Needed to chain jobs past a cluster wall-clock limit.
    resume = opsd_config.resume_from_checkpoint
    if resume in (None, False, "", "false", "False"):
        resume = None
    elif resume == "auto":
        resume = get_last_checkpoint(opsd_config.output_dir) if os.path.isdir(opsd_config.output_dir) else None
    print(f"resume_from_checkpoint = {resume}")

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    parser = TrlParser((dOPSDConfig, ModelConfig))
    opsd_config, model_config = parser.parse_args_and_config()
    main(opsd_config=opsd_config, model_config=model_config)
