"""Entry point: vanilla SFT training for diffusion LLMs (e.g. Dream-7B).

LLaDA-style masked-token objective: per example sample mask rate t ~ U[eps, 1],
mask each answer token i.i.d. with prob t, bidirectional forward over
[prompt | noisy_answer], CE on masked answer positions weighted by 1/t (the
diffusion ELBO importance weight). Prompt and right-pad tokens are never masked.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# Maps a --dataset choice to its hub id + source columns for `problem`/`solution`;
# the loader renames them and drops the rest, so adding a dataset is one entry.
DATASET_REGISTRY = {
    "mixchain": {
        "id": "horseee/MixChain-Z-PRM12K",
        "problem_field": "question",
        "solution_field": "answer",
    },
}
DEFAULT_DATASET = "mixchain"
# Canonical column names the collator consumes after the rename in main().
PROBLEM_FIELD = "problem"
SOLUTION_FIELD = "solution"


@dataclass
class DLLMScriptArguments(ScriptArguments):
    dataset: str = field(
        default=DEFAULT_DATASET,
        metadata={
            "help": "Training dataset key from DATASET_REGISTRY. "
            f"Choices: {sorted(DATASET_REGISTRY)}. 'mixchain' = horseee/MixChain-Z-PRM12K "
            "(question -> problem, answer -> solution)."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Optional run name suffix/config. Also appended to output_dir when set."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={
            "help": "WandB entity/user/team. Defaults to WANDB_ENTITY; leave unset for the logged-in user."
        },
    )
    wandb_project: str = field(
        default=None,
        metadata={"help": "WandB project. Defaults to WANDB_PROJECT or 'sft-dllm'."},
    )
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Disable WandB logging even if wandb is installed/logged in."},
    )
    mask_token_id: int = field(
        default=-1,
        metadata={"help": "Mask token id; -1 = use tokenizer.mask_token_id."},
    )
    sampling_eps: float = field(
        default=1e-3,
        metadata={"help": "Lower bound on the per-example answer mask rate."},
    )
    mask_schedule: str = field(
        default="diffusion",
        metadata={
            "help": "Mask-sampling strategy over answer positions (mirrors OPSDDLLMTrainer): "
            "'diffusion' (antithetic per-example rate t ∈ [diffusion_min_t, diffusion_max_t], "
            "i.i.d. Bernoulli per valid position — the LLaDA ELBO default) or 'fixed' "
            "(exact-count k = round(n_valid * fixed_mask_ratio) positions per example)."
        },
    )
    fixed_mask_ratio: str = field(
        default="0.75",
        metadata={
            "help": "Only used with --mask_schedule=fixed. Either a single float ('0.75') "
            "or a 'lo:hi' range ('0.25:0.75') sampled uniformly once per batch."
        },
    )
    diffusion_min_t: float = field(
        default=0.0,
        metadata={"help": "Only used with --mask_schedule=diffusion. Lower bound on the per-example mask rate (before sampling_eps floor)."},
    )
    diffusion_max_t: float = field(
        default=1.0,
        metadata={"help": "Only used with --mask_schedule=diffusion. Upper bound on the per-example mask rate."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={
            "help": "Max length (in tokens) of the chat-templated prompt. Keep "
            "max_prompt_length + max_answer_length < 2048 (Dream-v0 position limit)."
        },
    )
    max_answer_length: int = field(
        default=1408,
        metadata={
            "help": "Max length (in tokens) of the gold answer span. Long "
            "OpenThoughts solutions are truncated from the end, so an oversized "
            "solution loses its trailing \\boxed{}. 512 + 1408 = 1920 < 2048."
        },
    )


class SFTDLLMDataCollator:
    """Builds `[prompt | answer]` per example with an answer-region label mask.

    Prompts left-padded, answers right-padded, so pad tokens sit only on the
    outer edges (never between prompt and answer) — matters for bidirectional
    attention.
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 1024,
        max_answer_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, problem: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, features):
        prompts = [self._build_prompt(f["problem"]) for f in features]
        answers = [f["solution"] for f in features]

        p_ids = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        a_ids = self.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=self.max_answer_length,
            add_special_tokens=False,
        )["input_ids"]

        pad_id = self.tokenizer.pad_token_id
        max_p = max(len(x) for x in p_ids)
        max_a = max(len(x) for x in a_ids)
        total_len = max_p + max_a

        batch = len(features)
        input_ids = torch.full((batch, total_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, total_len), dtype=torch.long)
        answer_mask = torch.zeros((batch, total_len), dtype=torch.long)

        for i, (p, a) in enumerate(zip(p_ids, a_ids)):
            p_start = max_p - len(p)          # left-pad prompt
            input_ids[i, p_start:max_p] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, p_start:max_p] = 1

            input_ids[i, max_p:max_p + len(a)] = torch.tensor(a, dtype=torch.long)
            attention_mask[i, max_p:max_p + len(a)] = 1
            answer_mask[i, max_p:max_p + len(a)] = 1  # non-pad answer tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "labels": input_ids.clone(),  # placeholder; trainer overrides
        }


class SFTDLLMTrainer(Trainer):
    """HF Trainer with LLaDA-style masked-prediction loss for diffusion LMs."""

    def __init__(
        self,
        *args,
        mask_token_id: int,
        sampling_eps: float = 1e-3,
        mask_schedule: str = "diffusion",
        fixed_mask_ratio: str = "0.75",
        diffusion_min_t: float = 0.0,
        diffusion_max_t: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mask_schedule not in ("diffusion", "fixed"):
            raise ValueError(
                f"mask_schedule must be 'diffusion' or 'fixed', got {mask_schedule!r}"
            )
        self.mask_token_id = mask_token_id
        self.sampling_eps = sampling_eps
        self.mask_schedule = mask_schedule
        self.fixed_mask_ratio = fixed_mask_ratio
        self.diffusion_min_t = float(diffusion_min_t)
        self.diffusion_max_t = float(diffusion_max_t)

    def _sample_answer_mask(self, answer_mask: torch.Tensor):
        """Sample a mask over the answer span per `self.mask_schedule`.

        Returns mask [B, L] bool (positions to replace with <mask>) and
        p_mask_sample [B] float (per-example mask rate, used as the 1/t ELBO
        weight in compute_loss; constant fixed_mask_ratio in fixed mode).
        Both schedules enforce >=1 masked position per non-empty row.
        """
        if self.mask_schedule == "fixed":
            mask, p_mask_sample = self._fixed_mask(answer_mask)
        else:
            mask, p_mask_sample = self._diffusion_mask(answer_mask)

        # Guarantee >=1 masked position per non-empty row so the loss is defined
        # (done once here rather than per strategy).
        valid = answer_mask.bool()
        B = valid.shape[0]
        device = answer_mask.device
        for i in range(B):
            if valid[i].any() and not mask[i].any():
                idx = valid[i].nonzero(as_tuple=False).flatten()
                pick = idx[torch.randint(0, idx.numel(), (1,), device=device)]
                mask[i, pick] = True
        return mask, p_mask_sample

    def _diffusion_mask(self, answer_mask: torch.Tensor):
        """Antithetic per-example mask rate, i.i.d. Bernoulli per valid position.

        Pair (2k, 2k+1) shares base draw u_k: t=u_k and t=1-u_k (both halves must
        share u_k for the variance reduction). Rate then mapped [0,1] ->
        [max(diffusion_min_t, sampling_eps), diffusion_max_t]; equals the original
        (1-eps)*t + eps when min_t=0, max_t=1.
        """
        B, L = answer_mask.shape
        device = answer_mask.device

        half = (B + 1) // 2
        u = torch.rand(half, device=device)
        t = torch.empty(B, device=device)
        t[0::2] = u[: t[0::2].numel()]
        t[1::2] = 1.0 - u[: t[1::2].numel()]

        lower = max(self.diffusion_min_t, self.sampling_eps)
        upper = self.diffusion_max_t
        if upper <= lower:
            p = torch.full((B,), float(lower), device=device)
        else:
            p = lower + (upper - lower) * t  # [B]

        rand = torch.rand(B, L, device=device)
        valid = answer_mask.bool()
        mask = (rand < p[:, None]) & valid
        return mask, p

    def _fixed_mask(self, answer_mask: torch.Tensor):
        """Exact-count masking: k = round(n_valid * fixed_mask_ratio) per example.

        fixed_mask_ratio is a float ("0.75") or "lo:hi" range ("0.25:0.75"); a
        range samples one ratio per call applied batch-wide. k positions drawn
        uniformly without replacement from each answer span.
        """
        B, L = answer_mask.shape
        device = answer_mask.device
        valid = answer_mask.bool()

        mr = self._sample_fixed_mask_ratio()
        mask = torch.zeros_like(valid)
        for b in range(B):
            positions = valid[b].nonzero(as_tuple=False).flatten()
            n = positions.shape[0]
            if n == 0:
                continue
            k = max(1, int(round(n * mr)))
            k = min(k, n)
            chosen = positions[torch.randperm(n, device=device)[:k]]
            mask[b, chosen] = True
        p_mask_sample = torch.full((B,), float(mr), device=device, dtype=torch.float32)
        return mask, p_mask_sample

    def _sample_fixed_mask_ratio(self) -> float:
        """Parse self.fixed_mask_ratio ('0.75' or '0.25:0.75') into a scalar."""
        s = self.fixed_mask_ratio
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s)
        if ":" in s:
            lo, hi = (float(x) for x in s.split(":", 1))
            return lo + float(torch.rand(1).item()) * (hi - lo)
        return float(s)

    def _forward(self, model, input_ids, attention_mask):
        # Dream passes attention_mask straight to SDPA, so expand (B,L) -> (B,1,1,L).
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].bool()
        return model(input_ids=input_ids, attention_mask=attention_mask)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        answer_mask = inputs["answer_mask"]

        mask, p_mask = self._sample_answer_mask(answer_mask)
        noisy_input_ids = torch.where(
            mask,
            torch.full_like(input_ids, self.mask_token_id),
            input_ids,
        )

        outputs = self._forward(model, noisy_input_ids, attention_mask)
        # Dream convention: raw logits[i] predicts token[i+1]. Right-shift so
        # logits[i] aligns with token[i]; without it the loss is off by one and
        # the model becomes inconsistent with Dream's diffusion_generate.
        logits = outputs.logits  # [B, L, V]
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # LLaDA ELBO: per example (1/t)*SUM CE over masked tokens, normalized by
        # answer length (constant w.r.t. the mask), then batch-mean. Normalizing
        # by masked-token count would be wrong — that count ~ t, cancelling the
        # explicit 1/t weight and over-weighting low-mask examples.
        flat_logits = logits[mask]                    # [N, V]
        flat_targets = input_ids[mask]                # [N]
        per_token_ce = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")  # [N]

        B = input_ids.size(0)
        ex_idx = mask.nonzero(as_tuple=False)[:, 0]   # [N] source row of each masked token
        weights = (1.0 / p_mask.float())[ex_idx].to(per_token_ce.dtype)  # [N] 1/t ELBO weight

        answer_lengths = answer_mask.sum(dim=1).to(per_token_ce.dtype)  # [B]
        per_ex_loss = torch.zeros(B, device=input_ids.device, dtype=per_token_ce.dtype)
        per_ex_loss.index_add_(0, ex_idx, (per_token_ce * weights).to(per_ex_loss.dtype))
        per_ex_loss = per_ex_loss / answer_lengths.clamp_min(1.0)
        has_answer = answer_lengths > 0
        loss = per_ex_loss[has_answer].mean()

        return (loss, outputs) if return_outputs else loss


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


def init_wandb_or_disable(script_args, training_args, run_name: str, config: dict):
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return

    if script_args.disable_wandb or os.environ.get("WANDB_MODE") == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        training_args.report_to = []
        print("[sft_train_dllm] WandB disabled.")
        return

    entity = script_args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    project = script_args.wandb_project or os.environ.get("WANDB_PROJECT") or "sft-dllm"

    try:
        wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            config=config,
        )
        training_args.run_name = run_name
    except Exception as exc:
        os.environ["WANDB_MODE"] = "disabled"
        training_args.report_to = []
        print(
            "[sft_train_dllm] Warning: WandB init failed; continuing with WandB disabled. "
            f"Reason: {type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    parser = TrlParser((DLLMScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Dream is a diffusion LM; use the generic PEFT wrapper so attribute access
    # (e.g. diffusion_generate) delegates to the base model.
    model_args.lora_task_type = None

    # WandB Run Name
    model_name = model_args.model_name_or_path.split("/")[-1]
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )
    full_wandb_run_name = (
        script_args.run_config
        or f"SFT_DLLM_{model_name}_lr{lr_str}_bs{effective_batch_size}_ep{training_args.num_train_epochs}"
    )
    if script_args.run_config and not training_args.output_dir.endswith(script_args.run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)

    # WandB Initialization
    wandb_config = {
        "model_name": model_args.model_name_or_path,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "max_prompt_length": script_args.max_prompt_length,
        "max_answer_length": script_args.max_answer_length,
        "sampling_eps": script_args.sampling_eps,
        "mask_schedule": script_args.mask_schedule,
        "fixed_mask_ratio": script_args.fixed_mask_ratio,
        "diffusion_min_t": script_args.diffusion_min_t,
        "diffusion_max_t": script_args.diffusion_max_t,
        "use_peft": model_args.use_peft,
        "lora_r": model_args.lora_r if model_args.use_peft else None,
        "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "num_processes": num_processes,
        "trainer": "SFTDLLMTrainer",
    }
    init_wandb_or_disable(script_args, training_args, full_wandb_run_name, wandb_config)
    training_args.remove_unused_columns = False

    # Tokenizer
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
    print(f"[sft_train_dllm] Using mask_token_id = {mask_token_id}")

    # Model
    model_dtype = _resolve_dtype(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=True,
        # Default to sdpa: Dream-7B is unsafe with flash_attention_2.
        attn_implementation=model_args.attn_implementation or "sdpa",
        dtype=model_dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    try:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    if training_args.gradient_checkpointing:
        # use_reentrant=False required for PEFT/LoRA.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Avoid HF Trainer trying to re-load the model from a string path.
    training_args.model_init_kwargs = None

    # Dataset
    if script_args.dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"--dataset must be one of {sorted(DATASET_REGISTRY)} (got {script_args.dataset!r})."
        )
    ds_cfg = DATASET_REGISTRY[script_args.dataset]
    src_problem, src_solution = ds_cfg["problem_field"], ds_cfg["solution_field"]
    print(f"[sft_train_dllm] Loading dataset {script_args.dataset!r}: {ds_cfg['id']}")
    print(f"    {src_problem!r} -> {PROBLEM_FIELD!r}  /  {src_solution!r} -> {SOLUTION_FIELD!r}")
    dataset = load_dataset(ds_cfg["id"])
    train_dataset = dataset["train"]
    # Rename problem/solution to canonical names, drop every other column.
    missing = [c for c in (src_problem, src_solution) if c not in train_dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset {ds_cfg['id']} is missing expected column(s) {missing}; "
            f"has {train_dataset.column_names}."
        )
    train_dataset = train_dataset.rename_columns(
        {src_problem: PROBLEM_FIELD, src_solution: SOLUTION_FIELD}
    )
    train_dataset = train_dataset.select_columns([PROBLEM_FIELD, SOLUTION_FIELD])
    split_dataset = train_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Collator
    data_collator = SFTDLLMDataCollator(
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_answer_length=script_args.max_answer_length,
    )

    # Training
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)

    trainer = SFTDLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        mask_token_id=mask_token_id,
        sampling_eps=script_args.sampling_eps,
        mask_schedule=script_args.mask_schedule,
        fixed_mask_ratio=script_args.fixed_mask_ratio,
        diffusion_min_t=script_args.diffusion_min_t,
        diffusion_max_t=script_args.diffusion_max_t,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
