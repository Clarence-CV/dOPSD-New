# Copyright 2026 The Foundation AI Team. Licensed under the Apache License, Version 2.0.
"""On-Policy Self-Distillation (OPSD) trainer for diffusion LLMs (e.g. Dream-7B).

Same model plays both roles: the *teacher* receives the ground-truth solution
in its prompt as privileged context, while the *student* receives only the
problem. The student rolls out a completion via Dream's `diffusion_generate`;
a random mask pattern is then sampled over the valid (non-pad / non-post-EOS)
completion positions; both prompts are concatenated with the *noisy* (masked)
completion and run through a forward pass; JSD is computed only at the
masked positions:

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
"""

import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType, is_peft_model
from datasets import Dataset
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import disable_dropout_in_model, empty_cache
from trl.experimental.gold.gold_config import GOLDConfig

from data_collator_dllm import SelfDistillationDLLMDataCollator


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb


class OPSDDLLMTrainer(SFTTrainer):
    """OPSD trainer for diffusion language models."""

    _tag_names = ["trl", "opsd-dllm"]
    _name = "OPSD-DLLM"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        fixed_teacher: bool = False,
        mask_token_id: int | None = None,
        sampling_eps: float = 1e-3,
        max_prompt_length: int = 1024,
        max_answer_length: int = 1024,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        # Generation hyperparameters for Dream's diffusion_generate (student rollout).
        gen_max_new_tokens: int = 256,
        gen_steps: int = 256,
        gen_temperature: float = 0.2,
        gen_top_p: float = 0.95,
        gen_alg: str = "entropy",
        gen_alg_temp: float = 0.0,
    ):
        if data_collator is None:
            data_collator = SelfDistillationDLLMDataCollator(
                tokenizer=processing_class,
                max_prompt_length=max_prompt_length,
                max_answer_length=max_answer_length,
            )

        # SelfDistillationDLLMDataCollator tokenizes instruction/response/context
        # at collate time. SFTTrainer's default pre-tokenization expects a "text"
        # column and would crash on Dolly-style data; bypass it unconditionally.
        if args is not None:
            dk = dict(getattr(args, "dataset_kwargs", None) or {})
            dk.setdefault("skip_prepare_dataset", True)
            args.dataset_kwargs = dk

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(self.model)

        self.beta = args.beta
        self.temperature = getattr(args, "temperature", 1.0) or 1.0
        self.fixed_teacher = fixed_teacher
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.sampling_eps = sampling_eps

        # diffusion_generate hyperparameters
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_steps = gen_steps
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_alg = gen_alg
        self.gen_alg_temp = gen_alg_temp

        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif getattr(processing_class, "mask_token_id", None) is not None:
            self.mask_token_id = processing_class.mask_token_id
        else:
            raise ValueError(
                "Could not infer mask_token_id from tokenizer; pass mask_token_id= explicitly. "
                "It is required to noise the on-policy completion before the loss forward."
            )

        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forwards."
            )

        if self.fixed_teacher:
            print("\n[OPSD-DLLM] FIXED TEACHER MODE — teacher = base model w/o LoRA adapters.\n")

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # Keep whichever dataset columns the collator reads from.
        collator = self.data_collator
        keep = []
        for attr in ("instruction_field", "response_field", "context_field"):
            val = getattr(collator, attr, None)
            if val:
                keep.append(val)
        for column in keep:
            if self._signature_columns is None:
                self._signature_columns = []
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    @staticmethod
    def _shift_logits_dream(logits: torch.Tensor) -> torch.Tensor:
        """Dream-7B convention: logits[:, i] should predict the token at position i.

        Concretely, shift the model's raw output right by one position (with the
        position-0 logits duplicated, since BOS is always unmasked and that slot
        is unused downstream).
        """
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    def _forward(self, model, input_ids, attention_mask):
        """One forward pass returning shift-aligned logits.

        Dream's modeling_dream.py passes attention_mask straight to SDPA without
        `_prepare_4d_attention_mask`, so we expand the (B, L) mask to (B, 1, 1, L)
        bool here. Without this, SDPA receives the wrong shape and either errors
        or silently treats every position as visible.
        """
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].bool()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return self._shift_logits_dream(outputs.logits)

    def _build_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """Mark valid positions in a generated completion (1=keep, 0=ignore for loss).

        A completion-span position is invalid if (a) it is a pad token, or (b) it
        sits *strictly after* the first EOS in the same sequence. The EOS itself
        is kept so the model is rewarded for terminating at the right step.
        """
        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id
        B, _ = completion_ids.shape
        mask = torch.ones_like(completion_ids)
        if eos_id is not None:
            for i in range(B):
                eos_positions = (completion_ids[i] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first = int(eos_positions[0].item())
                    mask[i, first + 1:] = 0
        if pad_id is not None:
            mask[completion_ids == pad_id] = 0
        return mask

    def _generate_student_completion(self, model, prompt_ids, prompt_mask):
        """On-policy student rollout via Dream's diffusion_generate.

        Generation is non-differentiable (it samples discrete tokens), so this
        runs under torch.no_grad. The wrapped model is unwrapped via
        `unwrap_model_for_generation` so DeepSpeed/FSDP-sharded weights are
        gathered for the rollout, then re-sharded on context exit.

        Returns:
            completion_ids:  [B, gen_max_new_tokens] generated tokens
            completion_mask: [B, gen_max_new_tokens] 1 = real generated token,
                             0 = pad / strictly-post-EOS (ignored by JSD)
        """
        prompt_len = prompt_ids.shape[1]
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            with torch.no_grad():
                gen_out = unwrapped.diffusion_generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.gen_max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=self.gen_steps,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    alg=self.gen_alg,
                    alg_temp=self.gen_alg_temp,
                )
        completion_ids = gen_out.sequences[:, prompt_len:].contiguous()
        completion_mask = self._build_completion_mask(completion_ids)
        return completion_ids, completion_mask

    def _sample_mask(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Antithetic per-example mask rate, restricted to *valid* positions.

        Args:
            valid: [B, L] bool/int — 1 where the position is allowed to be masked
                   (real generated token, i.e. completion_mask).

        Returns:
            mask:           [B, L] bool — True at randomly masked positions
            p_mask_sample:  [B] float  — the per-example mask rate (diagnostics)
        """
        device = valid.device
        valid_b = valid.bool()
        B, L = valid_b.shape

        u0 = torch.rand(1, device=device, dtype=torch.float32)
        idx = torch.arange(B, device=device, dtype=torch.float32)
        t = (u0 + idx / B) % 1
        p_mask_sample = (1 - self.sampling_eps) * t + self.sampling_eps  # [B]
        p_mask_grid = p_mask_sample[:, None].expand(B, L)
        rand = torch.rand((B, L), device=device)
        mask = (rand < p_mask_grid) & valid_b

        # Guarantee at least one masked position per example with valid tokens —
        # otherwise that row contributes zero JSD terms.
        for i in range(B):
            if valid_b[i].any() and not mask[i].any():
                valid_idx = valid_b[i].nonzero(as_tuple=False).flatten()
                j = int(torch.randint(0, valid_idx.numel(), (1,), device=device).item())
                mask[i, int(valid_idx[j].item())] = True
        return mask, p_mask_sample

    @staticmethod
    def generalized_jsd_loss(
        student_logits: torch.Tensor,  # [N, V] — already gathered at masked positions
        teacher_logits: torch.Tensor,  # [N, V] — already gathered at masked positions
        beta: float = 0.5,
        temperature: float = 1.0,
        top_k: int | None = None,
        token_clip: float | None = None,
        per_token_weight: torch.Tensor | None = None,  # [N]
        reduction: str = "batchmean",
    ):
        """Token-level JSD between student and teacher distributions.

        Inputs are already gathered at masked positions (N = total masked tokens
        across the batch). For each masked token i:
            jsd_i = D_KL(P_teacher^i || M^i)*beta + D_KL(P_student^i || M^i)*(1-beta)
        where M is the (beta-mixed) distribution and each KL is summed over the
        vocabulary. The function returns the mean of jsd_i over the N tokens
        (or sum / raw vector, depending on `reduction`).

        `token_clip` clips each *per-token* JSD value (after vocab summation), so
        a few high-divergence tokens cannot dominate the gradient.
        """
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        if top_k is not None and top_k > 0:
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            # forward KL: D(P_teacher || P_student), per-element then sum over vocab.
            per_token_jsd = F.kl_div(
                student_log_probs, teacher_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)  # [N]
        elif beta == 1:
            # reverse KL: D(P_student || P_teacher).
            per_token_jsd = F.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)  # [N]
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_t = F.kl_div(
                mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)  # [N]
            kl_s = F.kl_div(
                mixture_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)  # [N]
            per_token_jsd = beta * kl_t + (1 - beta) * kl_s  # [N]

        # Per-token clip (operates on the already-summed-over-vocab token-level JSD).
        if token_clip is not None:
            per_token_jsd = per_token_jsd.clamp(max=token_clip)

        if per_token_weight is not None:
            per_token_jsd = per_token_jsd * per_token_weight

        n_tokens = max(1, per_token_jsd.shape[0])
        if reduction in ("batchmean", "mean"):
            return per_token_jsd.sum() / n_tokens
        elif reduction == "sum":
            return per_token_jsd.sum()
        else:
            return per_token_jsd  # raw [N]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_ids = inputs["student_prompt_input_ids"]
        student_prompt_mask = inputs["student_prompt_attention_mask"]
        teacher_prompt_ids = inputs["teacher_prompt_input_ids"]
        teacher_prompt_mask = inputs["teacher_prompt_attention_mask"]

        s_prompt_len = student_prompt_ids.shape[1]
        t_prompt_len = teacher_prompt_ids.shape[1]

        # 1. Student rollout via diffusion_generate (no grad — sampling is discrete).
        completion_ids, completion_mask = self._generate_student_completion(
            model, student_prompt_ids, student_prompt_mask
        )
        _, L_c = completion_ids.shape

        # 2. Sample a random mask pattern over the valid (real) completion positions
        #    and build the noisy completion that the model will actually see.
        mask_pattern, p_mask_sample = self._sample_mask(completion_mask)
        noisy_completion = torch.where(
            mask_pattern,
            torch.full_like(completion_ids, self.mask_token_id),
            completion_ids,
        )

        # 3. Concat [prompt | noisy_completion] for student & teacher. The attention mask
        #    over the completion span is `completion_mask` (1 for real or masked, 0 for
        #    pad / post-EOS).
        student_full_ids = torch.cat([student_prompt_ids, noisy_completion], dim=1)
        student_full_mask = torch.cat([student_prompt_mask, completion_mask], dim=1)
        teacher_full_ids = torch.cat([teacher_prompt_ids, noisy_completion], dim=1)
        teacher_full_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)

        # 4. Student forward (with grad). Slice out the completion span only.
        student_logits = self._forward(model, student_full_ids, student_full_mask)
        student_completion_logits = student_logits[:, s_prompt_len : s_prompt_len + L_c, :]
        del student_logits

        # 5. Teacher forward (no grad). For fixed_teacher, run the base model w/o LoRA.
        if self.fixed_teacher and is_peft_model(model):
            teacher_ctx = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            teacher_ctx = nullcontext()

        with torch.no_grad(), teacher_ctx:
            teacher_logits = self._forward(model, teacher_full_ids, teacher_full_mask)
            teacher_completion_logits = (
                teacher_logits[:, t_prompt_len : t_prompt_len + L_c, :].float().detach()
            )
        del teacher_logits
        empty_cache()

        # 6. Token-level JSD over MASKED completion positions only.
        student_masked = student_completion_logits[mask_pattern]  # [N, V]
        teacher_masked = teacher_completion_logits[mask_pattern]  # [N, V]

        per_token_jsd = self.generalized_jsd_loss(
            student_masked.float(),
            teacher_masked,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
            reduction="none",
        )  # [N]

        loss = per_token_jsd.mean() if per_token_jsd.numel() > 0 else per_token_jsd.sum()

        # 7. Per-step diagnostics.
        with torch.no_grad():
            mode = "train" if model.training else "eval"
            comp_lens = completion_mask.sum(dim=1).float()
            self._metrics[mode]["completion_len_mean"].append(float(comp_lens.mean().item()))
            self._metrics[mode]["mean_p_mask"].append(float(p_mask_sample.mean().item()))
            self._metrics[mode]["num_masked_tokens"].append(float(mask_pattern.sum().item()))
            self._metrics[mode]["frac_masked_in_completion"].append(
                float(mask_pattern.sum().item()) / max(1.0, float(completion_mask.sum().item()))
            )
            if per_token_jsd.numel() > 0:
                self._metrics[mode]["per_token_jsd_mean"].append(float(per_token_jsd.mean().item()))
                self._metrics[mode]["per_token_jsd_max"].append(float(per_token_jsd.max().item()))

        if return_outputs:
            class _Out:
                pass

            o = _Out()
            o.loss = loss
            return loss, o
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if len(val) > 0}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
