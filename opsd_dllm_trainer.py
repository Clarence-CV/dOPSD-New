# Copyright 2026 The Foundation AI Team. Licensed under the Apache License, Version 2.0.
"""On-Policy Self-Distillation (OPSD) trainer for diffusion LLMs (e.g. Dream-7B).

Same model plays both roles: the *teacher* receives the ground-truth solution
in its prompt as privileged context, while the *student* receives only the
problem. We mask N random tokens of the answer (LLaDA / Dream-style antithetic
sampling of t ~ U(eps, 1)) and ask both to predict the masked tokens, then
distill student → teacher via JSD on the masked-position distributions.
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
        sampling_eps: float = 1e-3,
        mask_token_id: int | None = None,
        max_prompt_length: int = 1024,
        max_answer_length: int = 1024,
        weight_loss_by_p_mask: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
    ):
        if data_collator is None:
            data_collator = SelfDistillationDLLMDataCollator(
                tokenizer=processing_class,
                max_prompt_length=max_prompt_length,
                max_answer_length=max_answer_length,
            )

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
        self.sampling_eps = sampling_eps
        self.weight_loss_by_p_mask = weight_loss_by_p_mask
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip

        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif getattr(processing_class, "mask_token_id", None) is not None:
            self.mask_token_id = processing_class.mask_token_id
        else:
            raise ValueError(
                "Could not infer mask_token_id from tokenizer; pass mask_token_id= explicitly."
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
        """One forward pass returning shift-aligned logits."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return self._shift_logits_dream(outputs.logits)

    def _sample_mask(self, batch_size: int, max_answer_len: int, answer_lengths: torch.Tensor, device):
        """Antithetic-sampled mask pattern in answer-relative coordinates.

        Returns:
            mask:           bool tensor [B, max_answer_len], True at masked positions
            p_mask_sample:  float tensor [B], the per-example mask rate (used for the
                            optional 1/p_mask reweighting of the loss).
        """
        u0 = torch.rand(1, device=device, dtype=torch.float32)
        idx = torch.arange(batch_size, device=device, dtype=torch.float32)
        t = (u0 + idx / batch_size) % 1
        p_mask_sample = (1 - self.sampling_eps) * t + self.sampling_eps  # [B]
        p_mask_grid = p_mask_sample[:, None].expand(batch_size, max_answer_len)

        rand = torch.rand((batch_size, max_answer_len), device=device)
        random_mask = rand < p_mask_grid

        positions = torch.arange(max_answer_len, device=device)[None, :].expand(batch_size, -1)
        valid = positions < answer_lengths[:, None]
        mask = random_mask & valid

        # Guarantee at least one masked position per example so the loss is non-degenerate.
        for i in range(batch_size):
            ai = int(answer_lengths[i].item())
            if ai > 0 and not bool(mask[i].any()):
                pos = int(torch.randint(0, ai, (1,), device=device).item())
                mask[i, pos] = True

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
        device = self.accelerator.device

        student_prompt_ids = inputs["student_prompt_input_ids"]
        student_prompt_mask = inputs["student_prompt_attention_mask"]
        teacher_prompt_ids = inputs["teacher_prompt_input_ids"]
        teacher_prompt_mask = inputs["teacher_prompt_attention_mask"]
        answer_ids = inputs["answer_input_ids"]
        answer_mask = inputs["answer_attention_mask"]
        answer_lengths = inputs["answer_lengths"]

        B, A = answer_ids.shape

        # 1. Sample mask pattern in answer-relative coordinates (same for student & teacher).
        mask_pattern, p_mask_sample = self._sample_mask(B, A, answer_lengths, device)

        # 2. Build noisy answer (mask token at masked positions; pad positions stay pad).
        noisy_answer = torch.where(
            mask_pattern, torch.full_like(answer_ids, self.mask_token_id), answer_ids
        )

        # 3. Concat [prompt | noisy_answer]. Left-padded prompts + right-padded answer means
        #    pad tokens only appear on the outer edges.
        student_input_ids = torch.cat([student_prompt_ids, noisy_answer], dim=1)
        student_attn_mask = torch.cat([student_prompt_mask, answer_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, noisy_answer], dim=1)
        teacher_attn_mask = torch.cat([teacher_prompt_mask, answer_mask], dim=1)

        s_prompt_len = student_prompt_ids.shape[1]
        t_prompt_len = teacher_prompt_ids.shape[1]

        # 4. Student forward (with grad).
        student_logits = self._forward(model, student_input_ids, student_attn_mask)
        student_answer_logits = student_logits[:, s_prompt_len : s_prompt_len + A, :]  # [B, A, V]

        # 5. Teacher forward (no grad). For fixed_teacher, run the base model w/o LoRA.
        if self.fixed_teacher and is_peft_model(model):
            teacher_ctx = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            teacher_ctx = nullcontext()

        with torch.no_grad(), teacher_ctx:
            teacher_logits = self._forward(model, teacher_input_ids, teacher_attn_mask)
            teacher_answer_logits = teacher_logits[:, t_prompt_len : t_prompt_len + A, :].float().detach()
        del teacher_logits
        empty_cache()

        # 6. Token-level KL/JSD: compute ONLY at masked positions of the answer span.
        #    `mask_pattern` is [B, A] bool — selecting it from the [B, A, V] answer logits
        #    yields [N, V] where N = total masked tokens in the batch (one row per masked token).
        student_masked = student_answer_logits[mask_pattern]  # [N, V]
        teacher_masked = teacher_answer_logits[mask_pattern]  # [N, V]

        per_token_weight = None
        if self.weight_loss_by_p_mask:
            # LLaDA-style 1/p_mask reweighting (broadcast per-sample p_mask to its masked tokens).
            p_mask_grid = p_mask_sample[:, None].expand(B, A)
            per_token_weight = (1.0 / p_mask_grid)[mask_pattern].detach().to(student_masked.dtype)

        # Per-token JSD (each entry = JSD between student and teacher at one masked token).
        per_token_jsd = self.generalized_jsd_loss(
            student_masked.float(),
            teacher_masked,
            beta=self.beta,
            temperature=self.temperature,
            top_k=self.top_k_loss,
            token_clip=self.jsd_token_clip,
            per_token_weight=per_token_weight,
            reduction="none",  # keep token-level vector for diagnostics
        )  # [N]

        loss = per_token_jsd.mean() if per_token_jsd.numel() > 0 else per_token_jsd.sum()

        # 7. Per-step diagnostics (token-level KL stats over the masked tokens).
        with torch.no_grad():
            mode = "train" if model.training else "eval"
            self._metrics[mode]["mean_p_mask"].append(float(p_mask_sample.mean().item()))
            self._metrics[mode]["frac_masked_in_answer"].append(
                float(mask_pattern.sum().item()) / max(1.0, float(answer_lengths.sum().item()))
            )
            self._metrics[mode]["num_masked_tokens"].append(float(mask_pattern.sum().item()))
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
