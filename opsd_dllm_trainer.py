# Copyright 2026 The Foundation AI Team. Licensed under the Apache License, Version 2.0.
"""On-Policy Self-Distillation (OPSD) trainer for diffusion LLMs.

One model plays both roles: the teacher gets the ground-truth solution in its
prompt as privileged context; the student gets only the problem. The student
rolls out a completion, a random mask is sampled over valid completion
positions, both prompts are concatenated with the noisy (masked) completion
and forwarded, and JSD is computed only at the masked positions.

Two backends via `student_backend`:
  * "dream"  — Dream-7B (bidirectional). Shifted logit convention
               (`raw[:, i-1]` predicts token `i`); needs `_shift_logits_dream`
               and a 4-D bool attention mask. Rollouts via `diffusion_generate`.
  * "llada"  — GSAI LLaDA (mask-diffusion CausalLM). Standard `logits[:, i]`
               semantics (no shift); 2-D mask passed through. Rollouts via the
               built-in `_llada_generate` sampler. Mirrors tabom_test_code.py.
"""

import inspect
import os
import textwrap
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


# Dream-7B diffusion-sampler dtype patches: Dream's released _sample /
# sample_tokens mix bf16 logits with fp32 confidence, which silently misranks
# the entropy-based unmask ordering and collapses rollouts into repetition
# ("the the the ..."). Same fix as the eval script; required so the on-policy
# rollout (and thus the distilled LoRA) isn't trained on garbage.
def _iter_diffusion_model_objects(model):
    """Yield wrapper/base-model objects that may own Dream generation methods."""
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        stack.extend(
            getattr(obj, attr, None)
            for attr in ("base_model", "model")
            if getattr(obj, attr, None) is not obj
        )


def _patch_diffusion_sample_tokens_dtype(model) -> bool:
    """Keep Dream's sample_tokens confidence dtype aligned with logits."""
    patched = False
    for obj in _iter_diffusion_model_objects(model):
        for method_name in ("diffusion_generate", "_sample"):
            method = getattr(obj, method_name, None)
            method_fn = getattr(method, "__func__", method)
            generation_globals = getattr(method_fn, "__globals__", None)
            if not generation_globals or "sample_tokens" not in generation_globals:
                continue

            sample_tokens = generation_globals["sample_tokens"]
            if getattr(sample_tokens, "_opsd_confidence_dtype_patch", False):
                patched = True
                continue

            def sample_tokens_dtype_safe(logits, *args, _sample_tokens=sample_tokens, **kwargs):
                confidence, x0 = _sample_tokens(logits, *args, **kwargs)
                if torch.is_tensor(confidence) and confidence.dtype != logits.dtype:
                    confidence = confidence.to(dtype=logits.dtype)
                return confidence, x0

            sample_tokens_dtype_safe._opsd_confidence_dtype_patch = True
            sample_tokens_dtype_safe._opsd_original_sample_tokens = sample_tokens
            generation_globals["sample_tokens"] = sample_tokens_dtype_safe
            patched = True
    return patched


def _patch_diffusion_sample_assignment_dtype(model) -> bool:
    """Patch Dream's _sample assignment that can mix bf16 destination and fp32 source."""
    old = "full_confidence[mask_index] = confidence"
    new = "full_confidence[mask_index] = confidence.to(dtype=full_confidence.dtype)"

    for obj in _iter_diffusion_model_objects(model):
        sample_method = getattr(obj, "_sample", None)
        sample_fn = getattr(sample_method, "__func__", sample_method)
        if not callable(sample_fn):
            continue
        if getattr(sample_fn, "_opsd_full_confidence_dtype_patch", False):
            return True

        try:
            source = textwrap.dedent(inspect.getsource(sample_fn))
        except (OSError, TypeError):
            continue

        if old not in source:
            continue

        patched_source = source.replace(old, new)
        generation_globals = sample_fn.__globals__
        exec(
            compile(patched_source, inspect.getsourcefile(sample_fn) or "<opsd_dream_patch>", "exec"),
            generation_globals,
        )
        patched_fn = generation_globals[sample_fn.__name__]
        patched_fn._opsd_full_confidence_dtype_patch = True
        patched_fn._opsd_original_sample = sample_fn

        owner = getattr(sample_method, "__self__", obj)
        setattr(owner.__class__, sample_fn.__name__, patched_fn)
        return True

    return False


def patch_diffusion_generation_dtype(model) -> bool:
    """Apply both Dream diffusion-sampler dtype patches. Idempotent."""
    _patch_diffusion_sample_tokens_dtype(model)
    return _patch_diffusion_sample_assignment_dtype(model)


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
        # off_policy: distill on the dataset ground-truth answer, no rollout.
        off_policy: bool = False,
        # student_backend: "dream" or "llada" — picks logit semantics + sampler.
        student_backend: str = "dream",
        # mask_schedule over JSD-valid positions (mirrors tabom_test_code.py,
        # minus "td"): "diffusion" = antithetic per-example rate + i.i.d.
        # Bernoulli; "fixed" = exact-count k=round(n_valid*fixed_mask_ratio),
        # where fixed_mask_ratio is a float or a "lo:hi" range sampled per batch.
        mask_schedule: str = "diffusion",
        fixed_mask_ratio: str = "0.75",
        diffusion_min_t: float = 0.0,
        diffusion_max_t: float = 1.0,
        # Student-rollout generation hyperparameters. gen_alg / gen_alg_temp are
        # Dream-only (ignored by the LLaDA sampler).
        gen_max_new_tokens: int = 256,
        gen_steps: int = 256,
        gen_temperature: float = 0.2,
        gen_top_p: float = 0.95,
        gen_alg: str = "entropy",
        gen_alg_temp: float = 0.0,
    ):
        if student_backend not in ("dream", "llada"):
            raise ValueError(
                f"student_backend must be 'dream' or 'llada', got {student_backend!r}"
            )
        if mask_schedule not in ("diffusion", "fixed"):
            raise ValueError(
                f"mask_schedule must be 'diffusion' or 'fixed', got {mask_schedule!r}. "
                "('td' from tabom_test_code.py is intentionally not supported yet — "
                "it requires per-example decoding_order trajectories from the collator.)"
            )
        if data_collator is None:
            data_collator = SelfDistillationDLLMDataCollator(
                tokenizer=processing_class,
                max_prompt_length=max_prompt_length,
                max_answer_length=max_answer_length,
            )

        # Collator tokenizes problem/solution at collate time; bypass
        # SFTTrainer's pre-tokenization (expects a "text" column, would crash).
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

        self.student_backend = student_backend

        # Apply the Dream dtype patch once on the unwrapped model (see above).
        # LLaDA uses its own sampler and doesn't need it.
        if self.student_backend == "dream":
            unwrapped_for_patch = self.accelerator.unwrap_model(self.model)
            if patch_diffusion_generation_dtype(unwrapped_for_patch):
                print("[OPSD-DLLM] Patched Dream diffusion generation dtype handling.")
            else:
                print(
                    "[OPSD-DLLM] Warning: could not patch Dream generation internals; "
                    "on-policy rollouts may collapse to repetitive tokens."
                )
        else:
            print(f"[OPSD-DLLM] student_backend={self.student_backend!r}; skipping Dream dtype patches.")

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(self.model)

        self.beta = args.beta
        self.temperature = getattr(args, "temperature", 1.0) or 1.0
        self.fixed_teacher = fixed_teacher
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.sampling_eps = sampling_eps
        self.off_policy = off_policy
        self.mask_schedule = mask_schedule
        self.fixed_mask_ratio = fixed_mask_ratio
        self.diffusion_min_t = float(diffusion_min_t)
        self.diffusion_max_t = float(diffusion_max_t)

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

        if self.off_policy:
            print(
                "\n[OPSD-DLLM] OFF-POLICY MODE — distilling on the dataset's ground-truth "
                "answer (no student rollout). Student sees the masked GT; teacher sees "
                "the concrete GT.\n"
            )

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Buffer for periodic JSON dumps of (prompt, completion) pairs, flushed
        # on a cadence aligned to save_steps.
        self._generation_outputs_buffer: list[dict] = []
        self._generation_save_frequency = int(getattr(args, "save_steps", 50) or 50)

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # Keep whichever dataset columns the collator reads from.
        collator = self.data_collator
        keep = []
        for attr in ("problem_field", "solution_field"):
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
        """Shift raw Dream logits right by one so `out[:, i]` predicts token i.

        Position-0 logits are duplicated (BOS is always unmasked, that slot is
        unused downstream).
        """
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    def _forward(self, model, input_ids, attention_mask):
        """One forward pass, logits aligned so `out[:, i]` predicts token i.

        Dream: modeling_dream passes attention_mask straight to SDPA, so expand
        (B, L) -> (B, 1, 1, L) bool, then shift logits to canonical alignment.
        LLaDA: standard HF CausalLM — 2-D mask passed through, no shift.
        `nan_to_num` clamps the ±inf LLaDA can emit at low-prob vocab slots in bf16.
        """
        if self.student_backend == "llada":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            return torch.nan_to_num(
                outputs.logits.float(), nan=0.0, posinf=1e4, neginf=-1e4
            )

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].bool()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return self._shift_logits_dream(outputs.logits)

    def _build_completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """Mark valid positions in a Dream completion (1=keep, 0=ignore).

        Dream's window is fixed-length and bidirectional, so every position is
        an intentional output — no AR-style "everything after first EOS is
        invalid" logic (that would zero out content on mid-sequence EOS). Only
        an explicit pad token distinct from EOS is masked out.
        """
        mask = torch.ones_like(completion_ids)
        pad_id = self.processing_class.pad_token_id
        eos_id = self.processing_class.eos_token_id
        if pad_id is not None and pad_id != eos_id:
            mask[completion_ids == pad_id] = 0
        return mask

    def _build_jsd_valid_mask(
        self, completion_ids: torch.Tensor, completion_mask: torch.Tensor
    ) -> torch.Tensor:
        """Loss-eligible positions: completion_mask minus the trailing <eos> run.

        Dream's fixed-length window pads a finished answer with <eos> filler;
        scoring JSD there wastes budget/gradient on trivial "emit <eos>". Drops
        the trailing run from the loss only (attention mask is untouched). The
        first (terminator) <eos> is kept so the student learns when to stop; a
        mid-sequence <eos> followed by real content stays valid.
        """
        eos_id = self.processing_class.eos_token_id
        if eos_id is None:
            return completion_mask

        device = completion_ids.device
        B, L = completion_ids.shape
        positions = torch.arange(L, device=device)

        # content = real, non-<eos> token (completion_mask drops distinct pad).
        is_content = (completion_ids != eos_id) & completion_mask.bool()

        # Last content token per row (-1 if the row is all <eos>).
        content_pos = torch.where(
            is_content, positions.expand(B, L), torch.full((B, L), -1, device=device)
        )
        last_content_idx = content_pos.max(dim=1).values  # [B]

        # Keep through the terminator <eos> (one past last content); drop rest.
        keep_until = last_content_idx + 1  # [B]
        trailing_keep = positions.unsqueeze(0) <= keep_until.unsqueeze(1)  # [B, L]

        return completion_mask * trailing_keep.long()

    def _generate_student_completion(self, model, prompt_ids, prompt_mask):
        """On-policy student rollout (Dream: diffusion_generate; LLaDA: _llada_generate).

        Runs under no_grad (sampling is discrete) with the model unwrapped
        (gathers sharded weights) and in eval() mode (dropout off), restoring
        train mode after.

        Returns completion_ids and completion_mask, both [B, gen_max_new_tokens]
        (mask: 1=real token, 0=pad/post-EOS, ignored by JSD).
        """
        prompt_len = prompt_ids.shape[1]
        was_training = model.training
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            unwrapped.eval()
            try:
                with torch.no_grad():
                    if self.student_backend == "llada":
                        gen_out = self._llada_generate(unwrapped, prompt_ids, prompt_mask)
                    else:
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
            finally:
                if was_training:
                    unwrapped.train()
        completion_ids = gen_out.sequences[:, prompt_len:].contiguous()
        completion_mask = self._build_completion_mask(completion_ids)
        return completion_ids, completion_mask

    @torch.no_grad()
    def _llada_generate(self, model, prompt_ids, prompt_mask):
        """LLaDA-style progressive-unmasking sampler.

        Appends gen_max_new_tokens <mask> to the prompt and runs gen_steps
        denoising passes; each forwards [prompt | completion] (2-D mask path
        matching tabom_test_code.py), applies temperature + top-p, and unmasks
        the top ceil(remaining_masks / remaining_steps) positions by confidence.
        A final pass force-unmasks any residual <mask> so output is concrete.
        gen_alg / gen_alg_temp (Dream-only) are ignored here.

        Returns an object with `.sequences = [B, prompt_len + gen_max_new_tokens]`,
        matching diffusion_generate's return_dict_in_generate interface.
        """
        B, L_p = prompt_ids.shape
        L_c = self.gen_max_new_tokens
        device = prompt_ids.device

        completion = torch.full(
            (B, L_c), self.mask_token_id, dtype=prompt_ids.dtype, device=device
        )
        full_ids = torch.cat([prompt_ids, completion], dim=1)
        comp_attn = torch.ones((B, L_c), dtype=prompt_mask.dtype, device=device)
        full_mask = torch.cat([prompt_mask, comp_attn], dim=1)

        steps = max(1, self.gen_steps)
        temp = max(self.gen_temperature, 1e-8) if self.gen_temperature > 0 else 1.0
        top_p = self.gen_top_p

        for step in range(steps):
            is_mask = full_ids[:, L_p:] == self.mask_token_id  # [B, L_c]
            if not is_mask.any():
                break

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=False,
                )
            logits = torch.nan_to_num(
                out.logits[:, L_p:, :].float(), nan=0.0, posinf=1e4, neginf=-1e4
            )

            if self.gen_temperature > 0:
                logits = logits / temp

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cumprobs > top_p
                # Keep at least the top-1 token per position.
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                idx_remove = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
                logits = logits.masked_fill(idx_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            confidence, sampled = probs.max(dim=-1)  # both [B, L_c]

            # Only currently-masked positions are candidates this step.
            confidence = torch.where(
                is_mask, confidence, torch.full_like(confidence, -1.0)
            )

            remaining_steps = steps - step
            for b in range(B):
                n_mask_b = int(is_mask[b].sum().item())
                if n_mask_b == 0:
                    continue
                # Ceil-div: spread remaining unmasks over remaining steps.
                k = (n_mask_b + remaining_steps - 1) // remaining_steps
                k = min(k, n_mask_b)
                _, top_idx = confidence[b].topk(k)
                full_ids[b, L_p + top_idx] = sampled[b, top_idx]

        # Force-unmask any residual <mask> with argmax so the rollout is
        # concrete before the JSD pass.
        residual = full_ids[:, L_p:] == self.mask_token_id
        if residual.any():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=False,
                )
            logits = torch.nan_to_num(
                out.logits[:, L_p:, :].float(), nan=0.0, posinf=1e4, neginf=-1e4
            )
            sampled = logits.argmax(dim=-1)
            full_ids[:, L_p:] = torch.where(residual, sampled, full_ids[:, L_p:])

        class _GenOut:
            pass

        gen_out = _GenOut()
        gen_out.sequences = full_ids
        return gen_out

    def _sample_mask(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a mask over valid positions per `self.mask_schedule`.

        valid: [B, L] — 1 where maskable (JSD-eligible generated token).
        Returns mask [B, L] bool and p_mask_sample [B] (per-example rate, diag).
        Schedules mirror tabom_test_code.py; both enforce >=1 masked position
        per non-empty row so every row contributes a JSD term.
        """
        if self.mask_schedule == "fixed":
            mask, p_mask_sample = self._fixed_mask(valid)
        else:
            mask, p_mask_sample = self._diffusion_mask(valid)

        # Guarantee >=1 masked position per non-empty row (else zero JSD terms).
        valid_b = valid.bool()
        B = valid_b.shape[0]
        device = valid.device
        for i in range(B):
            if valid_b[i].any() and not mask[i].any():
                valid_idx = valid_b[i].nonzero(as_tuple=False).flatten()
                j = int(torch.randint(0, valid_idx.numel(), (1,), device=device).item())
                mask[i, int(valid_idx[j].item())] = True
        return mask, p_mask_sample

    def _diffusion_mask(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Antithetic per-example mask rate, i.i.d. Bernoulli per valid position.

        Variance-reduced tabom `diffusion` schedule: a single antithetic uniform
        `(u0 + i/B) mod 1` stratifies the B rates over [diffusion_min_t,
        diffusion_max_t]. `sampling_eps` lower-bounds the rate.
        """
        device = valid.device
        valid_b = valid.bool()
        B, L = valid_b.shape

        u0 = torch.rand(1, device=device, dtype=torch.float32)
        idx = torch.arange(B, device=device, dtype=torch.float32)
        t_base = (u0 + idx / B) % 1  # antithetic uniform in [0, 1)

        lower = max(self.diffusion_min_t, self.sampling_eps)
        upper = self.diffusion_max_t
        if upper <= lower:
            p_mask_sample = torch.full(
                (B,), float(lower), device=device, dtype=torch.float32
            )
        else:
            # Linear interp [0,1) -> [lower, upper).
            p_mask_sample = lower + (upper - lower) * t_base

        p_mask_grid = p_mask_sample[:, None].expand(B, L)
        rand = torch.rand((B, L), device=device)
        mask = (rand < p_mask_grid) & valid_b
        return mask, p_mask_sample

    def _fixed_mask(self, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact-count masking: k = round(n_valid * mask_ratio) per example.

        Mirrors tabom_test_code.py. One ratio is sampled per call (a "lo:hi"
        range draws uniformly, like tabom's mr_global) and applied across the
        batch; k positions per example are drawn without replacement.
        """
        device = valid.device
        valid_b = valid.bool()
        B, L = valid_b.shape

        mr = self._sample_fixed_mask_ratio()
        mask = torch.zeros_like(valid_b)
        for b in range(B):
            positions = valid_b[b].nonzero(as_tuple=False).flatten()
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
        """Parse fixed_mask_ratio ("0.75" or "0.25:0.75") into a scalar."""
        s = self.fixed_mask_ratio
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s)
        if ":" in s:
            lo, hi = (float(x) for x in s.split(":", 1))
            return lo + float(torch.rand(1).item()) * (hi - lo)
        return float(s)

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
        """Token-level JSD over N pre-gathered masked tokens.

        Per token: jsd = beta*D_KL(P_teacher||M) + (1-beta)*D_KL(P_student||M),
        M the beta-mixed distribution, each KL summed over vocab. Returns the
        mean / sum / raw [N] vector per `reduction`.

        `token_clip` caps each per-(token, vocab-entry) KL BEFORE the vocab sum
        (AR-trainer semantics): tames rare high-disagreement entries without
        flattening the per-token JSD (which would kill the bulk gradient).
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
            # forward KL D(P_teacher || P_student), per-element [N, V]
            per_element_jsd = F.kl_div(
                student_log_probs, teacher_log_probs, reduction="none", log_target=True
            )
        elif beta == 1:
            # reverse KL D(P_student || P_teacher), per-element [N, V]
            per_element_jsd = F.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            )
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_t = F.kl_div(
                mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
            )  # [N, V]
            kl_s = F.kl_div(
                mixture_log_probs, student_log_probs, reduction="none", log_target=True
            )  # [N, V]
            per_element_jsd = beta * kl_t + (1 - beta) * kl_s  # [N, V]

        # Per-(token, vocab-entry) clip before the vocab sum (opsd_trainer.py semantics).
        if token_clip is not None:
            per_element_jsd = per_element_jsd.clamp(max=token_clip)

        per_token_jsd = per_element_jsd.sum(dim=-1)  # [N]

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

        # 1. Completion both prompts are scored on: on-policy = student rollout
        #    (no grad, fully concrete); off-policy = dataset ground-truth answer.
        if self.off_policy:
            if "answer_input_ids" not in inputs:
                raise KeyError(
                    "off_policy=True requires the collator to provide `answer_input_ids` "
                    "and `answer_attention_mask` (the ground-truth answer span). "
                    "SelfDistillationDLLMDataCollator emits both."
                )
            completion_ids = inputs["answer_input_ids"]
            completion_mask = inputs["answer_attention_mask"]
        else:
            completion_ids, completion_mask = self._generate_student_completion(
                model, student_prompt_ids, student_prompt_mask
            )
        _, L_c = completion_ids.shape

        # 1b. Buffer (prompt, completion) for the periodic dump. Decode the
        #     completion with special tokens kept so oddities stay visible.
        if self.accelerator.is_main_process:
            prompt_texts = self.processing_class.batch_decode(
                student_prompt_ids, skip_special_tokens=True
            )
            completion_texts = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=False
            )
            step_now = int(self.state.global_step)
            for p_text, c_text in zip(prompt_texts, completion_texts):
                self._generation_outputs_buffer.append(
                    {"step": step_now, "prompt": p_text, "completion": c_text}
                )

        # 2. Sample a random mask over the loss-eligible positions and build the
        #    noisy completion both prompts will see.
        jsd_valid_mask = self._build_jsd_valid_mask(completion_ids, completion_mask)
        mask_pattern, p_mask_sample = self._sample_mask(jsd_valid_mask)
        noisy_completion = torch.where(
            mask_pattern,
            torch.full_like(completion_ids, self.mask_token_id),
            completion_ids,
        )

        # 3. Concat [prompt | completion]. Student always sees the noisy
        #    completion; teacher sees the noisy one (on-policy) or the concrete
        #    ground-truth (off-policy, a privileged soft-label target).
        #    completion_mask (1=real/masked, 0=pad) is shared by both.
        teacher_completion = completion_ids if self.off_policy else noisy_completion
        student_full_ids = torch.cat([student_prompt_ids, noisy_completion], dim=1)
        student_full_mask = torch.cat([student_prompt_mask, completion_mask], dim=1)
        teacher_full_ids = torch.cat([teacher_prompt_ids, teacher_completion], dim=1)
        teacher_full_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)

        # 4. Student forward (with grad); slice the completion span.
        student_logits = self._forward(model, student_full_ids, student_full_mask)
        student_completion_logits = student_logits[:, s_prompt_len : s_prompt_len + L_c, :]
        del student_logits

        # 5. Teacher forward (no grad); fixed_teacher = base model w/o LoRA.
        #    eval() keeps the target distribution deterministic (dropout off).
        if self.fixed_teacher and is_peft_model(model):
            teacher_ctx = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            teacher_ctx = nullcontext()

        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad(), teacher_ctx:
                teacher_logits = self._forward(model, teacher_full_ids, teacher_full_mask)
                teacher_completion_logits = (
                    teacher_logits[:, t_prompt_len : t_prompt_len + L_c, :].float().detach()
                )
        finally:
            if was_training:
                model.train()
        del teacher_logits

        # 6. Token-level JSD over masked positions only.
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

        # 7. Diagnostics.
        with torch.no_grad():
            mode = "train" if model.training else "eval"
            comp_lens = completion_mask.sum(dim=1).float()
            self._metrics[mode]["completion_len_mean"].append(float(comp_lens.mean().item()))
            self._metrics[mode]["mean_p_mask"].append(float(p_mask_sample.mean().item()))
            self._metrics[mode]["num_masked_tokens"].append(float(mask_pattern.sum().item()))
            self._metrics[mode]["num_jsd_valid_tokens"].append(float(jsd_valid_mask.sum().item()))
            self._metrics[mode]["frac_masked_in_completion"].append(
                float(mask_pattern.sum().item()) / max(1.0, float(jsd_valid_mask.sum().item()))
            )
            if per_token_jsd.numel() > 0:
                self._metrics[mode]["per_token_jsd_mean"].append(float(per_token_jsd.mean().item()))
                self._metrics[mode]["per_token_jsd_max"].append(float(per_token_jsd.max().item()))

        # 8. Periodic dump; only flush on synced gradients to avoid duplicate
        #    writes under gradient accumulation.
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and getattr(self.accelerator, "sync_gradients", True)
        ):
            self._save_generation_outputs(int(self.state.global_step))

        if return_outputs:
            class _Out:
                pass

            o = _Out()
            o.loss = loss
            return loss, o
        return loss

    def _save_generation_outputs(self, step: int):
        """Flush the (prompt, completion) buffer to output_dir/generations JSON."""
        if not self.accelerator.is_main_process:
            return
        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        output_file = generations_dir / f"generations_step_{step}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "num_samples": len(self._generation_outputs_buffer),
                    "generations": self._generation_outputs_buffer,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\n[OPSD-DLLM] Saved {len(self._generation_outputs_buffer)} generations -> {output_file}\n"
        )
        self._generation_outputs_buffer.clear()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if len(val) > 0}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
