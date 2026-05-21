# Copyright 2026 The Foundation AI Team. Licensed under the Apache License, Version 2.0.
"""AR-to-dLLM on-policy distillation trainer.

Teacher = autoregressive LM (Qwen3, frozen).  Student = diffusion LM (Dream-7B).

Unlike opsd_dllm_trainer (self-distillation: one model plays both roles), here
the teacher is a *separate* model from a *different* architecture family. The
distillation pipeline per training step:

  1. Teacher reasoning.  The AR teacher receives the problem + reference
     solution as privileged context and generates an *analysis* of that
     solution (standard left-to-right `model.generate`).

  2. Append reasoning.  [reasoning_prompt | reasoning | transition] becomes the
     teacher's privileged prompt (one assistant turn, reasoning then a raw-text
     bridge into "now solve it yourself").

  3. Student rollout.  The diffusion student rolls out a completion from the
     problem-only prompt via Dream's `diffusion_generate`. A random mask
     pattern is then sampled over the valid completion positions.

  4. JSD loss.
         student : forward [s_prompt | NOISY    completion]  (diffusion, grad)
         teacher : forward [t_prompt | CONCRETE completion]  (AR teacher-forcing, no grad)
     JSD is taken only at the masked completion positions, between the
     student's denoising distribution and the teacher's next-token
     distribution.

Why the teacher sees the CONCRETE (un-noised) completion: it is autoregressive
— its logit at position i is P(x_i | x_{<i}). Feeding it <mask> tokens it has
never seen would corrupt that conditioning. So the teacher is teacher-forced on
the real completion and supplies a per-position next-token target; the student
learns to match it at whichever positions were noised this step.

Vocabulary: Dream is built on the Qwen2.5 vocabulary, so its regular
(non-special) token ids coincide with the Qwen3 teacher's. The completion
produced by Dream is therefore fed to the teacher directly, and the JSD is
computed over the overlapping vocabulary (see `_align_vocab`).
"""

import torch

from transformers.generation.configuration_utils import GenerationConfig

from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache

from opsd_dllm_trainer import OPSDDLLMTrainer


class OPSDAR2DLLMTrainer(OPSDDLLMTrainer):
    """On-policy distillation from a frozen AR teacher into a diffusion student.

    Subclasses OPSDDLLMTrainer to reuse the Dream-specific machinery unchanged:
    the diffusion-sampler dtype patch, the on-policy rollout
    (`_generate_student_completion`), the antithetic mask sampler
    (`_sample_mask`), the shift-aligned Dream forward (`_forward`), the
    token-level JSD (`generalized_jsd_loss`), the metrics `log()` and the
    periodic generation dump. Only `__init__` (to attach the separate AR
    teacher) and `compute_loss` (to swap self-distillation for AR teacher
    forcing) are overridden.
    """

    _tag_names = ["trl", "opsd-ar2dllm"]
    _name = "OPSD-AR2DLLM"

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,  # student (Dream) tokenizer
        teacher_model=None,  # frozen AR teacher (Qwen3)
        teacher_processing_class=None,  # teacher (Qwen3) tokenizer
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        mask_token_id: int | None = None,
        sampling_eps: float = 1e-3,
        max_prompt_length: int = 1024,
        max_answer_length: int = 1024,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        distill_vocab_size: int | None = None,
        # Teacher reasoning phase.
        reason_first: bool = True,
        max_reasoning_length: int = 2048,
        teacher_gen_temperature: float = 0.7,
        teacher_gen_top_p: float = 0.95,
        teacher_gen_top_k: int = 20,
        # Student rollout (Dream's diffusion_generate).
        gen_max_new_tokens: int = 256,
        gen_steps: int = 256,
        gen_temperature: float = 0.2,
        gen_top_p: float = 0.95,
        gen_alg: str = "entropy",
        gen_alg_temp: float = 0.0,
    ):
        if teacher_model is None:
            raise ValueError("teacher_model is required: the frozen AR teacher (e.g. Qwen3).")
        if teacher_processing_class is None:
            raise ValueError("teacher_processing_class is required: the AR teacher's tokenizer.")
        if data_collator is None:
            raise ValueError(
                "OPSDAR2DLLMTrainer needs an AR2DLLMDataCollator (student and teacher use "
                "separate tokenizers); construct one and pass it as data_collator=."
            )

        # The student side (Dream rollout, masking, JSD, dtype patch, dataset
        # bypass) is set up entirely by the parent. fixed_teacher=False: the
        # teacher here is a separate model, not a LoRA-disabled view.
        super().__init__(
            model=model,
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
            fixed_teacher=False,
            mask_token_id=mask_token_id,
            sampling_eps=sampling_eps,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_answer_length,
            top_k_loss=top_k_loss,
            jsd_token_clip=jsd_token_clip,
            gen_max_new_tokens=gen_max_new_tokens,
            gen_steps=gen_steps,
            gen_temperature=gen_temperature,
            gen_top_p=gen_top_p,
            gen_alg=gen_alg,
            gen_alg_temp=gen_alg_temp,
        )

        self.teacher_processing_class = teacher_processing_class
        self.reason_first = reason_first
        self.distill_vocab_size = distill_vocab_size

        # Freeze the teacher and keep KV cache on — it is inference-only.
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        if hasattr(teacher_model, "config"):
            teacher_model.config.use_cache = True

        # Device-place / wrap the teacher for inference, mirroring TRL's GKDTrainer.
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True
            )

        do_sample = teacher_gen_temperature is not None and teacher_gen_temperature > 0
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            do_sample=do_sample,
            temperature=teacher_gen_temperature if do_sample else None,
            top_p=teacher_gen_top_p if do_sample else None,
            top_k=teacher_gen_top_k if (do_sample and teacher_gen_top_k and teacher_gen_top_k > 0) else None,
            pad_token_id=teacher_processing_class.pad_token_id,
            eos_token_id=teacher_processing_class.eos_token_id,
            use_cache=True,
        )

        print(
            f"\n[OPSD-AR2DLLM] Teacher (AR) = {teacher_model.__class__.__name__}; "
            f"Student (diffusion) = {self.model.__class__.__name__}.\n"
            f"[OPSD-AR2DLLM] reason_first={reason_first}, "
            f"max_reasoning_length={max_reasoning_length}, "
            f"distill_vocab_size={distill_vocab_size or 'min(student, teacher)'}.\n"
        )

    # ------------------------------------------------------------------
    # Teacher reasoning phase
    # ------------------------------------------------------------------
    @staticmethod
    def _real_tokens(ids_row, mask_row, pad_id=None):
        """Return the list of *real* (non-pad) token ids in a padded row."""
        if mask_row is not None:
            return ids_row[mask_row.bool()].tolist()
        ids = ids_row.tolist()
        return ids if pad_id is None else [t for t in ids if t != pad_id]

    @staticmethod
    def _trim_generated(row, eos_id, pad_id):
        """Generated row -> token ids up to (and excluding) the first EOS / pad."""
        out = []
        for t in row.tolist():
            if t == eos_id or t == pad_id:
                break
            out.append(t)
        return out

    @staticmethod
    def _left_pad_rows(rows, max_len, pad_id, device):
        ids, mask = [], []
        for r in rows:
            n = max_len - len(r)
            ids.append([pad_id] * n + list(r))
            mask.append([0] * n + [1] * len(r))
        return (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device),
        )

    def _build_teacher_prompt_with_reasoning(self, inputs):
        """Run the AR teacher's reasoning phase and assemble its privileged prompt.

        Steps 1-2 of the pipeline: the teacher generates an analysis of the
        reference solution, then [reasoning_prompt | reasoning | transition] is
        assembled into a single LEFT-padded block so the completion that
        follows starts at a uniform column for every example.

        Returns:
            teacher_prompt_ids:  [B, T] left-padded teacher prompt token ids
            teacher_prompt_mask: [B, T] attention mask
            reasoning_texts:     list[str] decoded teacher reasoning (diagnostics)
        """
        reasoning_prompt_ids = inputs["teacher_reasoning_prompt_input_ids"]
        reasoning_prompt_mask = inputs["teacher_reasoning_prompt_attention_mask"]
        transition_ids = inputs["teacher_transition_input_ids"]
        device = reasoning_prompt_ids.device
        B = reasoning_prompt_ids.shape[0]

        pad_id = self.teacher_processing_class.pad_token_id
        eos_id = self.teacher_processing_class.eos_token_id

        reasoning_completion = None
        reasoning_texts = []
        if self.reason_first:
            Lr = reasoning_prompt_ids.shape[1]
            with unwrap_model_for_generation(
                self.teacher_model, self.accelerator
            ) as unwrapped_teacher:
                with torch.no_grad():
                    gen = unwrapped_teacher.generate(
                        input_ids=reasoning_prompt_ids,
                        attention_mask=reasoning_prompt_mask,
                        generation_config=self.reasoning_generation_config,
                    )
            reasoning_completion = gen[:, Lr:]
            reasoning_texts = self.teacher_processing_class.batch_decode(
                reasoning_completion, skip_special_tokens=True
            )

        # Assemble per example: real reasoning-prompt tokens (drop left pad)
        # + generated reasoning (trimmed at the first EOS / pad) + transition.
        rows = []
        for i in range(B):
            row = self._real_tokens(reasoning_prompt_ids[i], reasoning_prompt_mask[i])
            if reasoning_completion is not None:
                row = row + self._trim_generated(reasoning_completion[i], eos_id, pad_id)
            row = row + self._real_tokens(transition_ids[i], None, pad_id)
            rows.append(row)

        max_t = max(len(r) for r in rows)
        teacher_prompt_ids, teacher_prompt_mask = self._left_pad_rows(
            rows, max_t, pad_id, device
        )
        return teacher_prompt_ids, teacher_prompt_mask, reasoning_texts

    # ------------------------------------------------------------------
    # Teacher forward (autoregressive teacher forcing)
    # ------------------------------------------------------------------
    def _teacher_forward_ar(
        self, teacher_prompt_ids, teacher_prompt_mask, completion_ids, completion_mask, L_c
    ):
        """AR teacher forward on [t_prompt | CONCRETE completion] (teacher forcing).

        The teacher is autoregressive: `logits[:, i]` predicts token `i+1`, so
        the distribution over completion-token `j` is read at absolute index
        `t_prompt_len + j - 1`.

        The teacher prompt is LEFT-padded, so we pass explicit `position_ids`
        derived from the attention mask. A plain arange would be wrong under
        left padding and would corrupt RoPE.

        `logits_to_keep` asks the model to materialise logits only for the
        completion span (+1 for the boundary token whose logit predicts the
        first completion token), avoiding a [B, full_len, V] tensor over the
        long teacher prompt. Falls back to a full forward if the teacher's
        `forward()` does not accept that kwarg.
        """
        t_prompt_len = teacher_prompt_ids.shape[1]
        input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            try:
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    logits_to_keep=L_c + 1,
                )
                # With logits_to_keep=L_c+1 the kept tensor covers absolute
                # positions [t_prompt_len-1 .. t_prompt_len+L_c-1]; the first
                # L_c of those are the per-completion-token distributions.
                teacher_completion_logits = outputs.logits[:, :L_c, :]
            except TypeError:
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                teacher_completion_logits = outputs.logits[
                    :, t_prompt_len - 1 : t_prompt_len - 1 + L_c, :
                ]
        return teacher_completion_logits.float().detach()

    def _align_vocab(self, student_logits, teacher_logits):
        """Restrict both logit tensors to a shared vocabulary slice for the JSD.

        Dream (Qwen2.5-based) and Qwen3 share the regular BPE token range, and
        the completion is composed of those tokens. We compare distributions
        over the first `V` columns, where `V` defaults to the smaller of the
        two vocab sizes (this also drops Dream's extra <mask> column, which is
        never a genuine prediction target). Pass `distill_vocab_size` to clamp
        further, e.g. to the regular-token range only.
        """
        v_s = student_logits.shape[-1]
        v_t = teacher_logits.shape[-1]
        if self.distill_vocab_size:
            v = min(self.distill_vocab_size, v_s, v_t)
        else:
            v = min(v_s, v_t)
        if v_s != v:
            student_logits = student_logits[..., :v]
        if v_t != v:
            teacher_logits = teacher_logits[..., :v]
        return student_logits, teacher_logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_ids = inputs["student_prompt_input_ids"]
        student_prompt_mask = inputs["student_prompt_attention_mask"]
        s_prompt_len = student_prompt_ids.shape[1]

        # 1. On-policy student rollout via Dream's diffusion_generate (no grad).
        #    With gen_steps >= gen_max_new_tokens the completion is fully
        #    concrete (no residual <mask>), so it is safe to feed to the AR
        #    teacher's embedding table.
        completion_ids, completion_mask = self._generate_student_completion(
            model, student_prompt_ids, student_prompt_mask
        )
        _, L_c = completion_ids.shape

        # 2-3. Teacher reasoning phase + privileged-prompt assembly.
        teacher_prompt_ids, teacher_prompt_mask, reasoning_texts = (
            self._build_teacher_prompt_with_reasoning(inputs)
        )
        t_prompt_len = teacher_prompt_ids.shape[1]
        empty_cache()  # release the teacher-generation KV cache before the forwards

        # 3b. Buffer (prompt, reasoning, completion) for the periodic JSON dump.
        if self.accelerator.is_main_process:
            prompt_texts = self.processing_class.batch_decode(
                student_prompt_ids, skip_special_tokens=True
            )
            completion_texts = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=False
            )
            step_now = int(self.state.global_step)
            for i, (p_text, c_text) in enumerate(zip(prompt_texts, completion_texts)):
                self._generation_outputs_buffer.append(
                    {
                        "step": step_now,
                        "prompt": p_text,
                        "teacher_reasoning": reasoning_texts[i] if i < len(reasoning_texts) else "",
                        "completion": c_text,
                    }
                )

        # 4. Restrict JSD to the loss-eligible positions (trailing <eos> padding
        #    excluded), sample a random mask there -> noisy completion.
        jsd_valid_mask = self._build_jsd_valid_mask(completion_ids, completion_mask)
        mask_pattern, p_mask_sample = self._sample_mask(jsd_valid_mask)
        noisy_completion = torch.where(
            mask_pattern,
            torch.full_like(completion_ids, self.mask_token_id),
            completion_ids,
        )

        # 5. Student (diffusion) forward on [s_prompt | NOISY completion], with grad.
        student_full_ids = torch.cat([student_prompt_ids, noisy_completion], dim=1)
        student_full_mask = torch.cat([student_prompt_mask, completion_mask], dim=1)
        student_logits = self._forward(model, student_full_ids, student_full_mask)
        student_completion_logits = student_logits[:, s_prompt_len : s_prompt_len + L_c, :]
        del student_logits

        # 6. Teacher (AR) forward on [t_prompt | CONCRETE completion], no grad.
        teacher_completion_logits = self._teacher_forward_ar(
            teacher_prompt_ids, teacher_prompt_mask, completion_ids, completion_mask, L_c
        )

        # 7. Restrict both sides to the shared vocabulary.
        student_completion_logits, teacher_completion_logits = self._align_vocab(
            student_completion_logits, teacher_completion_logits
        )

        # 8. Token-level JSD over MASKED completion positions only.
        student_masked = student_completion_logits[mask_pattern]  # [N, V]
        teacher_masked = teacher_completion_logits[mask_pattern]  # [N, V]
        del student_completion_logits, teacher_completion_logits

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

        # 9. Per-step diagnostics.
        with torch.no_grad():
            mode = "train" if model.training else "eval"
            comp_lens = completion_mask.sum(dim=1).float()
            self._metrics[mode]["completion_len_mean"].append(float(comp_lens.mean().item()))
            self._metrics[mode]["teacher_prompt_len"].append(float(t_prompt_len))
            self._metrics[mode]["mean_p_mask"].append(float(p_mask_sample.mean().item()))
            self._metrics[mode]["num_masked_tokens"].append(float(mask_pattern.sum().item()))
            self._metrics[mode]["num_jsd_valid_tokens"].append(float(jsd_valid_mask.sum().item()))
            self._metrics[mode]["frac_masked_in_completion"].append(
                float(mask_pattern.sum().item()) / max(1.0, float(jsd_valid_mask.sum().item()))
            )
            if per_token_jsd.numel() > 0:
                self._metrics[mode]["per_token_jsd_mean"].append(float(per_token_jsd.mean().item()))
                self._metrics[mode]["per_token_jsd_max"].append(float(per_token_jsd.max().item()))

        # 10. Periodic generation dump (only when gradients have synced).
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
