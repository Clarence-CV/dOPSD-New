# Copyright 2026 The Foundation AI Team. Licensed under the Apache License, Version 2.0.
"""Trajectory-OPSD trainer for diffusion LLMs.

A variant of `OPSDDLLMTrainer` (see opsd_dllm_trainer.py) whose privileged
information and noise source both come from the student's *own decoding
trajectory* instead of a synthetic mask over the finished completion.

Contrast with the base on-policy OPSD:
  * base on-policy : student rolls out a completion (concrete); a synthetic mask
                     (diffusion / fixed) is sampled over the valid positions;
                     teacher sees the same noisy completion.
  * trajectory     : student rolls out *with its denoising history*. The noisy
                     completion is a *real* intermediate denoising step — per
                     example, the least-masked step whose masked fraction over
                     the loss-eligible region still exceeds `traj_mask_threshold`
                     (default 0.5, i.e. ">50% masked"). The teacher's privileged
                     information is the *concrete final rollout* (the endpoint of
                     the trajectory): it conditions on the tokens the student will
                     eventually decode, while the student only sees the
                     partially-decoded step. JSD is computed at the positions
                     that are <mask> in that step.

Because Dream / LLaDA freeze tokens once they are unmasked, the intermediate
trajectory state equals `final_completion` with the still-undecoded positions
re-masked — so it slots directly into the base trainer's noisy-completion
machinery: `noisy_completion = where(mask_pattern, <mask>, final_completion)`.

    student_prompt ─► diffusion_generate(output_history=True) ─► history + final
                                                                      │
                          pick step k = least-masked with frac > thr  │
                                                                      ▼
                                          mask_pattern = (history[k] == <mask>)
                                                                      │
                       ┌──────────────────────────────────────────────┤
                       │ student sees history[k] (noisy)              │ teacher sees
                       ▼                                              ▼ final (concrete)
   [s_prompt | noisy_completion]                  [t_prompt | final_completion]
                       │                                              │
                  forward (grad)                               forward (no_grad)
                       │                                              │
                       └────────── JSD on MASKED positions ──────────┘

This trainer is on-policy only (`off_policy` must be False). Both backends are
supported: Dream via `diffusion_generate(output_history=True)`, LLaDA via a
history-recording variant of the progressive-unmasking sampler.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model

from trl.models.utils import unwrap_model_for_generation

from opsd_dllm_trainer import OPSDDLLMTrainer


class OPSDDLLMTrajectoryTrainer(OPSDDLLMTrainer):
    """OPSD trainer that masks from a real decoding step and gives the teacher
    the concrete final rollout as privileged information."""

    _tag_names = ["trl", "opsd-dllm", "opsd-dllm-trajectory"]
    _name = "OPSD-DLLM-Trajectory"

    def __init__(
        self,
        *args,
        traj_mask_threshold: float = 0.5,
        traj_step_select: str = "least",
        **kwargs,
    ):
        # The base trainer only recognizes mask_schedule in {"diffusion","fixed"};
        # this trainer never calls `_sample_mask`, so leave mask_schedule at its
        # base default and drive masking entirely from the trajectory instead.
        kwargs.pop("mask_schedule", None)
        super().__init__(*args, **kwargs)

        if traj_step_select not in ("least", "most", "random"):
            raise ValueError(
                f"traj_step_select must be 'least', 'most' or 'random', got {traj_step_select!r}."
            )
        if self.off_policy:
            raise ValueError(
                "OPSDDLLMTrajectoryTrainer is on-policy (it derives the mask from the "
                "student's own decoding trajectory) and is incompatible with off_policy=True."
            )

        self.traj_mask_threshold = float(traj_mask_threshold)
        self.traj_step_select = traj_step_select

        print(
            "\n[OPSD-DLLM] TRAJECTORY MODE — student noise is a real decoding step "
            f"(select={self.traj_step_select!r}, masked-fraction threshold "
            f">{self.traj_mask_threshold:.2f}); teacher sees the concrete final rollout "
            "as privileged information.\n"
        )

    # ------------------------------------------------------------------
    # Rollout with decoding history.
    # ------------------------------------------------------------------
    def _generate_student_completion_with_history(self, model, prompt_ids, prompt_mask):
        """On-policy rollout that also returns the per-step denoising history.

        Mirrors `OPSDDLLMTrainer._generate_student_completion` but requests the
        trajectory:
          * Dream  : `diffusion_generate(output_history=True)`.
          * LLaDA  : `_llada_generate_with_history`.

        Returns:
            completion_ids:  [B, gen_max_new_tokens] final (concrete) rollout.
            completion_mask: [B, gen_max_new_tokens] 1 = real token, 0 = pad.
            history:         list of [B, gen_max_new_tokens] completion-span
                             snapshots (one per denoising step; <mask> at the
                             not-yet-decoded positions).
        """
        prompt_len = prompt_ids.shape[1]
        was_training = model.training
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            unwrapped.eval()
            try:
                with torch.no_grad():
                    if self.student_backend == "llada":
                        gen_out = self._llada_generate_with_history(
                            unwrapped, prompt_ids, prompt_mask
                        )
                    else:
                        gen_out = unwrapped.diffusion_generate(
                            prompt_ids,
                            attention_mask=prompt_mask,
                            max_new_tokens=self.gen_max_new_tokens,
                            output_history=True,
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

        # Dream appends a full-sequence snapshot per denoising step; our LLaDA
        # variant mirrors that. Slice each to the completion span.
        raw_history = getattr(gen_out, "history", None) or []
        history = [h[:, prompt_len:].contiguous() for h in raw_history]
        return completion_ids, completion_mask, history

    @torch.no_grad()
    def _llada_generate_with_history(self, model, prompt_ids, prompt_mask):
        """LLaDA progressive-unmasking sampler that records per-step snapshots.

        Identical sampling dynamics to `OPSDDLLMTrainer._llada_generate`; the
        only addition is appending `full_ids.clone()` after each denoising step
        (and the final force-unmask) so the trajectory can be replayed.

        Returns an object with `.sequences = [B, prompt_len + gen_max_new_tokens]`
        and `.history` = list of those full-sequence snapshots.
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

        history = []
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
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                idx_remove = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
                logits = logits.masked_fill(idx_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            confidence, sampled = probs.max(dim=-1)  # both [B, L_c]

            confidence = torch.where(
                is_mask, confidence, torch.full_like(confidence, -1.0)
            )

            remaining_steps = steps - step
            for b in range(B):
                n_mask_b = int(is_mask[b].sum().item())
                if n_mask_b == 0:
                    continue
                k = (n_mask_b + remaining_steps - 1) // remaining_steps
                k = min(k, n_mask_b)
                _, top_idx = confidence[b].topk(k)
                full_ids[b, L_p + top_idx] = sampled[b, top_idx]

            history.append(full_ids.clone())

        # Safety net: force-unmask any residual <mask> with its argmax sample.
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
            history.append(full_ids.clone())

        class _GenOut:
            pass

        gen_out = _GenOut()
        gen_out.sequences = full_ids
        gen_out.history = history
        return gen_out

    # ------------------------------------------------------------------
    # Trajectory mask selection.
    # ------------------------------------------------------------------
    def _trajectory_mask(self, completion_ids, jsd_valid_mask, history):
        """Pick, per example, a decoding step and return its still-masked positions.

        For each history snapshot, compute the masked fraction over the
        JSD-valid region (`jsd_valid_mask`). An example's *eligible* steps are
        those whose masked fraction strictly exceeds `traj_mask_threshold`.
        Among the eligible steps, pick one according to `traj_step_select`:
            "least"  — smallest masked fraction still above threshold (closest
                       to the 0.5 boundary; the default).
            "most"   — largest masked fraction (noisiest).
            "random" — uniform among the eligible steps.
        If no step qualifies (e.g. a very short trajectory), fall back to the
        most-masked available step so the example still contributes a mask.

        Args:
            completion_ids:  [B, L] final rollout (used only for shape/device).
            jsd_valid_mask:  [B, L] 1 where a position is loss-eligible.
            history:         list of [B, L] completion-span snapshots.

        Returns:
            mask:           [B, L] bool — True at the chosen step's masked positions
                            (restricted to JSD-valid).
            p_mask_sample:  [B] float  — chosen step's masked fraction (diagnostics).
        """
        device = completion_ids.device
        B, L = completion_ids.shape
        valid = jsd_valid_mask.bool()
        valid_counts = valid.sum(dim=1).clamp(min=1).float()  # [B]
        thr = self.traj_mask_threshold

        if not history:
            # Degenerate fallback: no trajectory captured. Treat the whole valid
            # region as masked (still satisfies the ≥1-mask invariant below).
            mask = valid.clone()
            p_mask_sample = mask.float().sum(dim=1) / valid_counts
            return self._ensure_min_one_mask(mask, valid), p_mask_sample

        # [S, B, L] — masked-and-valid indicator for every snapshot.
        is_mask_per_step = torch.stack(
            [(h == self.mask_token_id) & valid for h in history], dim=0
        )
        masked_counts = is_mask_per_step.sum(dim=2).float()  # [S, B]
        frac = masked_counts / valid_counts.unsqueeze(0)      # [S, B]

        qualifies = frac > thr            # [S, B]
        any_q = qualifies.any(dim=0)      # [B]

        if self.traj_step_select == "most":
            score = frac.clone()
            score[~qualifies] = -1.0
            chosen = score.argmax(dim=0)  # [B]
        elif self.traj_step_select == "random":
            chosen = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                qs = qualifies[:, b].nonzero(as_tuple=False).flatten()
                if qs.numel() == 0:
                    chosen[b] = int(frac[:, b].argmax().item())
                else:
                    j = int(torch.randint(0, qs.numel(), (1,), device=device).item())
                    chosen[b] = int(qs[j].item())
        else:  # "least" (default): smallest fraction still above threshold.
            score = frac.clone()
            score[~qualifies] = float("inf")
            chosen = score.argmin(dim=0)  # [B]

        # Where nothing qualifies, fall back to the most-masked available step.
        fallback = frac.argmax(dim=0)
        chosen = torch.where(any_q, chosen, fallback)  # [B]

        chosen_exp = chosen.view(1, B, 1).expand(1, B, L)
        mask = torch.gather(is_mask_per_step, 0, chosen_exp).squeeze(0)  # [B, L]
        p_mask_sample = torch.gather(frac, 0, chosen.view(1, B)).squeeze(0)  # [B]

        return self._ensure_min_one_mask(mask, valid), p_mask_sample

    @staticmethod
    def _ensure_min_one_mask(mask, valid):
        """Guarantee ≥1 masked position per row that has any valid token."""
        B = valid.shape[0]
        device = valid.device
        for i in range(B):
            if valid[i].any() and not mask[i].any():
                valid_idx = valid[i].nonzero(as_tuple=False).flatten()
                j = int(torch.randint(0, valid_idx.numel(), (1,), device=device).item())
                mask[i, int(valid_idx[j].item())] = True
        return mask

    # ------------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_ids = inputs["student_prompt_input_ids"]
        student_prompt_mask = inputs["student_prompt_attention_mask"]
        teacher_prompt_ids = inputs["teacher_prompt_input_ids"]
        teacher_prompt_mask = inputs["teacher_prompt_attention_mask"]

        s_prompt_len = student_prompt_ids.shape[1]
        t_prompt_len = teacher_prompt_ids.shape[1]

        # 1. On-policy rollout WITH decoding history. `completion_ids` is the
        #    concrete endpoint of the trajectory; `history` are the per-step
        #    snapshots the mask is drawn from.
        completion_ids, completion_mask, history = self._generate_student_completion_with_history(
            model, student_prompt_ids, student_prompt_mask
        )
        _, L_c = completion_ids.shape

        # 1b. Buffer (prompt, completion) for the periodic JSON dump.
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

        # 2. Build the noisy completion from a real decoding step. The mask is
        #    the set of still-undecoded positions at the chosen step; because
        #    decoded tokens are frozen during sampling, re-masking those
        #    positions in `completion_ids` reproduces that step exactly over the
        #    JSD-valid region.
        jsd_valid_mask = self._build_jsd_valid_mask(completion_ids, completion_mask)
        mask_pattern, p_mask_sample = self._trajectory_mask(
            completion_ids, jsd_valid_mask, history
        )
        noisy_completion = torch.where(
            mask_pattern,
            torch.full_like(completion_ids, self.mask_token_id),
            completion_ids,
        )

        # 3. Concat [prompt | completion]. The student sees the NOISY (real
        #    decoding-step) completion; the teacher sees the CONCRETE final
        #    rollout — its privileged information is the trajectory's endpoint,
        #    so its distribution at the masked positions is conditioned on the
        #    tokens the student will eventually decode.
        teacher_completion = completion_ids
        student_full_ids = torch.cat([student_prompt_ids, noisy_completion], dim=1)
        student_full_mask = torch.cat([student_prompt_mask, completion_mask], dim=1)
        teacher_full_ids = torch.cat([teacher_prompt_ids, teacher_completion], dim=1)
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
            self._metrics[mode]["num_jsd_valid_tokens"].append(float(jsd_valid_mask.sum().item()))
            self._metrics[mode]["frac_masked_in_completion"].append(
                float(mask_pattern.sum().item()) / max(1.0, float(jsd_valid_mask.sum().item()))
            )
            self._metrics[mode]["traj_num_steps"].append(float(len(history)))
            if per_token_jsd.numel() > 0:
                self._metrics[mode]["per_token_jsd_mean"].append(float(per_token_jsd.mean().item()))
                self._metrics[mode]["per_token_jsd_max"].append(float(per_token_jsd.max().item()))

        # 8. Periodic generation dump (only when gradients synced this step).
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
