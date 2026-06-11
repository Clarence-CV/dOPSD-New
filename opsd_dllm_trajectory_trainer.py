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
                     example, the least-masked step `k` whose masked fraction
                     over the loss-eligible region still exceeds
                     `traj_mask_threshold` (default 0.5, i.e. ">50% masked").
                     The teacher's privileged view is controlled by
                     `traj_teacher_gap`:
                       * None (default) — the *concrete final rollout* (endpoint
                         of the trajectory). The teacher sees the answer token at
                         every scored position, so its target is near-degenerate
                         (a "copy" of the student's own rollout).
                       * n (int ≥ 0)    — the trajectory state `n` steps AFTER the
                         student's step, i.e. `history[k + n]` (clamped to the
                         final state). The teacher sees more *surrounding* context
                         than the student but the positions still masked at step
                         `k + n` remain genuinely predictive — an honest
                         "peek-ahead" privilege rather than seeing the answer.
                     JSD is computed at the positions that are <mask> at step `k`.

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
                       ▼                                              ▼ history[k+n] (or final)
   [s_prompt | noisy_completion]                  [t_prompt | teacher_completion]
                       │                                              │
                  forward (grad)                               forward (no_grad)
                       │                                              │
                       └────────── JSD on step-k MASKED positions ───┘

This trainer is on-policy only (`off_policy` must be False). Both backends are
supported: Dream via `diffusion_generate(output_history=True)`, LLaDA via a
history-recording variant of the progressive-unmasking sampler.
"""

import re
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model

from trl.models.utils import unwrap_model_for_generation

from opsd_dllm_trainer import OPSDDLLMTrainer

# Math-equivalence verifier (handles fractions, radicals, algebraic forms — ~40%
# of MixChain-Z-PRM12K / MATH answers are non-integer, so plain string/number
# matching is wrong). Optional: if math_verify is not installed we fall back to a
# normalized string compare and warn once at trainer init.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - depends on the training env
    _mv_parse = _mv_verify = None
    _HAS_MATH_VERIFY = False


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
        traj_teacher_gap: int | None = None,
        traj_teacher_view: str = "snapshot",
        verify_gold_pi: bool = True,
        gold_pi_max_solution_tokens: int | None = None,
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
        if traj_teacher_gap is not None:
            traj_teacher_gap = int(traj_teacher_gap)
            if traj_teacher_gap < 0:
                raise ValueError(
                    f"traj_teacher_gap must be None or a non-negative int, got {traj_teacher_gap}."
                )
        if traj_teacher_view not in ("snapshot", "all_future"):
            raise ValueError(
                f"traj_teacher_view must be 'snapshot' or 'all_future', got {traj_teacher_view!r}."
            )

        self.traj_mask_threshold = float(traj_mask_threshold)
        self.traj_step_select = traj_step_select
        self.traj_teacher_gap = traj_teacher_gap
        self.traj_teacher_view = traj_teacher_view
        self.verify_gold_pi = bool(verify_gold_pi)
        self.gold_pi_max_solution_tokens = (
            int(gold_pi_max_solution_tokens)
            if gold_pi_max_solution_tokens is not None
            else None
        )

        if self.traj_teacher_view == "all_future":
            teacher_view = (
                "the FULL remaining trajectory (steps k+1 → final): per scored position, "
                "the teacher's predictive distribution is averaged over the future steps where "
                "that position is still masked (one teacher forward per remaining step — EXPENSIVE)"
            )
        elif self.traj_teacher_gap is None:
            teacher_view = "the concrete final rollout (trajectory endpoint)"
        else:
            teacher_view = (
                f"the trajectory state {self.traj_teacher_gap} step(s) after the student's step "
                "(history[k+gap], clamped to the final state)"
            )
        print(
            "\n[OPSD-DLLM] TRAJECTORY MODE — student noise is a real decoding step "
            f"(select={self.traj_step_select!r}, masked-fraction threshold "
            f">{self.traj_mask_threshold:.2f}); teacher sees {teacher_view} as privileged information."
        )
        if self.verify_gold_pi:
            verifier = (
                "math_verify (symbolic/numeric equivalence)"
                if _HAS_MATH_VERIFY
                else "STRING-MATCH ONLY — math_verify NOT installed; ~40% of MATH answers "
                "are non-integer and will be mis-verified. `pip install math_verify` is "
                "strongly recommended"
            )
            print(
                "[OPSD-DLLM] VERIFY-GATED GOLD PI — the student's final rollout is "
                "verified against the gold solution's \\boxed{} answer. CORRECT rollouts keep "
                "the on-policy trajectory PI above; WRONG rollouts instead get the FULL gold "
                "solution as the teacher's privileged context (the teacher re-scores the "
                "student's own noisy rollout at the masked positions, conditioned on the gold "
                f"solution). Verifier: {verifier}.\n"
            )
        else:
            print("[OPSD-DLLM] verify_gold_pi=False — gold-solution fallback disabled.\n")

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
            chosen:         [B] long   — per-example index into `history` of the
                            chosen step (None if no history was captured). Used by
                            the teacher n-step-ahead view.
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
            return self._ensure_min_one_mask(mask, valid), p_mask_sample, None

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

        return self._ensure_min_one_mask(mask, valid), p_mask_sample, chosen

    def _teacher_completion_n_ahead(self, completion_ids, history, chosen, n):
        """Build the teacher's completion as the trajectory state `n` steps after
        each example's chosen (student) step.

        For example `b`, the teacher sees `history[min(chosen[b] + n, S - 1)]` —
        i.e. `n` denoising steps further along the student's own trajectory,
        clamped to the final captured state. Positions already decoded by that
        step are concrete (the teacher sees them); positions still masked stay
        <mask> (the teacher predicts them, conditioned only on the extra context
        it has gained — an honest peek-ahead privilege rather than seeing the
        answer at the scored position).

        Falls back to the concrete final rollout when no trajectory/`chosen` is
        available.

        Returns:
            teacher_completion: [B, L] long
            teacher_step:       [B] long — the (clamped) step index the teacher saw
        """
        B, L = completion_ids.shape
        device = completion_ids.device
        if not history or chosen is None:
            return completion_ids, torch.full((B,), -1, dtype=torch.long, device=device)

        states = torch.stack(history, dim=0)  # [S, B, L]
        S = states.shape[0]
        teacher_step = (chosen + int(n)).clamp(max=S - 1)  # [B]
        idx = teacher_step.view(1, B, 1).expand(1, B, L)
        teacher_completion = torch.gather(states, 0, idx).squeeze(0).contiguous()  # [B, L]
        return teacher_completion, teacher_step

    @staticmethod
    def _build_teacher_full(
        teacher_prompt_ids,
        teacher_prompt_mask,
        completion,
        completion_mask,
        transition_ids=None,
        transition_mask=None,
    ):
        """Assemble [teacher_prompt | completion | optional transition] (ids, mask)."""
        ids_parts = [teacher_prompt_ids, completion]
        mask_parts = [teacher_prompt_mask, completion_mask]
        if transition_ids is not None and transition_ids.shape[1] > 0:
            if transition_mask is None:
                transition_mask = torch.ones_like(transition_ids)
            ids_parts.append(transition_ids)
            mask_parts.append(transition_mask)
        return torch.cat(ids_parts, dim=1), torch.cat(mask_parts, dim=1)

    def _teacher_masked_all_future(
        self,
        model,
        teacher_prompt_ids,
        teacher_prompt_mask,
        completion_mask,
        completion_ids,
        history,
        chosen,
        mask_pattern,
        t_prompt_len,
        L_c,
        transition_ids=None,
        transition_mask=None,
    ):
        """Aggregate the teacher target over the full remaining trajectory.

        For each scored position (masked at the student's step `k`), average the
        teacher's predictive distribution over the FUTURE steps `t in (k, final]`
        where that position is STILL masked — so the steps where the teacher has
        already decoded the token (and would just copy it) do not contaminate the
        target. Positions never masked in that window (decoded immediately after
        `k`) fall back to the teacher distribution on the final concrete rollout.

        Costs one teacher forward per distinct remaining step — EXPENSIVE.

        Returns:
            teacher_masked: [N, V] teacher LOG-probs, aligned to the row-major
                (b, l) order of `mask_pattern` (matches
                `student_completion_logits[mask_pattern]`).
        """
        device = completion_ids.device
        scored = mask_pattern.nonzero(as_tuple=False)  # [N, 2] (b, l), row-major
        N = scored.shape[0]
        b_n, l_n = scored[:, 0], scored[:, 1]

        teacher_ctx = (
            self.accelerator.unwrap_model(model).disable_adapter()
            if (self.fixed_teacher and is_peft_model(model))
            else nullcontext()
        )

        def _probs_at_scored(completion):
            full_ids, full_mask = self._build_teacher_full(
                teacher_prompt_ids, teacher_prompt_mask, completion,
                completion_mask, transition_ids, transition_mask,
            )
            logits = self._forward(model, full_ids, full_mask)
            comp = logits[:, t_prompt_len : t_prompt_len + L_c, :]
            sel = comp[b_n, l_n, :].float()  # [N, V]
            return F.softmax(sel, dim=-1)

        sum_probs = None
        count = torch.zeros(N, device=device)

        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad(), teacher_ctx:
                if history and chosen is not None:
                    states = torch.stack(history, dim=0)  # [S, B, L]
                    S = states.shape[0]
                    masked_states = states == self.mask_token_id  # [S, B, L]
                    chosen_n = chosen[b_n]  # [N]
                    t_start = max(int(chosen.min().item()) + 1, 1)
                    for t in range(t_start, S):
                        # scored positions masked at step t and in the (k, final] window
                        contribute = masked_states[t, b_n, l_n] & (t > chosen_n)  # [N]
                        if not bool(contribute.any()):
                            continue
                        probs = _probs_at_scored(states[t])  # [N, V]
                        if sum_probs is None:
                            sum_probs = torch.zeros(N, probs.shape[-1], device=device)
                        sum_probs[contribute] += probs[contribute]
                        count[contribute] += 1.0
                        del probs

                # Endpoint fallback for positions with no masked future step.
                need_fb = count == 0
                if sum_probs is None or bool(need_fb.any()):
                    probs = _probs_at_scored(completion_ids)  # [N, V]
                    if sum_probs is None:
                        sum_probs = torch.zeros(N, probs.shape[-1], device=device)
                    sum_probs[need_fb] = probs[need_fb]
                    count[need_fb] = 1.0
                    del probs
        finally:
            if was_training:
                model.train()

        # Diagnostic: avg number of future steps averaged per scored position
        # (1.0 ⇒ everything fell back to the endpoint).
        mode = "train" if was_training else "eval"
        self._metrics[mode]["teacher_future_steps_avg"].append(float(count.mean().item()))

        avg = sum_probs / count.clamp(min=1.0).unsqueeze(-1)  # [N, V]
        return avg.clamp_min(1e-12).log().detach()

    # ------------------------------------------------------------------
    # Verify-gated gold-solution privileged information.
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_boxed_answer(text):
        """Return the content of the LAST ``\\boxed{...}`` in `text` (nested-brace
        aware, e.g. ``\\boxed{\\frac{1}{2}}``), or None if there is none.

        This is the canonical MATH / MixChain-Z-PRM12K answer format: the gold
        ``solution`` always ends in ``\\boxed{<target>}`` and a MATH-tuned student
        emits the same. We scan from the LAST box so a worked solution that shows
        intermediate boxes still resolves to the final answer.
        """
        if not text:
            return None
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None
        start = idx + len(r"\boxed{")
        depth, i = 1, start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        return text[start : i - 1].strip() if depth == 0 else None

    @staticmethod
    def _preprocess_for_parse(answer):
        """Normalize an answer string so math_verify parses it.

        * ``\\dfrac`` / ``\\tfrac`` → ``\\frac`` (display-style fractions math_verify
          does not always parse; ~3% of MixChain golds use them).
        * ratio notation ``a:b`` → ``\\frac{a}{b}`` (mirrors grpo_train.py).
        """
        if answer is None:
            return None
        answer = answer.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
        if m:
            return rf"\frac{{{m.group(1)}}}{{{m.group(2)}}}"
        return answer

    def _answers_match(self, gold, pred):
        """True iff `pred` is mathematically equivalent to `gold`.

        Uses math_verify (``parse`` + ``verify``) for symbolic/numeric
        equivalence — essential here because a large fraction of MATH answers are
        fractions / radicals / expressions where string equality fails (``1/2``
        vs ``\\frac{1}{2}`` vs ``0.5``). Both sides are normalized first. Falls
        back to a whitespace-stripped, case-insensitive string match (covers MCQ
        letters, the ~20% of answers math_verify cannot parse, and the
        no-math_verify environment).
        """
        if gold is None or pred is None:
            return False
        gold_n = self._preprocess_for_parse(gold)
        pred_n = self._preprocess_for_parse(pred)
        if _HAS_MATH_VERIFY:
            try:
                g = _mv_parse(gold_n)
                p = _mv_parse(pred_n)
                if g and p:
                    return bool(_mv_verify(g, p))
            except Exception:
                pass
        gn = re.sub(r"\s+", "", gold_n).lower()
        pn = re.sub(r"\s+", "", pred_n).lower()
        return bool(gn) and gn == pn

    def _extract_pred_answer(self, completion_text):
        """Extract the student's final answer from a free-form completion.

        Prefer an explicit ``\\boxed{...}``; otherwise hand the raw completion to
        math_verify's ``parse`` (which pulls the last math expression / number out
        of free text) by returning the text itself. As a last resort (no
        math_verify), keep the substring after a trailing answer marker so the
        string fallback in `_answers_match` has something tight to compare.
        """
        boxed = self._extract_boxed_answer(completion_text)
        if boxed is not None:
            return boxed
        if _HAS_MATH_VERIFY:
            return completion_text  # parse() extracts the trailing expression
        # No math_verify: salvage the tail after the last answer cue.
        m = re.findall(r"(?:####|answer(?:\s+is)?\s*[:=]?)\s*(.+)", completion_text, re.IGNORECASE)
        return (m[-1].strip() if m else completion_text).splitlines()[0] if completion_text else None

    def _verify_rollouts(self, completion_ids, answer_ids, answer_mask):
        """Per-example correctness of the student rollout vs. the gold solution.

        Gold answer = the ``\\boxed{...}`` of the reference solution (verified to
        equal the dataset's `target` field for MixChain-Z-PRM12K). Student answer
        = `_extract_pred_answer`. Match via `_answers_match` (math-equivalence).
        Examples whose GOLD answer cannot be extracted are treated as correct, so
        they keep the on-policy trajectory PI rather than being handed an
        unverifiable gold target.

        Returns:
            correct: [B] bool on `completion_ids.device`.
        """
        tok = self.processing_class
        B = completion_ids.shape[0]
        comp_texts = tok.batch_decode(completion_ids, skip_special_tokens=True)
        correct = torch.ones(B, dtype=torch.bool, device=completion_ids.device)
        for b in range(B):
            gold_text = tok.decode(
                answer_ids[b][answer_mask[b].bool()], skip_special_tokens=True
            )
            gold = self._extract_boxed_answer(gold_text)
            if gold is None:
                continue  # unparseable gold -> keep the on-policy trajectory PI
            pred = self._extract_pred_answer(comp_texts[b])
            correct[b] = self._answers_match(gold, pred)
        return correct

    def _concat_prompt_solution(self, prompt_ids, prompt_mask, sol_ids, sol_mask, max_sol=None):
        """Left-padded ``[teacher_prompt | gold_solution]`` per row (real tokens only).

        Drops each row's prompt (left-pad) and solution (right-pad) padding,
        concatenates so the gold solution sits directly AFTER the prompt — and
        therefore directly BEFORE the completion the caller appends — then
        left-pads the batch. Pads land only on the outer left edge, never
        between the prompt, the solution, or the completion (Dream's
        bidirectional attention treats interior pads as real positions, so this
        matters).
        """
        pad_id = self.processing_class.pad_token_id
        device = prompt_ids.device
        B = prompt_ids.shape[0]
        rows = []
        for b in range(B):
            p = prompt_ids[b][prompt_mask[b].bool()]
            s = sol_ids[b][sol_mask[b].bool()]
            if max_sol is not None and s.numel() > max_sol:
                s = s[:max_sol]
            rows.append(torch.cat([p, s]))
        max_len = max(int(r.numel()) for r in rows)
        ids = torch.full((B, max_len), pad_id, dtype=prompt_ids.dtype, device=device)
        mask = torch.zeros((B, max_len), dtype=prompt_mask.dtype, device=device)
        for b, r in enumerate(rows):
            n = int(r.numel())
            ids[b, max_len - n :] = r
            mask[b, max_len - n :] = 1
        return ids, mask

    def _teacher_masked_gold_solution(
        self,
        model,
        teacher_prompt_ids,
        teacher_prompt_mask,
        answer_ids,
        answer_mask,
        noisy_completion,
        completion_mask,
        mask_pattern,
        L_c,
        transition_ids=None,
        transition_mask=None,
    ):
        """Teacher target for WRONG rollouts: the FULL gold solution as PI.

        The gold solution is prepended to the teacher prompt as privileged
        context; the teacher then *re-scores the student's own NOISY rollout*,
        predicting the step-`k` masked positions conditioned on
        ``[problem | gold solution | partial rollout]``. Crucially the scored
        positions stay <mask> for the teacher too (only the surrounding context
        gains the gold solution) — the teacher is never handed the answer token
        at a scored position, so the privilege is the gold solution itself and
        the teacher must transfer it onto the student's trajectory. This keeps
        the scored sequence identical to the student's (same `mask_pattern`,
        same positions), so the JSD is well-aligned.

        Returns:
            [N, V] teacher LOGITS gathered at `mask_pattern` (row-major (b, l)),
            aligned to `student_completion_logits[mask_pattern]`.
        """
        gold_prompt_ids, gold_prompt_mask = self._concat_prompt_solution(
            teacher_prompt_ids, teacher_prompt_mask, answer_ids, answer_mask,
            max_sol=self.gold_pi_max_solution_tokens,
        )
        g_prompt_len = gold_prompt_ids.shape[1]
        full_ids, full_mask = self._build_teacher_full(
            gold_prompt_ids, gold_prompt_mask, noisy_completion, completion_mask,
            transition_ids, transition_mask,
        )
        teacher_ctx = (
            self.accelerator.unwrap_model(model).disable_adapter()
            if (self.fixed_teacher and is_peft_model(model))
            else nullcontext()
        )
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad(), teacher_ctx:
                logits = self._forward(model, full_ids, full_mask)
                comp = logits[:, g_prompt_len : g_prompt_len + L_c, :].float().detach()
        finally:
            if was_training:
                model.train()
        return comp[mask_pattern]  # [N, V]

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
        mask_pattern, p_mask_sample, chosen = self._trajectory_mask(
            completion_ids, jsd_valid_mask, history
        )
        noisy_completion = torch.where(
            mask_pattern,
            torch.full_like(completion_ids, self.mask_token_id),
            completion_ids,
        )

        # 3. Build student input. The student always sees the NOISY (real
        #    decoding-step `k`) completion. The teacher's privileged view is set
        #    by `traj_teacher_view` / `traj_teacher_gap` (see step 5). JSD is
        #    always taken at the step-`k` masked positions (`mask_pattern`).
        #
        #    By default the trajectory collator gives the teacher the SAME prompt
        #    as the student (problem only) — the teacher input stays in the model's
        #    native decode format and the privilege comes purely from the
        #    completion tokens. Optional ablation framing (decode_intro_prompt /
        #    transition_prompt) is folded into `teacher_prompt_ids` / appended as a
        #    `teacher_transition_input_ids` suffix; the intro is already part of
        #    `teacher_prompt_ids`, so the teacher completion span still starts at
        #    `t_prompt_len`, and the transition (if any) sits after the span.
        transition_ids = inputs.get("teacher_transition_input_ids")
        transition_mask = inputs.get("teacher_transition_attention_mask")

        student_full_ids = torch.cat([student_prompt_ids, noisy_completion], dim=1)
        student_full_mask = torch.cat([student_prompt_mask, completion_mask], dim=1)

        # 4. Student forward (with grad). Slice out the completion span only.
        student_logits = self._forward(model, student_full_ids, student_full_mask)
        student_completion_logits = student_logits[:, s_prompt_len : s_prompt_len + L_c, :]
        del student_logits
        student_masked = student_completion_logits[mask_pattern]  # [N, V]

        # 4b. Verify the student's final rollout against the gold solution. This
        #     gates the teacher's privileged information per example (step 5):
        #       * correct rollout -> on-policy trajectory PI (future decoding
        #         steps / endpoint), exactly as before.
        #       * wrong rollout   -> the FULL gold solution becomes the teacher's
        #         privileged context and it re-scores the student's own noisy
        #         rollout, so we stop distilling a confidently-wrong answer and
        #         instead pull the student toward the gold-informed distribution.
        B0 = completion_ids.shape[0]
        answer_ids = inputs.get("answer_input_ids")
        answer_mask = inputs.get("answer_attention_mask")
        if self.verify_gold_pi and answer_ids is not None:
            correct = self._verify_rollouts(completion_ids, answer_ids, answer_mask)
        else:
            correct = torch.ones(B0, dtype=torch.bool, device=completion_ids.device)
        # Per-scored-position example index, in the row-major order of every
        # `[mask_pattern]` gather below — so a [N] boolean indexes [N, V] targets.
        row_b = mask_pattern.nonzero(as_tuple=False)[:, 0]  # [N]
        row_correct = correct[row_b]                         # [N]

        # 5. Teacher target at the scored (step-`k` masked) positions. Rows are
        #    filled from the correct branch (on-policy trajectory PI) and the
        #    wrong branch (gold-solution PI) independently; each branch is skipped
        #    entirely when no row needs it.
        #      * "all_future"      — average the teacher's predictive distribution
        #        over steps k+1→final where each position is still masked (many
        #        teacher forwards; returns log-probs directly).
        #      * "snapshot" + gap None — the CONCRETE final rollout (endpoint).
        #      * "snapshot" + gap n    — the state `n` steps ahead (history[k+n]).
        teacher_masked = torch.zeros_like(student_masked, dtype=torch.float32)  # [N, V]
        teacher_completion = None
        teacher_step = torch.full((B0,), -1, dtype=torch.long, device=completion_ids.device)

        # --- CORRECT rollouts: on-policy trajectory PI (unchanged). ---
        if bool(row_correct.any()):
            if self.traj_teacher_view == "all_future":
                std_masked = self._teacher_masked_all_future(
                    model, teacher_prompt_ids, teacher_prompt_mask, completion_mask,
                    completion_ids, history, chosen, mask_pattern,
                    t_prompt_len, L_c, transition_ids, transition_mask,
                )  # [N, V] log-probs
            else:
                if self.traj_teacher_gap is None:
                    teacher_completion = completion_ids
                else:
                    teacher_completion, teacher_step = self._teacher_completion_n_ahead(
                        completion_ids, history, chosen, self.traj_teacher_gap
                    )
                teacher_full_ids, teacher_full_mask = self._build_teacher_full(
                    teacher_prompt_ids, teacher_prompt_mask, teacher_completion,
                    completion_mask, transition_ids, transition_mask,
                )
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
                std_masked = teacher_completion_logits[mask_pattern]  # [N, V]
            teacher_masked[row_correct] = std_masked[row_correct]

        # --- WRONG rollouts: full gold solution as the teacher's PI. ---
        if bool((~row_correct).any()):
            gold_masked = self._teacher_masked_gold_solution(
                model, teacher_prompt_ids, teacher_prompt_mask,
                answer_ids, answer_mask, noisy_completion, completion_mask,
                mask_pattern, L_c, transition_ids, transition_mask,
            )  # [N, V] logits
            teacher_masked[~row_correct] = gold_masked[~row_correct]

        # 6. Token-level JSD over the scored positions.

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
            # Verify-gated gold-PI diagnostics: the rollout accuracy (the
            # curriculum signal — should rise over training) and how many scored
            # positions were routed to the gold-solution teacher this step.
            self._metrics[mode]["frac_rollout_correct"].append(float(correct.float().mean().item()))
            self._metrics[mode]["num_gold_pi_scored"].append(float((~row_correct).sum().item()))
            # Snapshot diagnostics only (all_future logs `teacher_future_steps_avg`
            # from inside `_teacher_masked_all_future`). Of the scored positions,
            # the fraction the teacher STILL sees as <mask> (genuinely predictive)
            # vs. concrete (sees the answer → near-degenerate copy target).
            if teacher_completion is not None and bool(mask_pattern.any()):
                teacher_still_masked = (
                    teacher_completion[mask_pattern] == self.mask_token_id
                ).float().mean().item()
                self._metrics[mode]["teacher_scored_frac_still_masked"].append(float(teacher_still_masked))
                valid_step = teacher_step >= 0
                if bool(valid_step.any()) and chosen is not None:
                    self._metrics[mode]["teacher_step_gap_mean"].append(
                        float((teacher_step[valid_step] - chosen[valid_step]).float().mean().item())
                    )
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
