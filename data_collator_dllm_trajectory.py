import torch

from data_collator_dllm import SelfDistillationDLLMDataCollator


class SelfDistillationDLLMTrajectoryDataCollator(SelfDistillationDLLMDataCollator):
    """Collator for trajectory-OPSD (see opsd_dllm_trajectory_trainer.py).

    Here the teacher's privilege is the student's own decoding trajectory, not
    the dataset solution, so `solution` is dropped from the teacher prompt.

    DEFAULT (recommended): NO framing — teacher prompt == student prompt
    (problem + assistant header); the trainer appends the decoding step as the
    assistant response: `[ teacher_prompt(problem) | <decoding step> ]`. This
    keeps the teacher input in the model's native decode format, so privilege
    comes purely from the completion tokens with no OOD prompt text.

    OPTIONAL (ablation): non-empty `decode_intro_prompt` (folded into the prompt,
    before the step) and/or `transition_prompt` (emitted as
    `teacher_transition_input_ids`, appended after the step). Both add OOD prompt
    text, so they are off by default. The student side is always unchanged.
    """

    # Framing templates, OFF by default (see class docstring); non-empty / None
    # only for ablations.
    DEFAULT_DECODE_INTRO_PROMPT = (
        "A complete, step-by-step solution to this problem is provided below as the "
        "response. Every step of it is correct and it reaches the right final answer. "
        "Reconstruct that exact reasoning faithfully and stay fully confident in each token."
    )
    DEFAULT_TRANSITION_PROMPT = (
        "\n\nThe step-by-step solution above is correct and reaches the right final answer. "
        "Every step is consistent with this reasoning."
    )

    def __init__(
        self,
        tokenizer,
        max_prompt_length=1024,
        max_answer_length=1024,
        problem_field: str = "problem",
        solution_field: str = "solution",
        target_field: str | None = None,
        decode_intro_prompt: str | None = "",
        transition_prompt: str | None = "",
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_answer_length,
            problem_field=problem_field,
            solution_field=solution_field,
        )
        self.target_field = target_field
        # "" (default) = OFF; None = use the DEFAULT_* template; any string = that text.
        self.decode_intro_prompt = (
            self.DEFAULT_DECODE_INTRO_PROMPT if decode_intro_prompt is None else decode_intro_prompt
        )
        self.transition_prompt = (
            self.DEFAULT_TRANSITION_PROMPT if transition_prompt is None else transition_prompt
        )

        # Pre-tokenize once (identical for every example).
        self._transition_ids = self.tokenizer(
            self.transition_prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        if not self.decode_intro_prompt and len(self._transition_ids) == 0:
            print(
                "[DLLMTrajectoryCollator] NO framing — teacher prompt == student prompt "
                "(problem only). Privilege comes solely from the completion (decoding-step) "
                "tokens; the teacher input stays in the model's native decode format."
            )
        else:
            print(
                "[DLLMTrajectoryCollator] teacher prompt frames the decoding step "
                f"(intro={len(self.decode_intro_prompt)} chars, "
                f"transition={len(self._transition_ids)} tokens) — ABLATION mode "
                "(adds out-of-distribution prompt text). Solution is NOT shown to the teacher."
            )

    def _build_teacher_prompt(self, problem: str, solution: str) -> str:
        user_message = f"{problem}\n\n{self.decode_intro_prompt}".rstrip() if self.decode_intro_prompt else problem
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, features):
        result = super().__call__(features)

        if len(self._transition_ids) > 0:
            B = len(features)
            trans = torch.tensor([list(self._transition_ids)] * B, dtype=torch.long)
            result["teacher_transition_input_ids"] = trans
            result["teacher_transition_attention_mask"] = torch.ones_like(trans)

        # Ground-truth final answer (`target`) for rollout verification.
        if self.target_field is not None and self.target_field in features[0]:
            targets = [str(f[self.target_field]) for f in features]
            t_ids = self.tokenizer(
                targets, padding=False, truncation=True, max_length=64,
                add_special_tokens=False,
            )["input_ids"]
            pad_id = self.tokenizer.pad_token_id
            max_t = max((len(x) for x in t_ids), default=1) or 1
            ids = torch.full((len(targets), max_t), pad_id, dtype=torch.long)
            mask = torch.zeros((len(targets), max_t), dtype=torch.long)
            for i, x in enumerate(t_ids):
                if x:
                    ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
                    mask[i, : len(x)] = 1
            result["target_input_ids"] = ids
            result["target_attention_mask"] = mask
        return result
