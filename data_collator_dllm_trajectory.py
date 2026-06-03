import torch

from data_collator_dllm import SelfDistillationDLLMDataCollator


class SelfDistillationDLLMTrajectoryDataCollator(SelfDistillationDLLMDataCollator):
    """Collator for trajectory-OPSD (see opsd_dllm_trajectory_trainer.py).

    Differs from the base `SelfDistillationDLLMDataCollator` in how the *teacher*
    input is framed. In trajectory mode the teacher's privileged information is
    the student's own decoding trajectory (the concrete final rollout), NOT the
    dataset's ground-truth solution — so `solution` is dropped from the teacher
    prompt. Instead, mirroring `data_collator.py`'s `reason_first` structure, the
    teacher input is wrapped around the (runtime) decoding step with:

        [ teacher_prompt(problem + decode_intro_prompt) | <decoding step> | transition_prompt ]
          └──────────── before the step ────────────┘   └ runtime ┘   └── after the step ──┘

      * `decode_intro_prompt`  — folded into the teacher's chat-templated prompt
        (the text BEFORE the decoding step). It frames the completion that the
        teacher will see as its assistant response as a known-correct reasoning.
      * `transition_prompt`    — emitted as a separate token tensor
        (`teacher_transition_input_ids`) that the trainer appends AFTER the
        decoding step, exactly like `reason_first`'s `teacher_transition_tokens`.

    The student side is unchanged: the student still sees only the problem
    (`_build_student_prompt`), so its completion span stays directly comparable
    to the teacher's at the masked positions.

    Both prompt texts are overridable via the constructor so the framing can be
    ablated without touching code.
    """

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
        decode_intro_prompt: str | None = None,
        transition_prompt: str | None = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_answer_length,
            problem_field=problem_field,
            solution_field=solution_field,
        )
        self.decode_intro_prompt = (
            self.DEFAULT_DECODE_INTRO_PROMPT if decode_intro_prompt is None else decode_intro_prompt
        )
        # Note: base __init__ already set self.transition_prompt; override it with
        # the trajectory-appropriate wording (or the caller's).
        self.transition_prompt = (
            self.DEFAULT_TRANSITION_PROMPT if transition_prompt is None else transition_prompt
        )

        # Pre-tokenize the transition once (identical for every example).
        self._transition_ids = self.tokenizer(
            self.transition_prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        print(
            "[DLLMTrajectoryCollator] teacher prompt frames the decoding step "
            f"(intro={len(self.decode_intro_prompt)} chars, "
            f"transition={len(self._transition_ids)} tokens). Solution is NOT shown to the teacher."
        )

    def _build_teacher_prompt(self, problem: str, solution: str) -> str:
        # Privilege comes from the decoding trajectory, not the GT solution, so
        # `solution` is intentionally unused. The decode-intro framing goes in
        # the user turn; `add_generation_prompt=True` ends the prompt at the
        # assistant header, after which the trainer appends the decoding step as
        # the (privileged, concrete) assistant response.
        user_message = f"{problem}\n\n{self.decode_intro_prompt}".rstrip()
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, features):
        result = super().__call__(features)

        # Emit the transition tokens for the trainer to append after the decoding
        # step. Identical text for every example, so no padding is needed.
        if len(self._transition_ids) > 0:
            B = len(features)
            trans = torch.tensor([list(self._transition_ids)] * B, dtype=torch.long)
            result["teacher_transition_input_ids"] = trans
            result["teacher_transition_attention_mask"] = torch.ones_like(trans)
        return result
