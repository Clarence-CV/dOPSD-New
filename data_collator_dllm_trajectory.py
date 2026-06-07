import torch

from data_collator_dllm import SelfDistillationDLLMDataCollator


class SelfDistillationDLLMTrajectoryDataCollator(SelfDistillationDLLMDataCollator):
    """Collator for trajectory-OPSD (see opsd_dllm_trajectory_trainer.py).

    In trajectory mode the teacher's privileged information is the student's own
    decoding trajectory (its decoding-step / final-rollout tokens), NOT the
    dataset's ground-truth solution — so `solution` is dropped from the teacher
    prompt.

    DEFAULT (recommended): NO framing. The teacher prompt is exactly the student
    prompt (problem only + assistant header), and the trainer appends the runtime
    decoding step as the assistant response:

        [ teacher_prompt(problem) | <decoding step> ]

    This keeps the teacher input in the model's NATIVE decode format
    (`[problem][partial completion]`) — the same format it sees at inference — so
    the teacher understands the decoding step in-distribution and the privilege
    comes purely from the completion tokens, with no out-of-distribution prompt
    text and no teacher/student conditioning mismatch.

    OPTIONAL (ablation only): pass non-empty `decode_intro_prompt` and/or
    `transition_prompt` to wrap the step (mirroring `data_collator.py`'s
    `reason_first`):

        [ teacher_prompt(problem + decode_intro_prompt) | <decoding step> | transition_prompt ]

      * `decode_intro_prompt`  — folded into the teacher's chat-templated prompt
        (text BEFORE the step).
      * `transition_prompt`    — emitted as `teacher_transition_input_ids`, which
        the trainer appends AFTER the step.

    Both add prompt text the model never saw during training/generation, so they
    are out-of-distribution and disabled by default. The student side is always
    unchanged (problem only), so the completion spans stay directly comparable.
    """

    # Optional framing templates (OFF by default). The recommended/default setup
    # is NO framing: the teacher's input is the same in-distribution format the
    # model actually decodes in — `[problem][partial completion]` — so the
    # privilege comes purely from the completion (decoding-step) tokens, not from
    # out-of-distribution prompt text. Pass a non-empty string (or None to use
    # these templates) only for ablations.
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
        # "" (default) = OFF; None = use the DEFAULT_* template; any string = that text.
        self.decode_intro_prompt = (
            self.DEFAULT_DECODE_INTRO_PROMPT if decode_intro_prompt is None else decode_intro_prompt
        )
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
        # Privilege comes from the decoding trajectory, not the GT solution, so
        # `solution` is intentionally unused. With the default (no intro), the
        # teacher prompt is exactly the student prompt; `add_generation_prompt=True`
        # ends it at the assistant header, after which the trainer appends the
        # decoding step as the (privileged) assistant response.
        user_message = f"{problem}\n\n{self.decode_intro_prompt}".rstrip() if self.decode_intro_prompt else problem
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
