import torch


class AR2DLLMDataCollator:
    """Data collator for AR-teacher -> dLLM-student on-policy distillation.

    The student is a diffusion LM (Dream-7B); the teacher is an autoregressive
    LM (Qwen3). The two models use *separate* tokenizers, so each side's prompt
    is tokenized with its own tokenizer. Per (problem, solution) example this
    collator produces three things:

      - student prompt:  chat-templated *problem only* (Dream tokenizer),
        left-padded. Dream's `diffusion_generate` rolls out the on-policy
        completion from here.

      - teacher reasoning prompt:  chat-templated problem + reference solution +
        an instruction to *analyse* that solution (teacher / Qwen3 tokenizer),
        left-padded — batched AR generation requires left padding. The trainer
        runs the teacher on this to obtain its privileged reasoning.

      - teacher transition tokens:  a fixed raw-text bridge (teacher tokenizer,
        `add_special_tokens=False`) appended *after* the generated reasoning,
        inside the same assistant turn. It tells the teacher to now derive the
        answer itself; the student's on-policy completion is teacher-forced
        directly after it.

    The reference solution is consumed *only* inside the teacher reasoning
    prompt — there is no separate answer span, because the student generates
    its own completion on-policy (exactly as in opsd_dllm_trainer).

    Padding: every sequence this collator emits is LEFT-padded. The student
    prompt feeds Dream's diffusion rollout; the teacher reasoning prompt feeds
    Qwen3's batched AR generation — both require left padding so the real
    tokens sit flush against the generation boundary.
    """

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        max_prompt_length: int = 1024,
        max_teacher_prompt_length: int = 3072,
        problem_field: str = "problem",
        solution_field: str = "solution",
        reason_first: bool = True,
    ):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_teacher_prompt_length = max_teacher_prompt_length
        self.problem_field = problem_field
        self.solution_field = solution_field
        self.reason_first = reason_first

        self.answer_instruction = (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        self.reason_instruction = (
            "Carefully analyze the reference solution above. Explain the key reasoning "
            "steps, the core ideas, and the problem-solving strategies it relies on. "
            "Do not introduce a different solution of your own here — only explain the "
            "reference solution."
        )
        # Raw-text bridge appended after the teacher's generated reasoning, inside
        # the same assistant turn. The student's on-policy completion is then
        # teacher-forced directly after this text.
        self.transition_text = (
            "\n\n---\n\nNow set the reference aside. Using your own words and "
            "independent reasoning, solve the problem stated above from scratch. "
            "Think step by step, and put your final answer within \\boxed{}.\n\n"
        )

        for tok in (student_tokenizer, teacher_tokenizer):
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

        print(
            f"[AR2DLLMCollator] student(pad={student_tokenizer.pad_token_id}, "
            f"mask={getattr(student_tokenizer, 'mask_token_id', None)}) | "
            f"teacher(pad={teacher_tokenizer.pad_token_id}, "
            f"eos={teacher_tokenizer.eos_token_id}) | "
            f"max_prompt_len={max_prompt_length}, "
            f"max_teacher_prompt_len={max_teacher_prompt_length}, "
            f"reason_first={reason_first}, "
            f"fields=(problem={problem_field}, solution={solution_field})"
        )

    @staticmethod
    def _apply_chat_template(tokenizer, user_msg: str, enable_thinking=None) -> str:
        """Apply a single-user-turn chat template, robust to `enable_thinking`.

        Dream's tokenizer template does not accept `enable_thinking`; Qwen3's
        does. We try with the kwarg and fall back without it on TypeError.
        """
        messages = [{"role": "user", "content": user_msg}]
        if enable_thinking is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_student_prompt(self, problem: str) -> str:
        user_msg = f"Problem: {problem}\n\n{self.answer_instruction}"
        return self._apply_chat_template(self.student_tokenizer, user_msg)

    def _build_teacher_reasoning_prompt(self, problem: str, solution: str) -> str:
        if self.reason_first:
            user_msg = (
                f"Problem: {problem}\n\n"
                f"Here is a correct reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n"
                f"{self.reason_instruction}"
            )
        else:
            # No reasoning phase: the teacher prompt is just problem + solution;
            # the transition text alone carries the "now solve it" instruction.
            user_msg = (
                f"Problem: {problem}\n\n"
                f"Here is a reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ==="
            )
        # enable_thinking=False keeps Qwen3's analysis as plain text (no <think>).
        return self._apply_chat_template(
            self.teacher_tokenizer, user_msg, enable_thinking=False
        )

    @staticmethod
    def _tokenize(tokenizer, texts, max_length):
        return tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

    @staticmethod
    def _left_pad(seqs, max_len, pad_id):
        ids, mask = [], []
        for s in seqs:
            pad_n = max_len - len(s)
            ids.append([pad_id] * pad_n + list(s))
            mask.append([0] * pad_n + [1] * len(s))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __call__(self, features):
        problems, solutions = [], []
        for feat in features:
            problems.append(feat[self.problem_field])
            solutions.append(feat[self.solution_field])

        student_prompts = [self._build_student_prompt(p) for p in problems]
        teacher_prompts = [
            self._build_teacher_reasoning_prompt(p, s)
            for p, s in zip(problems, solutions)
        ]

        s_ids = self._tokenize(self.student_tokenizer, student_prompts, self.max_prompt_length)
        t_ids = self._tokenize(
            self.teacher_tokenizer, teacher_prompts, self.max_teacher_prompt_length
        )

        student_prompt_ids, student_prompt_mask = self._left_pad(
            s_ids, max(len(x) for x in s_ids), self.student_tokenizer.pad_token_id
        )
        teacher_reasoning_prompt_ids, teacher_reasoning_prompt_mask = self._left_pad(
            t_ids, max(len(x) for x in t_ids), self.teacher_tokenizer.pad_token_id
        )

        # The transition text is identical for every example, so all rows have
        # the same length — a plain tensor stack needs no padding.
        transition_ids = self.teacher_tokenizer(
            [self.transition_text] * len(features),
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        transition_ids = torch.tensor(transition_ids, dtype=torch.long)

        return {
            "student_prompt_input_ids": student_prompt_ids,
            "student_prompt_attention_mask": student_prompt_mask,
            "teacher_reasoning_prompt_input_ids": teacher_reasoning_prompt_ids,
            "teacher_reasoning_prompt_attention_mask": teacher_reasoning_prompt_mask,
            "teacher_transition_input_ids": transition_ids,
        }
