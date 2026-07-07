import torch


class SelfDistillationDLLMDataCollator:
    """Data collator for dLLM (diffusion LM) self-distillation, e.g. Dream-7B.

    Student prompt = problem only; teacher prompt = same, optionally with the
    reference solution injected as privileged context. Answer ids are the
    tokenized solution; the trainer masks tokens in this span for both models.
    Prompts are LEFT-padded and answers RIGHT-padded so pad tokens sit only on
    the outer edges of `[prompt | answer]` (matters for bidirectional attention).
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length=1024,
        max_answer_length=1024,
        problem_field: str = "problem",
        solution_field: str = "solution",
        use_privileged_info: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        self.problem_field = problem_field
        self.solution_field = solution_field
        # False (default) = no-PI baseline (teacher prompt == student prompt);
        # True = teacher also sees the ground-truth solution. Single knob for the
        # PI-vs-no-PI experiment.
        self.use_privileged_info = use_privileged_info

        # Minimal header labeling the gold solution. Kept bare on purpose: an
        # earlier verbose "explore/backtrack" + answer-format framing (teacher-only)
        # got distilled as STYLE, ~2x'ing student output length and hurting GSM8K.
        # Differing from the student prompt by the solution alone isolates the PI.
        self.reference_solution_header = "\n\nReference solution:\n"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"[DLLMCollator] pad_id={self.tokenizer.pad_token_id}, "
            f"mask_id={getattr(self.tokenizer, 'mask_token_id', None)}, "
            f"max_prompt_len={self.max_prompt_length}, max_answer_len={self.max_answer_length}, "
            f"fields=(problem={self.problem_field}, solution={self.solution_field})"
        )

    def _build_student_prompt(self, problem: str) -> str:
        # Raw question in the chat template (matches eval/Dream/dream_train.py):
        # no prefix, no boxed/step-by-step instruction.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compose_teacher_user_turn(self, problem: str, solution: str) -> str:
        # User turn = student's problem text + reference solution appended. No
        # extra framing (see __init__), so the prompts differ by the solution
        # alone. add_generation_prompt appends the assistant header for scoring.
        user_content = f"{problem}{self.reference_solution_header}{solution}"
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_teacher_prompt(self, problem: str, solution: str) -> str:
        # No-PI baseline: teacher prompt == student prompt (problem only).
        if not self.use_privileged_info:
            return self._build_student_prompt(problem)

        # PI teacher: embed the gold `solution` as privileged context the student
        # never sees; conditioning sharpens the teacher's masked-position
        # distribution, which OPSD's JSD distills into the student.
        #
        # Token-budget the solution so the assembled prompt fits max_prompt_length;
        # otherwise right-truncation would drop the trailing assistant header and
        # break prompt/completion alignment.
        skeleton = self._compose_teacher_user_turn(problem, solution="")
        skeleton_len = len(self.tokenizer(skeleton, add_special_tokens=False)["input_ids"])
        # Margin absorbs decode/re-encode token drift below.
        budget = self.max_prompt_length - skeleton_len - 8
        if budget <= 0:
            # No room: fall back to a non-privileged (but header-complete) prompt.
            return self._build_student_prompt(problem)
        sol_ids = self.tokenizer(solution, add_special_tokens=False)["input_ids"]
        if len(sol_ids) > budget:
            solution = self.tokenizer.decode(sol_ids[:budget])
        return self._compose_teacher_user_turn(problem, solution)

    def _left_pad(self, seqs, max_len, pad_id):
        ids, mask = [], []
        for s in seqs:
            pad_n = max_len - len(s)
            ids.append([pad_id] * pad_n + list(s))
            mask.append([0] * pad_n + [1] * len(s))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _right_pad(self, seqs, max_len, pad_id):
        ids, mask = [], []
        for s in seqs:
            pad_n = max_len - len(s)
            ids.append(list(s) + [pad_id] * pad_n)
            mask.append([1] * len(s) + [0] * pad_n)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def _tokenize(self, texts, max_length):
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

    def __call__(self, features):
        problems, solutions = [], []
        for feat in features:
            problems.append(feat[self.problem_field])
            solutions.append(feat[self.solution_field])

        student_prompts = [self._build_student_prompt(p) for p in problems]
        teacher_prompts = [
            self._build_teacher_prompt(p, s) for p, s in zip(problems, solutions)
        ]

        s_ids = self._tokenize(student_prompts, self.max_prompt_length)
        t_ids = self._tokenize(teacher_prompts, self.max_prompt_length)
        a_ids = self._tokenize(solutions, self.max_answer_length)

        pad_id = self.tokenizer.pad_token_id
        max_s = max(len(x) for x in s_ids)
        max_t = max(len(x) for x in t_ids)
        max_a = max(len(x) for x in a_ids)

        student_prompt_ids, student_prompt_mask = self._left_pad(s_ids, max_s, pad_id)
        teacher_prompt_ids, teacher_prompt_mask = self._left_pad(t_ids, max_t, pad_id)
        answer_ids, answer_mask = self._right_pad(a_ids, max_a, pad_id)
        answer_lengths = torch.tensor([len(x) for x in a_ids], dtype=torch.long)

        return {
            "student_prompt_input_ids": student_prompt_ids,
            "student_prompt_attention_mask": student_prompt_mask,
            "teacher_prompt_input_ids": teacher_prompt_ids,
            "teacher_prompt_attention_mask": teacher_prompt_mask,
            "answer_input_ids": answer_ids,
            "answer_attention_mask": answer_mask,
            "answer_lengths": answer_lengths,
        }
