import torch


class SelfDistillationDLLMDataCollator:
    """Data collator for dLLM (diffusion LM) self-distillation, e.g. Dream-7B.

    Processes math-style (problem, solution) datasets. For each example:
      - student prompt: chat template containing only the problem with the
        "reason step by step, \\boxed{}" instruction.
      - teacher prompt: chat template containing the problem AND the reference
        solution, followed by a transition prompt that asks the teacher to
        rederive the answer in its own words.
      - answer ids:     tokenized reference solution; the trainer randomly
        masks tokens inside this span so both student and teacher predict the
        same masked positions.

    Padding strategy: prompts are LEFT-padded to the per-batch maximum prompt
    length and the answer is RIGHT-padded to the per-batch maximum answer
    length. When the trainer concatenates `[prompt | answer]`, pad tokens sit
    only on the outer edges of the full sequence (never between prompt and
    answer), which matters because the model's bidirectional attention treats
    every position equally.
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
        # When False (default) the teacher prompt is IDENTICAL to the student
        # prompt (problem only) — the no-privileged-information baseline. When
        # True the teacher additionally sees the ground-truth solution as
        # privileged context (see _build_teacher_prompt). This is the single
        # knob the PI-vs-no-PI controlled experiment toggles.
        self.use_privileged_info = use_privileged_info

        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        # GSM8K answers are plain integers. lm-eval's gsm8k extractors look for
        # the exact "#### <n>" line (strict-match) or the final number in the
        # text (flexible-extract). A \boxed{} / LaTeX instruction is AIME-style
        # and makes the student emit "$\boxed{...}$", which the GSM8K
        # flexible-extract regex mis-parses (it grabs "$" instead of the value).
        # Ask for the GSM8K final-answer format instead so the distilled student
        # produces directly extractable answers.
        self.answer_instruction = (
            "Please reason step by step, and give your final answer on a new line "
            "in the exact format: #### <answer>, where <answer> is a single number."
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"[DLLMCollator] pad_id={self.tokenizer.pad_token_id}, "
            f"mask_id={getattr(self.tokenizer, 'mask_token_id', None)}, "
            f"max_prompt_len={self.max_prompt_length}, max_answer_len={self.max_answer_length}, "
            f"fields=(problem={self.problem_field}, solution={self.solution_field})"
        )

    def _build_student_prompt(self, problem: str) -> str:
        # Match eval/Dream/dream_train.py: raw question wrapped in the chat
        # template, no "Problem:" prefix, no boxed/step-by-step instruction.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _compose_teacher_user_turn(self, problem: str, solution: str) -> str:
        # User turn = problem + the ground-truth reference solution + a
        # transition prompt that tells the model to re-derive the answer in its
        # own words. add_generation_prompt=True appends the assistant header so
        # the teacher generates from there.
        user_content = (
            f"{problem}\n\n"
            f"Reference solution:\n{solution}"
            f"{self.transition_prompt}"
            f"{self.answer_instruction}"
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_teacher_prompt(self, problem: str, solution: str) -> str:
        # No-PI baseline: teacher prompt == student prompt (problem only), so
        # on-policy JSD compares two predictive distributions on the SAME
        # condition. This is the configuration that reached the 82% GSM8K
        # baseline.
        if not self.use_privileged_info:
            return self._build_student_prompt(problem)

        # PRIVILEGED-INFORMATION TEACHER: the teacher's prompt embeds the
        # ground-truth `solution` as privileged context the student never sees.
        # Conditioned on the gold solution, the teacher's distribution over the
        # (masked) answer positions is sharpened; OPSD's JSD then distills that
        # distribution into the student, which is conditioned on the problem
        # alone. The reference solution IS the teacher's privileged information.
        #
        # The solution is token-budgeted so the assembled prompt fits inside
        # max_prompt_length. Otherwise the tokenizer's right-truncation in
        # _tokenize() would drop the trailing assistant generation header and
        # break the teacher forward's prompt/completion alignment.
        skeleton = self._compose_teacher_user_turn(problem, solution="")
        skeleton_len = len(self.tokenizer(skeleton, add_special_tokens=False)["input_ids"])
        # Leave a small margin to absorb decode/re-encode token drift below.
        budget = self.max_prompt_length - skeleton_len - 8
        if budget <= 0:
            # No room for the solution: fall back to a non-privileged prompt
            # rather than emitting a header-less (truncated) teacher prompt.
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
