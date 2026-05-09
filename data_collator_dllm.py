import torch


class SelfDistillationDLLMDataCollator:
    """Data collator for dLLM (diffusion LM) self-distillation, e.g. Dream-7B.

    For each (problem, solution) example we build:
      - student prompt: chat template containing only the problem.
      - teacher prompt: chat template containing the problem AND the solution
        as a privileged reference ("here is a reference solution: ...").
      - answer ids:     tokenized solution; the trainer randomly masks tokens
        inside this span so both student and teacher predict the same masked
        positions.

    Padding strategy: prompts are LEFT-padded to the per-batch maximum prompt
    length and the answer is RIGHT-padded to the per-batch maximum answer
    length. When the trainer concatenates `[prompt | answer]`, this places
    pad tokens only on the outer edges of the full sequence (never between
    prompt and answer), which is important because the model's bidirectional
    attention treats every position equally.
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length=1024,
        max_answer_length=1024,
        instruction_field: str = "instruction",
        response_field: str = "response",
        context_field: str | None = "context",
    ):
        """
        Args:
            instruction_field: dataset column for the user instruction (Dolly: "instruction").
            response_field:    dataset column for the gold response (Dolly: "response").
            context_field:     optional dataset column for supporting context (Dolly: "context";
                               often empty string for open-ended tasks). Pass None if absent.
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length
        self.instruction_field = instruction_field
        self.response_field = response_field
        self.context_field = context_field

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"[DLLMCollator] pad_id={self.tokenizer.pad_token_id}, "
            f"mask_id={getattr(self.tokenizer, 'mask_token_id', None)}, "
            f"max_prompt_len={self.max_prompt_length}, max_answer_len={self.max_answer_length}, "
            f"fields=(instr={self.instruction_field}, resp={self.response_field}, ctx={self.context_field})"
        )

    def _build_prompts(self, instruction: str, response: str, context: str | None = None):
        ctx_block = f"\n\nContext:\n{context.strip()}" if context and context.strip() else ""

        student_user = f"{instruction.strip()}{ctx_block}"
        teacher_user = (
            f"{instruction.strip()}{ctx_block}\n\n"
            "Here is a reference response to this instruction:\n"
            f"=== Reference Response Begin ===\n{response.strip()}\n=== Reference Response End ===\n\n"
            "Using the reference response above as guidance, write your own response in your own words."
        )

        student_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": student_user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        teacher_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return student_prompt, teacher_prompt

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

    def __call__(self, features):
        student_prompts, teacher_prompts, answers = [], [], []
        for feat in features:
            instruction = feat[self.instruction_field]
            response = feat[self.response_field]
            context = feat.get(self.context_field) if self.context_field else None
            sp, tp = self._build_prompts(instruction, response, context)
            student_prompts.append(sp)
            teacher_prompts.append(tp)
            answers.append(response)

        s_ids = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        t_ids = self.tokenizer(
            teacher_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        a_ids = self.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=self.max_answer_length,
            add_special_tokens=False,
        )["input_ids"]

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
