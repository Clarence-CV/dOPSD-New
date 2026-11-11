# Answer graders in recent dLLM post-training papers (survey, 2026-11-09)

Web survey (arXiv HTML + public code) done after run path_v1 collapsed on truncated-but-"correct" rollouts
(`results/path_opsd/path_v1_collapse.md`). "Not stated" = checked and not found.

## Table

| Paper (venue/yr) | Grader | Extraction | Truncation / EOS / format |
|---|---|---|---|
| d1 / diffu-GRPO (NeurIPS 2025) | GSM8K exact string on `<answer>`; MATH500 Hendrycks `is_equiv` (strip_string, no sympy) | last boxed (train); eval GSM8K first boxed | **no EOS/length check**; format rewards (tags, boxed) |
| wd1 / wd1++ (2507.08838) | d1 rewards; wd1++ HF Math-Verify | d1-style | none stated; max len 256 |
| TraceRL / TraDo (ICLR 2026) | own `is_equal`: strip_string exact -> sympy or math_verify, 1 s timeout; repetition filter | last boxed, fallbacks | **train: `len >= max_gen_length-5` -> wrong**; groups with acc >0.8 / <0.2 dropped |
| DiRL / DiPO (2512.22234) | last boxed -> math_verify OR parse_latex, 30 s timeout | last boxed only | **`len >= max_tokens - block_size` -> 0**; optional overlong / repetition penalties |
| StepRS-GRPO (2610.00661) | boxed extractor + math verifier (unnamed) | boxed span before first termination token | **no termination token -> reward 0** regardless of boxed (eval; train not explicit) |
| AGRPO (2510.04019) | HF math-verify | boxed in answer tags | none stated |
| d-OPSD THU (NeurIPS 2026) | verbatim d1 rewards | `<answer>` / last boxed | none; reports post-peak collapse; excludes Dream (format instability) |
| dOPSD NUS (2607.04428) | exact match / "mathematical equivalence" (unnamed) | extracted final answer | none |
| dParallel, Fast-dLLM v1/v2, Dream, D2F | lm-eval-harness `minerva_math` (+ `math_verify` metric), `gsm8k_cot` flexible-extract | harness defaults (last boxed / last number) | eval only, no EOS rule |
| DMax (2604.08302) | own multi-stage: normalized exact -> sympy -> math_verify snippet | up to 10 candidates: near-end marker, last boxed, last LaTeX block (last 800 chars), last 3 lines (+ RHS of `=`), whole text if <= 200 chars | **no finish check at scoring**; training data: "discard incomplete generations" |
| LLaDA 1.5 / 2.0 / 2.1 | DPO / EBPO; verifier not described | not stated | not stated |

## Evidence on lenient graders
- hkust-nlp 2505.22203: rule verifiers (verl, Qwen-Math, Math-Verify) precision ~0.99-1.0, recall 0.78-0.88; model verifiers get hacked.
- Fuzzing RLVR verifiers 2606.01066: requiring an explicit final-answer marker cut math false positives 0.833 -> 0.233; "stricter is not always better" (coverage cost).
- MAPoRL 2502.18439: models append junk after `\boxed{}`; penalty on tokens after the box.
- No dLLM paper reports the truncated-but-boxed hack explicitly, but TraceRL, DiRL and StepRS-GRPO all guard against it by construction; the d1 lineage (d1, wd1, d2, d-TreeRPO, d-OPSD) does not, and it is the lineage reporting collapse.

## Recommendation
1. **Training acceptance (strict):** terminated before the budget (EOS present / `len < max - margin`) AND last `\boxed{}` present AND Math-Verify match (`LatexExtractionConfig(boxed_match_priority=0)`, timeout -> wrong); optional strip_string exact pre-check; no tail-number fallback.
2. **Evaluation (lenient, comparability):** keep DMax `val_math.py` or lm-eval `minerva_math`+math_verify, reported separately. DMax's matching stages are fine; its weakness is the extraction fallbacks + no finish check.
3. Log accuracy under both graders; the gap is a diagnostic.
4. Paper: state grader + config, extraction rule, truncation rule, generation budget (TraceRL App. B, StepRS-GRPO App. C.3 as templates).

URLs: d1 https://arxiv.org/abs/2504.12216 ; TraceRL https://arxiv.org/html/2509.06949 ; DiRL https://arxiv.org/html/2512.22234v1 ; StepRS-GRPO https://arxiv.org/html/2610.00661 ; AGRPO https://arxiv.org/html/2510.04019v1 ; d-OPSD https://arxiv.org/html/2606.18195 ; dOPSD https://arxiv.org/html/2607.04428 ; DMax https://arxiv.org/html/2604.08302 ; Math-Verify https://github.com/huggingface/Math-Verify ; verifier robustness https://arxiv.org/html/2505.22203v2 ; fuzzing verifiers https://arxiv.org/html/2606.01066 ; MAPoRL https://arxiv.org/pdf/2502.18439
