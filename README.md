# Joint-reference OPSD on LLaDA2.0-mini

On-policy self-distillation for block-diffusion LLMs: the frozen model drafts early (a readout of an intermediate
layer), the confident drafts of one denoising step are fed back as soft messages to each other through a small
trainable injector, and a leave-one-out Teacher (same frozen model, siblings revealed) supplies the training target.
Goal: a better accuracy–latency trade-off of parallel decoding (more tokens per forward at equal accuracy).
Method and evidence: [doc/Selective_Draft_OPSD.md](doc/Selective_Draft_OPSD.md); engineering plan:
[doc/llada2-mini_engineering_plan.md](doc/llada2-mini_engineering_plan.md); handoff: [doc/llada2-mini_handoff.md](doc/llada2-mini_handoff.md).

## Layout

| path | what |
|---|---|
| `selective_draft/` | the package (about 2k lines): backend, decoder, guided model, trainers, evaluation, CLI |
| `selective_draft/vendor/` | DMax's MATH scorer (Apache-2.0, unmodified) and the official LLaDA2.0-mini remote code used by the CPU tests |
| `configs/` | `llada2_mini_math.json` (the real run), `fake_llada2_cpu.json` (CPU toy model), the question-level split manifest |
| `tests/` | CPU contract tests on a tiny model built from the official LLaDA2.0-mini code |
| `tacc/` | Lonestar6: environment, one resumable pipeline job, sync/submit, queue status |
| `results/`, `doc/` | experiment records and the written plan |

## Decoding protocol (DMax's evaluation of LLaDA2.0-mini)

Block 32 counted from position 0 (the prompt tail may share the first block), greedy, commit every position with
confidence >= min(0.95, best - 1e-5) (positions whose argmax is MASK are not eligible), stop after the block that
contains an EOS, total length floored to a multiple of 32 (`max_new_tokens` 2048), completion = every generated
non-EOS token, DMax's prompt (`<problem>\nLet's think step by step, and put your final answer within \boxed{}.`) and
DMax's scorer. NFE and TPF follow DMax's eval script (`nfe_dinfer`, `tpf_dinfer_mean`). Details and the reasons for
each choice are in the header of `selective_draft/generation.py`. An optional dInfer-style prefix KV cache
(`generation.cache_mode=prefix`) is token-identical to recomputing the prefix on the toy model (tested); it is off by
default and only used with guidance off.

## Use

```bash
# CPU tests (need transformers 4.57; plain scripts, no pytest)
for t in decoder guidance training data; do python tests/test_$t.py; done

# Lonestar6: once on a login node, then one resumable job per stage list
bash tacc/setup_llada2_env.sh
STAGES="baseline" SBATCH_ARGS="-t 10:00:00" bash tacc/sync_and_submit_llada2.sh
STAGES="states profile readout joint jointeval" bash tacc/sync_and_submit_llada2.sh   # after the baseline looks right
bash tacc/qstatus.sh <JOBID>                       # queue position and a rough start estimate
```

`tacc/llada2_pipeline.slurm` documents the stages (baseline, states, profile, readout, joint, jointeval), their
stopping rules, and the knobs; resubmitting the same `RUN` resumes. CLI: `python -m selective_draft.cli --help`.

## History

LLaDA-8B Joint-reference OPSD (MATH v1: accuracy flat, steps -18%, latency -11%) is preserved at tag
`llada-joint-v1` (branch `joint`); the THU d-OPSD reproduction lives on branch `THUdOPSD`.
