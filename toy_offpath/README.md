# Toy experiment — "the noise is off the decoding path"

A small, self-contained experiment that makes one claim intuitive and measurable:

> A dLLM unmasks tokens in a **confidence-ordered, easy→hard** schedule. So at any
> masked fraction *p*, the **on-policy** partial state has a defining property:
> every revealed position was more confident than every still-masked one.
> A **uniform random** mask of the finished rollout breaks this — it reveals
> hard/late tokens and hides easy/early ones, producing a partial state the model
> **never visits**. The natural AR→dLLM port (re-mask a random subset of the
> rollout) therefore supervises *off the model's own decoding path*.

**Clean control.** In both partial states the *visible token values* are the final
rollout tokens (we re-mask the finished answer). The only thing that differs is
**which** positions are masked at level *p*. Any measured gap is due purely to the
masking pattern.

## Why the BASE model

This is a **motivation** experiment, so it must run on the **base**
`Dream-org/Dream-v0-Instruct-7B`, *not* your trained checkpoint. The phenomenon is
a property of the pretrained dLLM's decoding dynamics. Showing it on your
on-policy-distilled model would be circular ("of course it follows its own path —
you trained it to"). Use the trained model only as an optional secondary panel
that shows the property persists/sharpens.

## Files

- `prompts.py` — built-in short reasoning prompts used as an **offline fallback** (`--dataset none`); the default run uses gsm8k.
- `run_toy.py` — rolls out completions, **logs the unmask schedule** (commit step + commit confidence per position), builds on-policy vs uniform partial states, computes metrics.
- `visualize.py` — renders the 4-panel figure + prints the qualitative example.

## Run

```bash
cd toy_offpath

# 1) generate rollouts + compute metrics
#    default: 128 fixed-seed gsm8k test questions (matches the gsm8k_cot eval).
#    single A40 (48 GB), ~10-15 min.
python run_toy.py \
    --pretrained Dream-org/Dream-v0-Instruct-7B \
    --dataset gsm8k \
    --num_prompts 128 \
    --max_new_tokens 256 \
    --tokens_per_step 2 \
    --num_uniform_draws 4 \
    --out_dir results

# offline / quick smoke test: use the built-in prompts instead
#   python run_toy.py --dataset none --num_prompts 12 --max_new_tokens 128 --out_dir results_smoke

# 2) make the figure
python visualize.py --out_dir results
```

The gsm8k subset is a fixed-seed sample of the test split (reproducible via
`--seed`); it needs the `datasets` package (already in the project env) and a
one-time download. To check sample-size sufficiency, rerun with
`--num_prompts 256` and confirm the curves overlap within error bars.

Outputs (in `results/`):
- `offpath_figure.png` — the 4-panel figure.
- `results.json`, `m2_confidence.npz` — raw numbers.
- `qualitative_example.txt` — one on-policy vs uniform partial state side by side.

To confirm the property also holds (and sharpens) on your model, rerun with
`--pretrained /path/to/your/trained/checkpoint --out_dir results_trained`.

## What the figure shows (how to confirm the obstacle)

- **(a) Frontier cartoon.** Positions sorted by commit-confidence. On-policy mask is
  a clean threshold cut (the hard tail); uniform mask is scattered across the whole
  range — it visibly hides easy tokens and reveals hard ones.
- **(b) Off-path penalty (ΔNLL).** For positions masked under *both* schemes (same
  token, same position, only context differs), the model predicts the masked token
  **worse** from the uniform context. The curve sits **above 0** → the uniform
  partial state is a harder, off-distribution context. This is the rigorous,
  confound-free signal (token difficulty is held fixed by matching positions).
- **(c) Off-path fraction.** Fraction of uniform-masked tokens that are actually
  "easy" (above the on-policy frontier). On-policy is 0 by construction; uniform is
  large → it routinely masks tokens the model would have committed early.
- **(d) Masked-position confidence.** On-policy masks concentrate at low confidence
  (genuinely hard remaining tokens); uniform leaves high-confidence tokens masked —
  a configuration the model never produces during generation.

Panels (a)+(d) make the *structural* claim visceral; (b)+(c) make the *functional*
consequence quantitative. Together they confirm the obstacle.

## Notes / knobs

- Decoding is greedy entropy (`temperature 0`) for a deterministic, reproducible
  trajectory. `tokens_per_step` only controls speed — masks are ranked by the
  logged commit-confidence, so the schedule stays fine-grained regardless.
- The maskable region is the answer content up to the first EOS (trailing
  pad/EOS positions are excluded so they don't dominate).
- `--num_uniform_draws` averages several random masks per *p* for a smooth band.
