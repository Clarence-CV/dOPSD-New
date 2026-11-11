# Why EAGLE-style drafter training is stable while OPSD collapses (survey, 2026-11-10)

Web survey (arXiv HTML/abstracts). Companion to `doc/lit_graders_dllm_rl.md`, `doc/lit_sibling_remask.md`.
Question: drafters (EAGLE family) are trained to predict several tokens the target will accept, same goal as our
parallel-decoding training, yet never collapse; our OPSD runs and THU d-OPSD peak and collapse.

## Per-paper facts

| Paper | Labels | Train-inference mismatch | Data | Notes |
|---|---|---|---|---|
| EAGLE (ICML'24, 2401.15077) | frozen target's teacher-forced features (Smooth-L1) + CE through the target LM head | no rollout; U(-0.1,0.1) feature noise | fixed ShareGPT; target-generated 2.88x vs fixed 2.78x | n-alpha drops 0.79 -> 0.67 with self-predicted inputs |
| EAGLE-2 (2406.16858) | same | — | — | draft confidence ~ acceptance rate ("well calibrated"); dynamic tree |
| **EAGLE-3** (NeurIPS'25, 2503.01840) | tokens / target soft distribution; no feature regression | **training-time test**: draft's own hidden state replaces the target feature at later positions and is fed back (paper 1+2 steps; SpecForge default 7, 0.8^k weights). Fig. 4: dropping the feature constraint alone leaves 1-alpha very low; TTT flattens the n-alpha curve; tau 4.05 -> 5.37 -> 6.13 | **target-generated responses**, ~8x more data, acceptance keeps rising with data | inputs on-policy, labels frozen |
| SpecForge (2603.18567) | EAGLE-3 recipe | TTT length ablated 1..17: 1 -> 3 sharpest; math/code peak ~13 | regenerating responses with the target helps; data coverage matters | EAGLE-1's "low data sensitivity" contradicted |
| **HASS** (ICLR'25, 2408.15766) | Top-K (K=5) of the target distribution | Align-k: step j uses the draft's own features (detached) as inputs; labels stay the target's. 1/2/3/4/5 steps: 4.78/5.11/5.15/5.16/5.11. Feeding draft *tokens* instead of features hurt (4.20 vs 4.34) | fixed ShareGPT; self-distillation marginal/negative at 13B | names EAGLE's problem "feature-level exposure bias" |
| Falcon (AAAI'25, 2412.12639) | target dist KL + token CE | CSGD: replace a decaying fraction of draft outputs with the target's | — | k=2 semi-AR; k>=3 degrades |
| Medusa / Medusa-2 (2401.10774) | CE on t+k+1 tokens, 0.8^k | — | **self-distillation (target answers prompts)**: training the backbone on it with hard labels degrades it -> KL anchor to the frozen original | no collapse once anchored |
| Hydra (COLM'24) | CE to the base's distribution ("teacher loss", best single change) | hidden-state noise hurt | ShareGPT | — |
| DistillSpec (ICLR'24, 2310.08461) | f-divergences to the frozen target | draft-generated / target-generated / mixed data compared; Thm 4.1 acceptance >= f(on-policy KD loss) | — | only degradation seen: fixed-data SupKD late in training |
| ReDrafter, FastDraft, Osprey, DFlash (2602.06036), VAT (2608.30135), PARD-2 | all: frozen target's tokens or soft distribution | DFlash: clean anchor per block, no self-conditioning; VAT: simulate verification, loss re-anchored at the first rejection + 1-layer accept head; PARD-2: weight positions by the cumulative target confidence of the prefix | all moved to target-generated responses | |
| **Nemotron-Labs-Diffusion** (NVIDIA, 2607.05722) | frozen AR verifier's tau=3 softened distribution (stop-grad, top-200) + CE on its argmax; **loss only on the accepted prefix + first rejected position** | **90% of slots use drafter-sampled tokens** (on-policy drafts) | — | the nearest "model is its own drafter": diffusion pass drafts, AR pass verifies; acceptance 6.82 vs 2.75 EAGLE-3 |
| dLLM self-speculation, training-free: SSD (2510.04147), Spiffy (2509.18085), FreeDave (2510.00294), Trajectory-SD (2608.27514), FailFast, DiffuSpec, PSD, SimSD | target = the same model's confidence-ordered 1-token/step (or top-S) path; verify = next-step prediction equals the draft | lossless for blockwise-causal / dual cache, near-lossless for bidirectional LLaDA (bf16) | — | failure mode = out-of-order commitment / "trajectory drift". **Nobody trains the dLLM as its own drafter against its own serial decoding.** |

## OPSD-side evidence

- **d-OPSD THU (2606.18195)**: "prone to policy collapse after achieving peak performance"; without pointwise KL clipping GSM8K collapses around step 150 (69.37 at step 500); fixed teacher beats unfixed; stabilization "future work"; best-of-25-steps checkpoint selection.
- **SDSD (2606.26091)**: the optimal self-distillation policy = base policy tilted by the pointwise conditional mutual information between the rollout and the self-sampled demonstration, not by reward -> rich-get-richer on dominant modes: pass@1 up, pass@k flat/down; external diverse demonstrations did not prevent the collapse; token entropy is not a sufficient diversity metric.
- **One Symptom, Three Levers (2608.25936)**: the teacher is "no stronger than the student, only better informed"; the asymmetry that generates the signal also biases it; levers = where (token weighting), what (privileged info), when (teacher dynamics).
- **OPTD (2608.02942)**: frozen question-only teacher; set-bottleneck objective pushing verified future candidates to the decoder's release threshold + frozen-teacher KL anchor elsewhere; consistency-guided compression = longest confidence-ordered prefix whose joint commitment preserves the teacher's outcome. Abstract only; directly adjacent to our goal — read in full.
- dOPSD NUS (2607.04428): no collapse curve reported (best checkpoint).

## Synthesis

1. Stable recipes share one property: **the label-generating process is frozen and external to the trained parameters**, and acceptance depends on it only through argmax/soft agreement. The draft never writes its own labels, so its sharpness cannot feed back.
2. In OPSD the teacher is the student plus privileged info: every update moves the labels. SDSD gives the fixed point (MI tilt, pass@k collapse); THU confirms the curve; Medusa-2 saw the same with self-generated hard labels and fixed it with a frozen-model KL anchor.
3. Correctness filtering does not fix it (SDSD; our path_v1: a drifting student satisfies "correct" with loops).
4. Granularity matters: drafters match the frozen target's *distribution* at on-policy states (Top-K, softened); reverse KL / JSD to a sharpening teacher is mode-seeking on a moving target (THU's own hypothesis).
5. **"On-policy states + frozen labels" is the documented fix for exposure bias**: EAGLE-3 Fig. 4/7, HASS Align-k, Nemotron 90% sampled drafts. What matters is realistic input states while labels stay anchored; feeding draft tokens as *labels* hurts (HASS).
6. Transfer to a dLLM as its own drafter: freeze the base; states = the student's own rollouts; labels = the frozen base re-predicting the masked positions with the committed siblings revealed (the LOO / sibling-marginalized target), matched as a soft distribution; weight by verification (loss on the accepted prefix + first rejection; VAT/Nemotron/PARD-2), which maps onto "how many tokens to commit this step".
7. Lossless target for dLLMs is settled: the same model's serial (1 token/step) confidence-ordered path, verified by next-step agreement (SSD/Spiffy/FreeDave). With that target, **no gold answer and no grader are needed for the training signal**; correctness is inherited from the base.
8. Pitfalls: exposure bias across depth (n-alpha), too many simulated steps saturate/hurt small drafters (HASS 5, SpecForge >13), noise augmentation inconsistent (EAGLE yes, Hydra no), train/inference block-size mismatch (DFlash large -> small only), monitor pass@k / semantic diversity, not accuracy alone (SDSD).

Unverified: EAGLE-4 training changes (source-available only); "SpecDiff", "DREAM-SD" as distinct papers; OPTD/PSD/SimSD full methods (abstract only).

URLs: EAGLE-3 https://arxiv.org/abs/2503.01840 · SpecForge https://arxiv.org/abs/2603.18567 · HASS https://arxiv.org/abs/2408.15766 · Medusa https://arxiv.org/abs/2401.10774 · DistillSpec https://arxiv.org/abs/2310.08461 · VAT https://arxiv.org/abs/2608.30135 · PARD-2 https://arxiv.org/abs/2605.08632 · DFlash https://arxiv.org/abs/2602.06036 · Nemotron-Labs-Diffusion https://arxiv.org/abs/2607.05722 · SSD https://arxiv.org/abs/2510.04147 · Spiffy https://arxiv.org/abs/2509.18085 · FreeDave https://arxiv.org/abs/2510.00294 · Trajectory-SD https://arxiv.org/abs/2608.27514 · d-OPSD https://arxiv.org/abs/2606.18195 · SDSD https://arxiv.org/abs/2606.26091 · Three Levers https://arxiv.org/abs/2608.25936 · OPTD https://arxiv.org/abs/2608.02942
