# Inter-token dependence, remasking and train-time parallel-decoding safety in dLLMs (survey, 2026-11-09)

Web survey (arXiv abstracts/HTML; "abstract only" where noted). Companion to `doc/lit_graders_dllm_rl.md`.
Purpose: position the sibling / LOO / Path OPSD line (`doc/plan_a_loo_defer.md`, `doc/plan_path_opsd.md`).

## A. Remasking / correction / dependency-aware decoding (training-free unless stated)

| Paper | Diagnosis of sibling inconsistency | Inference action | Trains the marginal? mass goes where |
|---|---|---|---|
| Fast-dLLM (2505.22618) | Thm 1: if every marginal > 1-eps and (n+1)eps <= 1, argmax of product = argmax of joint; TV < (3n-1)eps/2; tight at eps = 1/(n+1) | commit all p >= tau | no |
| ReMDM (2503.00307, NeurIPS'25) | errors from "independence of the parallelized decoding"; no formal measure | remask decoded tokens w.p. sigma_t (confidence-based etc.), marginals preserved | no |
| WINO (2507.18578) | early-error accumulation | draft p >= tau1 (0.5-0.7); shadow block = LOO re-prediction; remask if shadow prob < tau2 (0.9) | no |
| **Roll Out and Roll Back / WINO+ (2605.16941)** | uses WINO-verified trajectories as supervision | after training: plain single-threshold decoding, no shadow block | **yes**: CE on positions WINO finalized at that step; **entropy-MAX (L_defer) on confident-and-wrong not-yet-finalized positions**; entropy-min on correct-but-low-confidence; GSM8K 73.24 -> 76.58 at 6.83x fewer steps |
| COVER (2602.06161) | WINO revisions ~90-97% flip-flop | exact LOO in one pass (KV override + diagonal correction); replace / remask by seed score | no |
| Remask, Don't Replace / T2M (2604.18738) | wrong visible tokens steer neighbours into confident wrong answers; [M] only lowers confidence | reset tokens with p(x_old) < 0.7 to [M]; AIME25 30 -> 43.3 on LLaDA2.1-mini | no |
| PC-Sampler, KLASS, LESS, Prophet, EB-Sampler, DUS | position bias / temporal stability / entropy sums / adjacency | selection rules | no |
| Cai & Li (2603.22248) | per-iteration KL error <= sum of mutual information among co-committed tokens | entropy-sum rule, E[T] = O~(H/eps) | no |
| PUNT (2510.21961) | conditional-independence test: KL with vs without anchors' sampled values | recursive anchor/test split, log2|M| passes/step | no; future work: distil into one-pass student |
| DAPD, DAWN, DOS, CLAD, DARD, TACG, RPD | attention-proxy dependency graphs | independent-set scheduling | no |
| DEMASK (2604.02560) | D_ij = E_{x_j}[TV(p(x_i), p(x_i | x_j revealed))], intervention labels | small Q/K head predicts D_ij; bounded-cumulative-dependency set | head only, marginals untouched |
| Mean-Field Parallel Decoding (2606.15805), Measured Coupling (2611.08307) | pairwise coupling (JSD similarity / log-margin drop after revealing j) | sigma fixed point over commit probabilities | no (explicitly refuse training); accuracy not always preserved |
| CoCommit (2607.04469, LLaDA2.1-mini) | factorized-commit error = conditional total correlation; confidence cannot recover it | add learned marker to bundle positions, re-run last 3 layers so they attend to each other; **no token values exchanged** | LoRA + marker, CE on clean tokens; +1.5 GSM8K, +5.5 HumanEval+ |
| Discrete Copula Diffusion (ICLR'25), APD (NeurIPS'25), CoDiLA (2603.20216) | marginals miss dependencies | external AR copula / AR verifier (CoDiLA conditions the verifier on soft embeddings of the dLLM marginals) | no (external) |
| PVF (2601.12247) | a masked position keeps several continuations alive | reveal a candidate, check other high-confidence top-1s unchanged | no |
| Re-evaluating Confidence Remasking (2606.12232), CaRE (2607.24763) | — | under greedy decoding WINO/Saber/COVER ~ Fast-dLLM at common block lengths; "the model, not the rule, is the bottleneck"; compute-matched rankings reverse | — |

Self-correction training (target for a wrong visible token): PRISM (2510; separate quality head whose minimiser is the LOO posterior p(x_i = y_i | y with i masked), marginal regularised to stay fixed), RemeDi (confidence head), Corrective DLM (ground truth under random corruption), ProSeCo, SCOPE, SCDD, Guided Star-Shaped (BCE error predictor), Informed correctors.

## B. Train-time methods for parallel decoding

| Paper | Target at a masked position | Train-inference handling | External layer at inference |
|---|---|---|---|
| dParallel (2509.26488) | hard token the frozen teacher committed in its own semi-AR trajectory + entropy-min only where student argmax = teacher token; no defer term | threshold 0.45-0.5; GSM8K 256 -> 30 steps | no |
| LLaDA2.0 CAP (2512.15745) | SFT + entropy-min on correctly predicted tokens | threshold 0.95 best, 0.85 "unacceptable" | no |
| DMax (2604.08302) | on-policy uniform training (recover clean token from own erroneous predictions) + soft parallel decoding | TPF 2.04 -> 5.47 GSM8K | no |
| SPEED (ICML'26 poster, no arXiv) | temp-scaled KL to teacher only at teacher-correct positions | slow-fast: defer tokens whose JSD with/without preceding block is high | partial probe |
| **CForce (2608.13925, inclusionAI)** | later-stage prediction distribution (stop-grad, same model); fwd KL + c_j reverse KL; CE anchor on newly revealed | **pre-collected self-rollouts at tau 0.95, fixed**; admits drift; TPF 3.60 -> 6.42, acc 84.3 -> 80.9 (LLaDA2.0-mini) | no |
| TSD (2609.15177) | own distribution at the commitment timestep (tau 0.9 trajectory), JSD beta 0.5, verifier-weighted; reverse KL collapsed | on-policy trajectories | no |
| CD4LM (2601.02236) | frozen teacher at a less-masked view (ground-truth revealed subset), softened KL | random mask ratios | no |
| T3D (2602.12262) | teacher's sampled clean tokens given intermediate states in teacher order; Thm 4.1 student conditional TC <= teacher's | TokPS 8 LLaDA avg 17.1 -> 47.1 | no |
| TIDE CompDemo (2604.26951) | teacher sees the complementary half of the masked set revealed (ground truth), two passes | not framed as marginalization | no |
| d-OPSD THU (2606.18195) | same model with a random 25% of still-masked positions revealed from the student's own final answer; reverse KL | fixed 2 tok/step, no parallel-decoding results | no |
| dOPSD NUS (2607.04428) | average of teacher predictions over all future steps where i is still masked (path mixture), fwd KL, correct rollouts only | 1 tok/step; >1 "degrades sharply" | no |
| D2F (2508.09192), Multi-Block DLM (2606.29215) | KL to a teacher seeing all noisy blocks; ground-truth-noised inputs; MultiTF randomized noise groups | tau 0.9 | no |
| Learn2PD (2509.25188), Unmasking Policies (2512.09106), TraceLock (2605.24697), dUltra (2512.21446) | external commit policy / stability head on a frozen base (dUltra: base + planner, GRPO) | — | yes |
| TABOM (2605.11854) | Boltzmann ranking: earlier-decoded tokens must have lower entropy | targets random-mask vs confidence-order mismatch | no |
| CLLM (ICML'24), Jacobi Forcing (2512.14681) | AR analogue: own converged fixed point, noisy drafts kept and corrected | — | rejection recycling |

## C. Theory

| Paper | Statement |
|---|---|
| ParallelBench (2510.04767) | one-step min-KL >= conditional total correlation; fine-tuning only fixes zero-TC tasks |
| Cai & Li (2603.22248) | error <= sum of mutual information among co-committed tokens |
| From Bits to Rounds (2511.21103) | rounds >= total information / per-round budget |
| **Consistent Diffusion LMs (2605.00161)** | explicit Chapman-Kolmogorov per position; target E_{x_s ~ p(x_s|x_t)}[g(x_s)_i], single-sample MC, JSD to EMA; **authors: the optimum is still the per-token posterior marginal; "does not overcome the token-factorization barrier"** |
| Gao & Sun (2512.12889) | same marginalization at the full-state level |
| PTP (ICLR'26) | independent prediction is limited even with infinite capacity; fix = auxiliary uniforms (joint via shared noise); K-Forcing (2606.10820) AR analogue |

## D. Soft vs hard tokens as context

| Paper | Construction | Verdict |
|---|---|---|
| Soft-Masked DLM (ICLR'26), EvoToken (ACL'26), RCD (ICML'26), DMax | expected embedding of still-masked positions mixed with e_MASK, entropy-gated | positive at low NFE **when mixed with MASK and gated**; RCD: naive sum fails, top-1 hard feedback degrades (qualitative) |
| Soft Thinking critique (2508.03440), Continuous Mixture Collapse (2609.02049) | sum p e as AR input | **collapses to the top-1 path** ("greedy pitfall"); mixture drifts to one component |
| Soft Tokens, Hard Truths (Meta, 2509) | soft + Gaussian noise in RL | best = train soft, infer hard; soft inference no gain |
| CoCommit | no token values exchanged, only a marker | +1.5-5.5 pts |

## Synthesis (agent's, checked against the tables)

1. Nobody trains the denoiser's marginal toward the sibling-marginalized target sum_sib p(sib|S) q(c|S,sib) with p(sib|S) weights and LOO evaluation. Closest: Consistent DLM (CK target, factorized bridge, no gain beyond the marginal); dOPSD NUS (path mixture over future states); d-OPSD / TIDE / CD4LM / CForce / TSD (single revealed context = implicit one-sample estimator, never stated). The only paper that deliberately moves mass off a premature position is WINO+ (entropy-max on confident-wrong, entropy-min on verified). PRISM learns the LOO quantity as a separate head. No denoiser puts mass on MASK or an abstain head.
2. Impossibility evidence for internalization: ParallelBench (TC lower bound), PTP App. B. These bound the joint error, not whether a defer-aware marginal can avoid committing. Methods keeping an external layer: copula/APD, CoDiLA, DEMASK, Learn2PD, Unmasking Policies, TraceLock, COVER/WINO/PVF/Mean-Field/Measured Coupling, CoCommit. Counter-evidence: Re-evaluating Confidence Remasking (external remasking ~ Fast-dLLM at normal block lengths; "may require changes to training"); COVER: WINO revisions mostly flip-flop.
3. Soft context collapses toward the greedy sibling, so a soft teacher context does NOT approximate a sibling mixture; a sampled hard sibling is a proper one-sample estimator of E_sib[.] when siblings come from the model itself. CoCommit's marker is a third option.

Not found: "SwiftDLM" (the user's local PDF is an ICLR 2027 submission, not on arXiv), CoT2, Gong predictor-corrector; SPEED details; several entries abstract-only; venue labels from READMEs unverified.
