# PI / intra-step coordination analysis report

generated 2026-10-03T00:07:37, code be724bc

THRESHOLD 0.9, primary = trained step-448. GSM8K, 300 seeded test problems, T=0.9, analysis re-scored in fp16. best = adapter step-448 (picked by greedy sweep: base 78.3%, step-448 82.7%). Teacher = base weights + PI (fixed teacher), as in training.

## Runs

| run | records | rollouts | accuracy | steps | threshold | temperature | gen_length | block_length | adapter | teacher | pi_samples | teacher_retain_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_thr | /tmp/pipe_gsm/best_thr0.9_t0.9 | 300 | 0.797 | 128 | 0.9 | 0.9 | 256 | 32 | /scratch/11277/yilunai0430/dopsd/runs/gsm_opsd/adapters/step-448 | fixed | 1 | 0.25 |
| base_thr | /tmp/pipe_gsm/base_thr0.9_t0.9 | 300 | 0.793 | 128 | 0.9 | 0.9 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |

Figures 1-3 and 5-8, 10, 11 use the primary run `best_thr`; figure 4 uses every fixed-budget run; figure 9 uses the threshold run.

## Definitions

All quantities are computed per revealed token i at the step t where it was committed (C_t = tokens
revealed at step t, y_i = its value in the final rollout). Tokens of the last block (no future, so no
PI) are not analysed. Unless stated otherwise, summaries exclude special tokens (EOS/EOT) and tokens
after the answer end. CIs are 95% cluster bootstrap over rollouts.

| name | definition | reading |
|---|---|---|
| g | log p_T(y_i \| S_t, PI) - log p_S(y_i \| S_t) | PI alignment with the rollout's own token; a gain only on correct rollouts |
| D_S | log p_S(y_i \| S_t, y[C_t\\i]) - log p_S(y_i \| S_t) | what the co-decoded siblings add for the student |
| D_T | log p_T(y_i \| S_t, PI, y[C_t\\i]) - log p_T(y_i \| S_t, PI) | what siblings still add once PI is known |
| coord | D_S - D_T | part of the sibling information PI already provides (hypothesis: > 0 on correct rollouts) |
| *_ctrl | same with a matched non-sibling control (same block, revealed later, matched on distance, student confidence and reveal delay) | sibling - control > 0 means same-step siblings are special |
| js_sib / js_pi | JS(p_S, p_sib) / JS(p_S, p_T), full vocabulary, nats | size of the distribution shift |
| cos_pi_sib | cos(p_T - p_S, p_sib - p_S) over the vocabulary | do the PI shift and the sibling shift agree |
| pair co-gain | P(g_i > 0 and g_j > 0) for co-decoded pairs vs distance-matched pairs revealed at different steps | PI lifts co-decoded tokens together beyond position effects |
| sharpen / defer / redirect | teacher top-1 = y_i and g > 0 / teacher top-1 = y_i and g <= 0 / teacher top-1 != y_i | what PI does to the token; defer can be useful (hold the slot back) |
| rank_S / rank_T | rank of i among all masked slots of the block by top-1 prob (1 = most confident) | decoding-order preference |
| rank_T_norm | (rank_T - 1) / (#masked in block - 1) | > 0.5: teacher would commit it late (premature commit) |
| order_overlap | \|C_t & teacher's top-\|C_t\|\| / \|C_t\| | would the teacher reveal the same slots now |
| order_spearman | Spearman(conf_S, conf_T) over the block's masked slots | overall order agreement |

## Results

### PI gain (fig1)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| g | correct | 239 | 48070 | mean | -0.0176289 | [-0.0246952, -0.0116006] |
| g | wrong | 61 | 12836 | mean | -0.0258019 | [-0.0352762, -0.0174028] |
| frac(p_S<0.5 & g>0.5) | correct | 239 |  | fraction | 0.00264198 | [0.0021887, 0.00312164] |
| frac(p_S<0.5 & g>0.5) | wrong | 61 |  | fraction | 0.0039732 | [0.00297164, 0.00507782] |

### Sibling gap (fig2)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| delta | correct | 239 | 48070 | mean | -0.000877399 | [-0.00446998, 0.00237241] |
| delta | wrong | 61 | 12836 | mean | -0.0117796 | [-0.0228897, -0.00278191] |
| js_sib | correct | 239 | 48070 | mean | 0.00594931 | [0.00565433, 0.00625134] |
| js_sib | wrong | 61 | 12836 | mean | 0.00735894 | [0.00674484, 0.00800878] |

### Sibling gap vs step (fig3)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| step in [0,8) | correct | 239 |  | mean | 0.00730052 | [0.00658629, 0.00803828] |
| js_sib \| step in [8,16) | correct | 197 |  | mean | 0.00540851 | [0.00465677, 0.00617636] |
| js_sib \| step in [16,24) | correct | 232 |  | mean | 0.00685345 | [0.00596784, 0.00778536] |
| js_sib \| step in [24,32) | correct | 211 |  | mean | 0.00606987 | [0.00526672, 0.0070028] |
| js_sib \| step in [32,40) | correct | 210 |  | mean | 0.00459381 | [0.00410397, 0.00512048] |
| js_sib \| step in [40,48) | correct | 226 |  | mean | 0.00549659 | [0.00470561, 0.00636378] |
| js_sib \| step in [48,56) | correct | 202 |  | mean | 0.00598366 | [0.00477952, 0.00734429] |
| js_sib \| step in [56,64) | correct | 187 |  | mean | 0.00462982 | [0.00391521, 0.00550156] |
| js_sib \| step in [64,72) | correct | 144 |  | mean | 0.00668885 | [0.00536241, 0.00822661] |
| js_sib \| step in [72,80) | correct | 123 |  | mean | 0.00644077 | [0.00502746, 0.00811328] |
| js_sib \| step in [80,88) | correct | 48 |  | mean | 0.00492358 | [0.00338023, 0.00671437] |
| js_sib \| step in [88,96) | correct | 12 |  | mean | 0.00742879 | [0.00375614, 0.0105338] |
| js_sib \| step in [96,104) | correct | 4 |  | mean | 0.0151261 | [0.00444165, 0.0298824] |
| js_sib \| step in [104,112) | correct | 4 |  | mean | 0.00157921 | [0.000641921, 0.00240984] |
| js_sib \| step in [112,120) | correct | 0 |  | mean |  |  |
| js_sib \| step in [0,8) | wrong | 61 |  | mean | 0.00735346 | [0.00607739, 0.00883762] |
| js_sib \| step in [8,16) | wrong | 50 |  | mean | 0.00540542 | [0.00419307, 0.00657326] |
| js_sib \| step in [16,24) | wrong | 58 |  | mean | 0.00749861 | [0.00596396, 0.0092494] |
| js_sib \| step in [24,32) | wrong | 58 |  | mean | 0.00915109 | [0.00591677, 0.0131082] |
| js_sib \| step in [32,40) | wrong | 56 |  | mean | 0.00590697 | [0.00478918, 0.00712023] |
| js_sib \| step in [40,48) | wrong | 60 |  | mean | 0.00556737 | [0.00418046, 0.00714592] |
| js_sib \| step in [48,56) | wrong | 55 |  | mean | 0.00701377 | [0.00534913, 0.00874295] |
| js_sib \| step in [56,64) | wrong | 54 |  | mean | 0.00925049 | [0.00657542, 0.0120985] |
| js_sib \| step in [64,72) | wrong | 43 |  | mean | 0.00978507 | [0.00578562, 0.0143105] |
| js_sib \| step in [72,80) | wrong | 40 |  | mean | 0.00794818 | [0.00486104, 0.0120882] |
| js_sib \| step in [80,88) | wrong | 22 |  | mean | 0.00787945 | [0.00389655, 0.0125375] |
| js_sib \| step in [88,96) | wrong | 6 |  | mean | 0.00401741 | [0.00149401, 0.00622495] |
| js_sib \| step in [96,104) | wrong | 5 |  | mean | 0.00211114 | [0.00111382, 0.00356996] |
| js_sib \| step in [104,112) | wrong | 1 |  | mean | 0.00477844 | [0.00477844, 0.00477844] |
| js_sib \| step in [112,120) | wrong | 1 |  | mean | 8.05434e-06 | [8.05434e-06, 8.05434e-06] |
| delta \| step in [0,8) | correct | 239 |  | mean | 0.00530685 | [-0.000774693, 0.0100427] |
| delta \| step in [8,16) | correct | 197 |  | mean | 0.00274669 | [-0.00505664, 0.00842887] |
| delta \| step in [16,24) | correct | 232 |  | mean | -0.000603953 | [-0.00940062, 0.00640131] |
| delta \| step in [24,32) | correct | 211 |  | mean | -0.00551232 | [-0.0216142, 0.00635227] |
| delta \| step in [32,40) | correct | 210 |  | mean | 0.00864616 | [0.0071876, 0.0101799] |
| delta \| step in [40,48) | correct | 226 |  | mean | 0.000550265 | [-0.00651788, 0.00622095] |
| delta \| step in [48,56) | correct | 202 |  | mean | -0.0103654 | [-0.0277641, 0.00412555] |
| delta \| step in [56,64) | correct | 187 |  | mean | 0.00214275 | [-0.00871827, 0.00892467] |
| delta \| step in [64,72) | correct | 144 |  | mean | -0.0114967 | [-0.0334346, 0.00486609] |
| delta \| step in [72,80) | correct | 123 |  | mean | -0.00759146 | [-0.0241306, 0.00549769] |
| delta \| step in [80,88) | correct | 48 |  | mean | 0.00382436 | [-0.00506519, 0.010105] |
| delta \| step in [88,96) | correct | 12 |  | mean | 0.00598647 | [-0.00647581, 0.0200047] |
| delta \| step in [96,104) | correct | 4 |  | mean | -0.0489713 | [-0.150858, 0.0185854] |
| delta \| step in [104,112) | correct | 4 |  | mean | -0.000844585 | [-0.0251056, 0.00457073] |
| delta \| step in [112,120) | correct | 0 |  | mean |  |  |
| delta \| step in [0,8) | wrong | 61 |  | mean | 0.00525836 | [-0.00180188, 0.0111548] |
| delta \| step in [8,16) | wrong | 50 |  | mean | 0.00871422 | [0.00547378, 0.0118622] |
| delta \| step in [16,24) | wrong | 58 |  | mean | -0.00428987 | [-0.0225541, 0.00807157] |
| delta \| step in [24,32) | wrong | 58 |  | mean | -0.0235792 | [-0.0685729, 0.00808159] |
| delta \| step in [32,40) | wrong | 56 |  | mean | 0.0106556 | [0.00691105, 0.0138771] |
| delta \| step in [40,48) | wrong | 60 |  | mean | 0.00379413 | [-0.00376331, 0.0101415] |
| delta \| step in [48,56) | wrong | 55 |  | mean | -0.0113025 | [-0.0364659, 0.00923808] |
| delta \| step in [56,64) | wrong | 54 |  | mean | -0.0259402 | [-0.0742814, 0.00525991] |
| delta \| step in [64,72) | wrong | 43 |  | mean | -0.0643034 | [-0.185634, 0.00302185] |
| delta \| step in [72,80) | wrong | 40 |  | mean | -0.0284182 | [-0.0894028, 0.0113782] |
| delta \| step in [80,88) | wrong | 22 |  | mean | -0.0524553 | [-0.162057, 0.00703756] |
| delta \| step in [88,96) | wrong | 6 |  | mean | 0.0121366 | [0.00440904, 0.019746] |
| delta \| step in [96,104) | wrong | 5 |  | mean | 0.00631536 | [0.00277076, 0.0106995] |
| delta \| step in [104,112) | wrong | 1 |  | mean | -0.00815271 | [-0.00815271, -0.00815271] |
| delta \| step in [112,120) | wrong | 1 |  | mean | -0.00549978 | [-0.00549978, -0.00549978] |

### Sibling gap vs |C_t| (fixed budgets) (fig4)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib | correct | 239 | best_thr \|C_t\|=12 | mean | 0.00594931 | [0.00565433, 0.00625134] |
| js_sib | correct | 238 | base_thr \|C_t\|=6 | mean | 0.00523361 | [0.00506803, 0.00538598] |
| js_sib | wrong | 61 | best_thr \|C_t\|=9 | mean | 0.00735894 | [0.00674484, 0.00800878] |
| js_sib | wrong | 62 | base_thr \|C_t\|=5 | mean | 0.00495591 | [0.00460263, 0.00530393] |
| delta | correct | 239 | best_thr \|C_t\|=12 | mean | -0.000877399 | [-0.00446998, 0.00237241] |
| delta | correct | 238 | base_thr \|C_t\|=6 | mean | 0.0133146 | [0.0125588, 0.0139882] |
| delta | wrong | 61 | best_thr \|C_t\|=9 | mean | -0.0117796 | [-0.0228897, -0.00278191] |
| delta | wrong | 62 | base_thr \|C_t\|=5 | mean | 0.0123815 | [0.0111371, 0.0135773] |

### PI shift vs sibling shift (fig5)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| cos_pi_sib | correct | 239 | 48070 | mean | 0.156366 | [0.134646, 0.178434] |
| cos_pi_sib | wrong | 61 | 12836 | mean | 0.102253 | [0.0586756, 0.144455] |
| frac(cos>0) | correct | 239 |  | fraction | 0.494113 | [0.482653, 0.505689] |
| frac(cos>0) | wrong | 61 |  | fraction | 0.479433 | [0.456847, 0.501491] |

### B. Coordination gap (fig6)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S | correct | 239 | ctrl_ok tokens | mean | -0.00917676 | [-0.0169869, -0.00208861] |
| D_T | correct | 239 | ctrl_ok tokens | mean | 0.0240917 | [0.0167167, 0.0315622] |
| D_S_ctrl | correct | 239 | ctrl_ok tokens | mean | 0.0063926 | [0.00119895, 0.0107717] |
| D_T_ctrl | correct | 239 | ctrl_ok tokens | mean | 0.031483 | [0.0245182, 0.0385823] |
| D_S | wrong | 61 | ctrl_ok tokens | mean | -0.0135155 | [-0.0273778, -0.00203962] |
| D_T | wrong | 61 | ctrl_ok tokens | mean | 0.0265955 | [0.0149338, 0.038022] |
| D_S_ctrl | wrong | 61 | ctrl_ok tokens | mean | 0.000809166 | [-0.0086088, 0.00925436] |
| D_T_ctrl | wrong | 61 | ctrl_ok tokens | mean | 0.0361944 | [0.0270317, 0.0458146] |
| coord | correct | 239 | ctrl_ok tokens | mean | -0.0332684 | [-0.0408413, -0.0257088] |
| coord_ctrl | correct | 239 | ctrl_ok tokens | mean | -0.0250904 | [-0.0327479, -0.017839] |
| coord | wrong | 61 | ctrl_ok tokens | mean | -0.040111 | [-0.0529196, -0.0283654] |
| coord_ctrl | wrong | 61 | ctrl_ok tokens | mean | -0.0353853 | [-0.0469716, -0.024875] |

### C. Sibling vs matched control (paired) (fig7)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S - D_S_ctrl (paired) | correct | 239 | ctrl_ok tokens | mean | -0.0155694 | [-0.0212165, -0.010522] |
| D_S - D_S_ctrl (paired) | wrong | 61 | ctrl_ok tokens | mean | -0.0143246 | [-0.0257505, -0.0054631] |
| coord - coord_ctrl (paired) | correct | 239 | ctrl_ok tokens | mean | -0.008178 | [-0.013075, -0.00385777] |
| coord - coord_ctrl (paired) | wrong | 61 | ctrl_ok tokens | mean | -0.00472569 | [-0.0102741, 0.000562714] |

### A. Pairwise co-gain (distance-matched) (fig8)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| pair co-gain P(both g>0) co-decoded | correct | 239 | 253299 co-decoded pairs with a distance match | fraction | 0.368213 |  |
| pair co-gain distance-matched null | correct | 239 | perm p=0.0020 | fraction | 0.309596 | [0.308236, 0.310878] |
| pair co-gain real - matched expectation | correct | 239 | rollout bootstrap CI | difference | 0.0586103 | [0.0424824, 0.0756792] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | correct | 239 | real=0.1969 | fraction | 0.11531 | [0.109221, 0.121551] |
| pair co-gain P(both g>0) co-decoded | wrong | 61 | 59492 co-decoded pairs with a distance match | fraction | 0.336499 |  |
| pair co-gain distance-matched null | wrong | 61 | perm p=0.0020 | fraction | 0.287899 | [0.285147, 0.290636] |
| pair co-gain real - matched expectation | wrong | 61 | rollout bootstrap CI | difference | 0.0485323 | [0.0222882, 0.0756376] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | wrong | 61 | real=0.2035 | fraction | 0.109865 | [0.0981707, 0.1181] |

### Sibling gap vs natural |C_t| (threshold) (fig9)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| \|C_t\| in 2 | correct | 224 | best_thr | mean | 0.00919485 | [0.00704773, 0.0119694] |
| js_sib \| \|C_t\| in 3-4 | correct | 235 | best_thr | mean | 0.0100465 | [0.00868667, 0.0115886] |
| js_sib \| \|C_t\| in 5-8 | correct | 238 | best_thr | mean | 0.00809758 | [0.00735727, 0.00894708] |
| js_sib \| \|C_t\| in 9+ | correct | 239 | best_thr | mean | 0.00545509 | [0.0051303, 0.00579801] |
| js_sib \| \|C_t\| in 2 | wrong | 59 | best_thr | mean | 0.00597561 | [0.00479867, 0.0071531] |
| js_sib \| \|C_t\| in 3-4 | wrong | 61 | best_thr | mean | 0.0114639 | [0.00878767, 0.0148179] |
| js_sib \| \|C_t\| in 5-8 | wrong | 61 | best_thr | mean | 0.0099764 | [0.00847808, 0.0116571] |
| js_sib \| \|C_t\| in 9+ | wrong | 61 | best_thr | mean | 0.00733988 | [0.00644153, 0.00839636] |
| D_S \| \|C_t\| in 2 | correct | 224 | best_thr | mean | -0.0185193 | [-0.0394438, -0.00269813] |
| D_S \| \|C_t\| in 3-4 | correct | 235 | best_thr | mean | -0.0279444 | [-0.0541223, -0.00642176] |
| D_S \| \|C_t\| in 5-8 | correct | 238 | best_thr | mean | -0.00151827 | [-0.013158, 0.00748795] |
| D_S \| \|C_t\| in 9+ | correct | 239 | best_thr | mean | 0.00395583 | [0.000939489, 0.00659359] |
| D_S \| \|C_t\| in 2 | wrong | 59 | best_thr | mean | 0.00934774 | [0.00482431, 0.0140487] |
| D_S \| \|C_t\| in 3-4 | wrong | 61 | best_thr | mean | -0.0273313 | [-0.0820214, 0.0109986] |
| D_S \| \|C_t\| in 5-8 | wrong | 61 | best_thr | mean | -0.00543655 | [-0.0212083, 0.0069027] |
| D_S \| \|C_t\| in 9+ | wrong | 61 | best_thr | mean | -0.0152529 | [-0.0328031, -0.00174279] |

### PI effect types (fig10)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| sharpen | correct | 239 |  | mean | 0.513647 | [0.500936, 0.526818] |
| defer | correct | 239 |  | mean | 0.472873 | [0.460092, 0.48563] |
| redirect | correct | 239 |  | mean | 0.0134803 | [0.0115757, 0.015441] |
| sharpen | wrong | 61 |  | mean | 0.488158 | [0.45954, 0.51569] |
| defer | wrong | 61 |  | mean | 0.496416 | [0.468874, 0.524312] |
| redirect | wrong | 61 |  | mean | 0.0154254 | [0.0124135, 0.0185677] |
| defer\|redirect D_S<0 (conflict) | correct | 239 |  | mean | 0.595338 | [0.573837, 0.614912] |
| defer\|redirect D_S>=0 | correct | 239 |  | mean | 0.488806 | [0.474586, 0.502966] |
| defer\|redirect D_S<0 (conflict) | wrong | 61 |  | mean | 0.649312 | [0.613485, 0.687503] |
| defer\|redirect D_S>=0 | wrong | 61 |  | mean | 0.515643 | [0.485083, 0.54693] |

### Decoding order (fig11)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| overlap | correct | 239 |  | mean | 0.587364 | [0.578744, 0.596556] |
| overlap | wrong | 61 |  | mean | 0.564986 | [0.550747, 0.579155] |
| all | correct | 239 |  | mean | 0.34714 | [0.338989, 0.355559] |
| D_S<0 | correct | 239 |  | mean | 0.399283 | [0.382856, 0.416749] |
| D_S>=0 | correct | 239 |  | mean | 0.352554 | [0.344501, 0.360875] |
| all | wrong | 61 |  | mean | 0.311078 | [0.290522, 0.330342] |
| D_S<0 | wrong | 61 |  | mean | 0.366405 | [0.328589, 0.400429] |
| D_S>=0 | wrong | 61 |  | mean | 0.3185 | [0.297319, 0.339316] |
| spearman | correct | 239 |  | mean | 0.280615 | [0.266623, 0.295016] |
| spearman | wrong | 61 |  | mean | 0.280569 | [0.255602, 0.304119] |

## Files

- `summary.csv`: every number above
- `tokens.csv.gz`: one row per revealed token (all runs)
- `steps.csv.gz`: one row per decoding step
- `rollouts.csv`: one row per rollout with question/gold/prediction/completion
- `fig*.png`: figures
