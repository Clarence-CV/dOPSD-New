# PI / intra-step coordination analysis report

generated 2026-10-03T00:08:12, code be724bc

THRESHOLD 0.9, primary = BASE. GSM8K, 300 seeded test problems, T=0.9, analysis re-scored in fp16. best = adapter step-448 (picked by greedy sweep: base 78.3%, step-448 82.7%). Teacher = base weights + PI (fixed teacher), as in training.

## Runs

| run | records | rollouts | accuracy | steps | threshold | temperature | gen_length | block_length | adapter | teacher | pi_samples | teacher_retain_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_thr | /tmp/pipe_gsm/base_thr0.9_t0.9 | 300 | 0.793 | 128 | 0.9 | 0.9 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |
| best_thr | /tmp/pipe_gsm/best_thr0.9_t0.9 | 300 | 0.797 | 128 | 0.9 | 0.9 | 256 | 32 | /scratch/11277/yilunai0430/dopsd/runs/gsm_opsd/adapters/step-448 | fixed | 1 | 0.25 |

Figures 1-3 and 5-8, 10, 11 use the primary run `base_thr`; figure 4 uses every fixed-budget run; figure 9 uses the threshold run.

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
| g | correct | 238 | 50282 | mean | 0.0431946 | [0.0407099, 0.0455301] |
| g | wrong | 62 | 13179 | mean | 0.0522727 | [0.0474255, 0.0570722] |
| frac(p_S<0.5 & g>0.5) | correct | 238 |  | fraction | 0.0143192 | [0.0131422, 0.0155508] |
| frac(p_S<0.5 & g>0.5) | wrong | 62 |  | fraction | 0.0219288 | [0.0188758, 0.0250489] |

### Sibling gap (fig2)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| delta | correct | 238 | 50282 | mean | 0.0133146 | [0.0125588, 0.0139882] |
| delta | wrong | 62 | 13179 | mean | 0.0123815 | [0.0111371, 0.0135773] |
| js_sib | correct | 238 | 50282 | mean | 0.00523361 | [0.00506803, 0.00538598] |
| js_sib | wrong | 62 | 13179 | mean | 0.00495591 | [0.00460263, 0.00530393] |

### Sibling gap vs step (fig3)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| step in [0,8) | correct | 238 |  | mean | 0.00471555 | [0.00434413, 0.00509603] |
| js_sib \| step in [8,16) | correct | 236 |  | mean | 0.0037996 | [0.00334957, 0.00425301] |
| js_sib \| step in [16,24) | correct | 221 |  | mean | 0.00456036 | [0.00390576, 0.00524881] |
| js_sib \| step in [24,32) | correct | 235 |  | mean | 0.00530039 | [0.00479524, 0.00579706] |
| js_sib \| step in [32,40) | correct | 193 |  | mean | 0.00417981 | [0.00365842, 0.00471094] |
| js_sib \| step in [40,48) | correct | 232 |  | mean | 0.0060602 | [0.00540325, 0.006713] |
| js_sib \| step in [48,56) | correct | 214 |  | mean | 0.00508164 | [0.00453293, 0.00562598] |
| js_sib \| step in [56,64) | correct | 191 |  | mean | 0.00524156 | [0.00448384, 0.0059683] |
| js_sib \| step in [64,72) | correct | 210 |  | mean | 0.0056566 | [0.00501595, 0.00637434] |
| js_sib \| step in [72,80) | correct | 182 |  | mean | 0.00515212 | [0.00453392, 0.00571006] |
| js_sib \| step in [80,88) | correct | 195 |  | mean | 0.00521714 | [0.00466782, 0.00584392] |
| js_sib \| step in [88,96) | correct | 188 |  | mean | 0.00546583 | [0.0048787, 0.00608982] |
| js_sib \| step in [96,104) | correct | 176 |  | mean | 0.00564086 | [0.00491892, 0.00635489] |
| js_sib \| step in [104,112) | correct | 174 |  | mean | 0.0056982 | [0.00506425, 0.00638167] |
| js_sib \| step in [112,120) | correct | 148 |  | mean | 0.00626359 | [0.00517038, 0.00753019] |
| js_sib \| step in [120,128) | correct | 74 |  | mean | 0.0049004 | [0.00411179, 0.00569565] |
| js_sib \| step in [128,136) | correct | 36 |  | mean | 0.00638696 | [0.00492474, 0.00780507] |
| js_sib \| step in [136,144) | correct | 12 |  | mean | 0.00283626 | [0.00111734, 0.00484684] |
| js_sib \| step in [144,152) | correct | 1 |  | mean | 1.48295e-06 | [1.48295e-06, 1.48295e-06] |
| js_sib \| step in [0,8) | wrong | 62 |  | mean | 0.00473605 | [0.00391597, 0.00549871] |
| js_sib \| step in [8,16) | wrong | 62 |  | mean | 0.00283371 | [0.00217711, 0.00353267] |
| js_sib \| step in [16,24) | wrong | 55 |  | mean | 0.00391933 | [0.00263777, 0.00524137] |
| js_sib \| step in [24,32) | wrong | 62 |  | mean | 0.00389547 | [0.00312912, 0.00466475] |
| js_sib \| step in [32,40) | wrong | 55 |  | mean | 0.00431402 | [0.00328242, 0.00537867] |
| js_sib \| step in [40,48) | wrong | 61 |  | mean | 0.00466724 | [0.00365221, 0.00576226] |
| js_sib \| step in [48,56) | wrong | 59 |  | mean | 0.00458217 | [0.00345882, 0.00575719] |
| js_sib \| step in [56,64) | wrong | 51 |  | mean | 0.0051445 | [0.00393818, 0.00627185] |
| js_sib \| step in [64,72) | wrong | 58 |  | mean | 0.0063608 | [0.00506684, 0.00786126] |
| js_sib \| step in [72,80) | wrong | 49 |  | mean | 0.00504241 | [0.00392031, 0.00613368] |
| js_sib \| step in [80,88) | wrong | 51 |  | mean | 0.00570345 | [0.00454525, 0.00696778] |
| js_sib \| step in [88,96) | wrong | 50 |  | mean | 0.005224 | [0.00345531, 0.00724481] |
| js_sib \| step in [96,104) | wrong | 47 |  | mean | 0.00645052 | [0.00506371, 0.00786975] |
| js_sib \| step in [104,112) | wrong | 48 |  | mean | 0.00535975 | [0.00418781, 0.00646683] |
| js_sib \| step in [112,120) | wrong | 33 |  | mean | 0.0049879 | [0.00370126, 0.00621387] |
| js_sib \| step in [120,128) | wrong | 31 |  | mean | 0.00506393 | [0.00359089, 0.00649443] |
| js_sib \| step in [128,136) | wrong | 22 |  | mean | 0.00464545 | [0.00340415, 0.00578629] |
| js_sib \| step in [136,144) | wrong | 6 |  | mean | 0.0043712 | [0.00236075, 0.00647449] |
| js_sib \| step in [144,152) | wrong | 1 |  | mean | 0.00178748 | [0.00178748, 0.00178748] |
| delta \| step in [0,8) | correct | 238 |  | mean | 0.0135727 | [0.0124025, 0.0147227] |
| delta \| step in [8,16) | correct | 236 |  | mean | 0.0104524 | [0.00901182, 0.0119133] |
| delta \| step in [16,24) | correct | 221 |  | mean | 0.0131341 | [0.0111612, 0.0151288] |
| delta \| step in [24,32) | correct | 235 |  | mean | 0.0133754 | [0.0118125, 0.0149069] |
| delta \| step in [32,40) | correct | 193 |  | mean | 0.0119147 | [0.0102659, 0.0135325] |
| delta \| step in [40,48) | correct | 232 |  | mean | 0.0152728 | [0.0109075, 0.0184222] |
| delta \| step in [48,56) | correct | 214 |  | mean | 0.0132057 | [0.01146, 0.0148814] |
| delta \| step in [56,64) | correct | 191 |  | mean | 0.014551 | [0.012425, 0.016723] |
| delta \| step in [64,72) | correct | 210 |  | mean | 0.0135186 | [0.00981858, 0.0162179] |
| delta \| step in [72,80) | correct | 182 |  | mean | 0.0138313 | [0.0119091, 0.0156553] |
| delta \| step in [80,88) | correct | 195 |  | mean | 0.0140961 | [0.0123444, 0.0159365] |
| delta \| step in [88,96) | correct | 188 |  | mean | 0.0137911 | [0.0117634, 0.0158326] |
| delta \| step in [96,104) | correct | 176 |  | mean | 0.0122311 | [0.00738962, 0.015457] |
| delta \| step in [104,112) | correct | 174 |  | mean | 0.0145785 | [0.012342, 0.0167326] |
| delta \| step in [112,120) | correct | 148 |  | mean | 0.00978764 | [0.00186747, 0.0154201] |
| delta \| step in [120,128) | correct | 74 |  | mean | 0.0129058 | [0.0108221, 0.015053] |
| delta \| step in [128,136) | correct | 36 |  | mean | 0.0184432 | [0.013719, 0.0228445] |
| delta \| step in [136,144) | correct | 12 |  | mean | 0.00760422 | [0.00266774, 0.0133497] |
| delta \| step in [144,152) | correct | 1 |  | mean | -0.00797087 | [-0.00797087, -0.00797087] |
| delta \| step in [0,8) | wrong | 62 |  | mean | 0.0141312 | [0.0116448, 0.0164491] |
| delta \| step in [8,16) | wrong | 62 |  | mean | 0.00772392 | [0.00498152, 0.0102032] |
| delta \| step in [16,24) | wrong | 55 |  | mean | 0.0107549 | [0.00688288, 0.0149466] |
| delta \| step in [24,32) | wrong | 62 |  | mean | 0.0101584 | [0.00811874, 0.0123353] |
| delta \| step in [32,40) | wrong | 55 |  | mean | 0.0123896 | [0.00930382, 0.0155764] |
| delta \| step in [40,48) | wrong | 61 |  | mean | 0.0106312 | [0.00524068, 0.0144998] |
| delta \| step in [48,56) | wrong | 59 |  | mean | 0.0112863 | [0.00687906, 0.0148192] |
| delta \| step in [56,64) | wrong | 51 |  | mean | 0.0141711 | [0.010465, 0.0175428] |
| delta \| step in [64,72) | wrong | 58 |  | mean | 0.0134528 | [0.0036838, 0.0200196] |
| delta \| step in [72,80) | wrong | 49 |  | mean | 0.0137281 | [0.0106262, 0.0168013] |
| delta \| step in [80,88) | wrong | 51 |  | mean | 0.013957 | [0.0100251, 0.017936] |
| delta \| step in [88,96) | wrong | 50 |  | mean | 0.00791689 | [-0.00364234, 0.0155075] |
| delta \| step in [96,104) | wrong | 47 |  | mean | 0.0154412 | [0.00981188, 0.0198815] |
| delta \| step in [104,112) | wrong | 48 |  | mean | 0.0156505 | [0.012139, 0.0190309] |
| delta \| step in [112,120) | wrong | 33 |  | mean | 0.0134994 | [0.00935216, 0.0173323] |
| delta \| step in [120,128) | wrong | 31 |  | mean | 0.0122542 | [0.00834389, 0.0160433] |
| delta \| step in [128,136) | wrong | 22 |  | mean | 0.0101689 | [0.00676143, 0.0138602] |
| delta \| step in [136,144) | wrong | 6 |  | mean | 0.0140274 | [0.00766166, 0.019636] |
| delta \| step in [144,152) | wrong | 1 |  | mean | 0.00527722 | [0.00527722, 0.00527722] |

### Sibling gap vs |C_t| (fixed budgets) (fig4)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib | correct | 238 | base_thr \|C_t\|=6 | mean | 0.00523361 | [0.00506803, 0.00538598] |
| js_sib | correct | 239 | best_thr \|C_t\|=12 | mean | 0.00594931 | [0.00565433, 0.00625134] |
| js_sib | wrong | 62 | base_thr \|C_t\|=5 | mean | 0.00495591 | [0.00460263, 0.00530393] |
| js_sib | wrong | 61 | best_thr \|C_t\|=9 | mean | 0.00735894 | [0.00674484, 0.00800878] |
| delta | correct | 238 | base_thr \|C_t\|=6 | mean | 0.0133146 | [0.0125588, 0.0139882] |
| delta | correct | 239 | best_thr \|C_t\|=12 | mean | -0.000877399 | [-0.00446998, 0.00237241] |
| delta | wrong | 62 | base_thr \|C_t\|=5 | mean | 0.0123815 | [0.0111371, 0.0135773] |
| delta | wrong | 61 | best_thr \|C_t\|=9 | mean | -0.0117796 | [-0.0228897, -0.00278191] |

### PI shift vs sibling shift (fig5)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| cos_pi_sib | correct | 238 | 50282 | mean | 0.595037 | [0.58145, 0.60892] |
| cos_pi_sib | wrong | 62 | 13179 | mean | 0.528398 | [0.493953, 0.561694] |
| frac(cos>0) | correct | 238 |  | fraction | 0.771827 | [0.764125, 0.779512] |
| frac(cos>0) | wrong | 62 |  | fraction | 0.741483 | [0.72303, 0.759428] |

### B. Coordination gap (fig6)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S | correct | 238 | ctrl_ok tokens | mean | 0.0120782 | [0.0111275, 0.0129149] |
| D_T | correct | 238 | ctrl_ok tokens | mean | 0.00500573 | [0.0027847, 0.00749603] |
| D_S_ctrl | correct | 238 | ctrl_ok tokens | mean | 0.0147344 | [0.0140169, 0.015397] |
| D_T_ctrl | correct | 238 | ctrl_ok tokens | mean | 0.00627879 | [0.00463462, 0.00863906] |
| D_S | wrong | 62 | ctrl_ok tokens | mean | 0.0107307 | [0.00937227, 0.012017] |
| D_T | wrong | 62 | ctrl_ok tokens | mean | 0.00514462 | [0.00408525, 0.00627079] |
| D_S_ctrl | wrong | 62 | ctrl_ok tokens | mean | 0.0131499 | [0.0113105, 0.0145998] |
| D_T_ctrl | wrong | 62 | ctrl_ok tokens | mean | 0.00608299 | [0.00487063, 0.00736198] |
| coord | correct | 238 | ctrl_ok tokens | mean | 0.00707243 | [0.00457556, 0.00915253] |
| coord_ctrl | correct | 238 | ctrl_ok tokens | mean | 0.00845558 | [0.0060692, 0.0101082] |
| coord | wrong | 62 | ctrl_ok tokens | mean | 0.00558611 | [0.00380786, 0.00730707] |
| coord_ctrl | wrong | 62 | ctrl_ok tokens | mean | 0.00706693 | [0.0047949, 0.00902198] |

### C. Sibling vs matched control (paired) (fig7)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S - D_S_ctrl (paired) | correct | 238 | ctrl_ok tokens | mean | -0.00265621 | [-0.00344205, -0.00189611] |
| D_S - D_S_ctrl (paired) | wrong | 62 | ctrl_ok tokens | mean | -0.00241919 | [-0.00399014, -0.000317853] |
| coord - coord_ctrl (paired) | correct | 238 | ctrl_ok tokens | mean | -0.00138316 | [-0.00237585, -6.58778e-05] |
| coord - coord_ctrl (paired) | wrong | 62 | ctrl_ok tokens | mean | -0.00148082 | [-0.00318979, 0.000597165] |

### A. Pairwise co-gain (distance-matched) (fig8)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| pair co-gain P(both g>0) co-decoded | correct | 238 | 173085 co-decoded pairs with a distance match | fraction | 0.916111 |  |
| pair co-gain distance-matched null | correct | 238 | perm p=0.0020 | fraction | 0.837994 | [0.836597, 0.839304] |
| pair co-gain real - matched expectation | correct | 238 | rollout bootstrap CI | difference | 0.0780685 | [0.0652944, 0.0918055] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | correct | 238 | real=0.8216 | fraction | 0.719748 | [0.7142, 0.725311] |
| pair co-gain P(both g>0) co-decoded | wrong | 62 | 38957 co-decoded pairs with a distance match | fraction | 0.910311 |  |
| pair co-gain distance-matched null | wrong | 62 | perm p=0.0020 | fraction | 0.863504 | [0.860601, 0.866251] |
| pair co-gain real - matched expectation | wrong | 62 | rollout bootstrap CI | difference | 0.0467107 | [0.03159, 0.0630506] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | wrong | 62 | real=0.8046 | fraction | 0.686996 | [0.674312, 0.699368] |

### Sibling gap vs natural |C_t| (threshold) (fig9)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| \|C_t\| in 2 | correct | 237 | base_thr | mean | 0.00567308 | [0.00495138, 0.00661316] |
| js_sib \| \|C_t\| in 3-4 | correct | 237 | base_thr | mean | 0.00652526 | [0.00619739, 0.00687036] |
| js_sib \| \|C_t\| in 5-8 | correct | 237 | base_thr | mean | 0.0072462 | [0.00688231, 0.0076391] |
| js_sib \| \|C_t\| in 9+ | correct | 237 | base_thr | mean | 0.00657606 | [0.00621352, 0.00692379] |
| js_sib \| \|C_t\| in 2 | wrong | 61 | base_thr | mean | 0.00501883 | [0.00427375, 0.00590205] |
| js_sib \| \|C_t\| in 3-4 | wrong | 62 | base_thr | mean | 0.00674457 | [0.00575479, 0.00783291] |
| js_sib \| \|C_t\| in 5-8 | wrong | 62 | base_thr | mean | 0.00738254 | [0.00674234, 0.00809042] |
| js_sib \| \|C_t\| in 9+ | wrong | 61 | base_thr | mean | 0.00685673 | [0.00619365, 0.00753537] |
| D_S \| \|C_t\| in 2 | correct | 237 | base_thr | mean | 0.00922543 | [0.00328034, 0.0137537] |
| D_S \| \|C_t\| in 3-4 | correct | 237 | base_thr | mean | 0.0171915 | [0.0161172, 0.0182582] |
| D_S \| \|C_t\| in 5-8 | correct | 237 | base_thr | mean | 0.0174868 | [0.0151484, 0.0193805] |
| D_S \| \|C_t\| in 9+ | correct | 237 | base_thr | mean | 0.0180171 | [0.0169998, 0.0190247] |
| D_S \| \|C_t\| in 2 | wrong | 61 | base_thr | mean | 0.00965913 | [0.00581217, 0.0129899] |
| D_S \| \|C_t\| in 3-4 | wrong | 62 | base_thr | mean | 0.0134772 | [0.00761193, 0.0181539] |
| D_S \| \|C_t\| in 5-8 | wrong | 62 | base_thr | mean | 0.0193101 | [0.0171472, 0.0214991] |
| D_S \| \|C_t\| in 9+ | wrong | 61 | base_thr | mean | 0.0188696 | [0.016698, 0.0209836] |

### PI effect types (fig10)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| sharpen | correct | 238 |  | mean | 0.919713 | [0.913982, 0.925238] |
| defer | correct | 238 |  | mean | 0.06915 | [0.0641892, 0.0743577] |
| redirect | correct | 238 |  | mean | 0.0111372 | [0.0100327, 0.0123433] |
| sharpen | wrong | 62 |  | mean | 0.90151 | [0.88969, 0.913433] |
| defer | wrong | 62 |  | mean | 0.084149 | [0.0737282, 0.0946718] |
| redirect | wrong | 62 |  | mean | 0.014341 | [0.0117046, 0.0172441] |
| defer\|redirect D_S<0 (conflict) | correct | 238 |  | mean | 0.117802 | [0.104813, 0.130996] |
| defer\|redirect D_S>=0 | correct | 238 |  | mean | 0.0605912 | [0.0546258, 0.0666452] |
| defer\|redirect D_S<0 (conflict) | wrong | 62 |  | mean | 0.125828 | [0.101166, 0.151643] |
| defer\|redirect D_S>=0 | wrong | 62 |  | mean | 0.0730691 | [0.0612213, 0.0851689] |

### Decoding order (fig11)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| overlap | correct | 238 |  | mean | 0.505921 | [0.497646, 0.514053] |
| overlap | wrong | 62 |  | mean | 0.498619 | [0.483209, 0.514292] |
| all | correct | 238 |  | mean | 0.268187 | [0.260289, 0.27616] |
| D_S<0 | correct | 238 |  | mean | 0.285284 | [0.267149, 0.302733] |
| D_S>=0 | correct | 238 |  | mean | 0.273177 | [0.264491, 0.28217] |
| all | wrong | 62 |  | mean | 0.241976 | [0.223439, 0.260723] |
| D_S<0 | wrong | 62 |  | mean | 0.246358 | [0.212926, 0.280961] |
| D_S>=0 | wrong | 62 |  | mean | 0.250056 | [0.229563, 0.26971] |
| spearman | correct | 238 |  | mean | 0.222227 | [0.209492, 0.234546] |
| spearman | wrong | 62 |  | mean | 0.252168 | [0.222196, 0.280361] |

## Files

- `summary.csv`: every number above
- `tokens.csv.gz`: one row per revealed token (all runs)
- `steps.csv.gz`: one row per decoding step
- `rollouts.csv`: one row per rollout with question/gold/prediction/completion
- `fig*.png`: figures
