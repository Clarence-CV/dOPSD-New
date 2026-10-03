# PI / intra-step coordination analysis report

generated 2026-10-02T23:58:11, code be724bc

FIXED BUDGET (128 steps), primary = BASE. GSM8K, 300 seeded test problems, T=0.9, analysis re-scored in fp16. best = adapter step-448 (picked by greedy sweep: base 78.3%, step-448 82.7%). Teacher = base weights + PI (fixed teacher), as in training.

## Runs

| run | records | rollouts | accuracy | steps | threshold | temperature | gen_length | block_length | adapter | teacher | pi_samples | teacher_retain_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_s128 | /tmp/pipe_gsm/base_s128_t0.9 | 300 | 0.750 | 128 | none (fixed budget) | 0.9 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |
| best_s128 | /tmp/pipe_gsm/best_s128_t0.9 | 300 | 0.820 | 128 | none (fixed budget) | 0.9 | 256 | 32 | /scratch/11277/yilunai0430/dopsd/runs/gsm_opsd/adapters/step-448 | fixed | 1 | 0.25 |

Figures 1-3 and 5-8, 10, 11 use the primary run `base_s128`; figure 4 uses every fixed-budget run; figure 9 uses the threshold run.

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
| g | correct | 225 | 47498 | mean | 0.0207179 | [0.0157918, 0.0254091] |
| g | wrong | 75 | 16218 | mean | 0.0274915 | [0.0186429, 0.0363804] |
| frac(p_S<0.5 & g>0.5) | correct | 225 |  | fraction | 0.0272011 | [0.0251676, 0.0291755] |
| frac(p_S<0.5 & g>0.5) | wrong | 75 |  | fraction | 0.0400173 | [0.0356062, 0.0443125] |

### Sibling gap (fig2)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| delta | correct | 225 | 47498 | mean | -0.129502 | [-0.146443, -0.113495] |
| delta | wrong | 75 | 16218 | mean | -0.163062 | [-0.191035, -0.135272] |
| js_sib | correct | 225 | 47498 | mean | 0.0199505 | [0.0188272, 0.021204] |
| js_sib | wrong | 75 | 16218 | mean | 0.0257854 | [0.0234922, 0.0281217] |

### Sibling gap vs step (fig3)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| step in [0,8) | correct | 225 |  | mean | 0.0155342 | [0.0126012, 0.0189208] |
| js_sib \| step in [8,16) | correct | 225 |  | mean | 0.0350286 | [0.0310244, 0.0389743] |
| js_sib \| step in [16,24) | correct | 225 |  | mean | 0.0296229 | [0.0261862, 0.0332044] |
| js_sib \| step in [24,32) | correct | 225 |  | mean | 0.0264411 | [0.0229186, 0.0301271] |
| js_sib \| step in [32,40) | correct | 225 |  | mean | 0.0238168 | [0.02045, 0.0273547] |
| js_sib \| step in [40,48) | correct | 225 |  | mean | 0.0214835 | [0.018434, 0.0248323] |
| js_sib \| step in [48,56) | correct | 225 |  | mean | 0.0179212 | [0.0151286, 0.0211125] |
| js_sib \| step in [56,64) | correct | 225 |  | mean | 0.0190105 | [0.0156399, 0.0226187] |
| js_sib \| step in [64,72) | correct | 217 |  | mean | 0.017863 | [0.01472, 0.0212191] |
| js_sib \| step in [72,80) | correct | 218 |  | mean | 0.0166434 | [0.0134777, 0.0198633] |
| js_sib \| step in [80,88) | correct | 202 |  | mean | 0.0118915 | [0.00950288, 0.014417] |
| js_sib \| step in [88,96) | correct | 202 |  | mean | 0.0126651 | [0.010293, 0.0150668] |
| js_sib \| step in [96,104) | correct | 187 |  | mean | 0.0133351 | [0.0109072, 0.0158406] |
| js_sib \| step in [104,112) | correct | 189 |  | mean | 0.0129876 | [0.010243, 0.0159299] |
| js_sib \| step in [0,8) | wrong | 75 |  | mean | 0.0224755 | [0.0166139, 0.0291929] |
| js_sib \| step in [8,16) | wrong | 75 |  | mean | 0.0389345 | [0.0315998, 0.0465767] |
| js_sib \| step in [16,24) | wrong | 75 |  | mean | 0.0307296 | [0.024026, 0.0374439] |
| js_sib \| step in [24,32) | wrong | 75 |  | mean | 0.0290184 | [0.022733, 0.036034] |
| js_sib \| step in [32,40) | wrong | 75 |  | mean | 0.0334477 | [0.0259505, 0.0410989] |
| js_sib \| step in [40,48) | wrong | 75 |  | mean | 0.0244806 | [0.0192993, 0.0297966] |
| js_sib \| step in [48,56) | wrong | 75 |  | mean | 0.0258254 | [0.019883, 0.0320698] |
| js_sib \| step in [56,64) | wrong | 75 |  | mean | 0.0278899 | [0.0199786, 0.0362384] |
| js_sib \| step in [64,72) | wrong | 74 |  | mean | 0.0234413 | [0.0170607, 0.0305992] |
| js_sib \| step in [72,80) | wrong | 75 |  | mean | 0.0217357 | [0.0166132, 0.0274708] |
| js_sib \| step in [80,88) | wrong | 71 |  | mean | 0.0235747 | [0.0178211, 0.0297716] |
| js_sib \| step in [88,96) | wrong | 69 |  | mean | 0.019451 | [0.0137348, 0.0259753] |
| js_sib \| step in [96,104) | wrong | 66 |  | mean | 0.0179315 | [0.0129185, 0.0241078] |
| js_sib \| step in [104,112) | wrong | 66 |  | mean | 0.0193306 | [0.0132534, 0.0260716] |
| delta \| step in [0,8) | correct | 225 |  | mean | -0.0608098 | [-0.114013, -0.0129024] |
| delta \| step in [8,16) | correct | 225 |  | mean | -0.278453 | [-0.358006, -0.202742] |
| delta \| step in [16,24) | correct | 225 |  | mean | -0.151355 | [-0.209653, -0.0944177] |
| delta \| step in [24,32) | correct | 225 |  | mean | -0.232475 | [-0.308134, -0.162524] |
| delta \| step in [32,40) | correct | 225 |  | mean | -0.0899186 | [-0.146258, -0.0376683] |
| delta \| step in [40,48) | correct | 225 |  | mean | -0.175037 | [-0.245422, -0.113418] |
| delta \| step in [48,56) | correct | 225 |  | mean | -0.0699005 | [-0.113802, -0.0339402] |
| delta \| step in [56,64) | correct | 225 |  | mean | -0.168447 | [-0.239665, -0.103735] |
| delta \| step in [64,72) | correct | 217 |  | mean | -0.119539 | [-0.179141, -0.0676939] |
| delta \| step in [72,80) | correct | 218 |  | mean | -0.12355 | [-0.185372, -0.0696472] |
| delta \| step in [80,88) | correct | 202 |  | mean | -0.0494134 | [-0.0923159, -0.014046] |
| delta \| step in [88,96) | correct | 202 |  | mean | -0.0973049 | [-0.158155, -0.0408888] |
| delta \| step in [96,104) | correct | 187 |  | mean | -0.0340495 | [-0.0697844, -0.00397769] |
| delta \| step in [104,112) | correct | 189 |  | mean | -0.125089 | [-0.198276, -0.0626058] |
| delta \| step in [0,8) | wrong | 75 |  | mean | -0.154577 | [-0.269971, -0.0556902] |
| delta \| step in [8,16) | wrong | 75 |  | mean | -0.228866 | [-0.354752, -0.109236] |
| delta \| step in [16,24) | wrong | 75 |  | mean | -0.141783 | [-0.257861, -0.0414702] |
| delta \| step in [24,32) | wrong | 75 |  | mean | -0.233044 | [-0.368561, -0.114987] |
| delta \| step in [32,40) | wrong | 75 |  | mean | -0.188815 | [-0.313504, -0.0779656] |
| delta \| step in [40,48) | wrong | 75 |  | mean | -0.164673 | [-0.267255, -0.0790443] |
| delta \| step in [48,56) | wrong | 75 |  | mean | -0.102994 | [-0.188528, -0.0278245] |
| delta \| step in [56,64) | wrong | 75 |  | mean | -0.200637 | [-0.332704, -0.083478] |
| delta \| step in [64,72) | wrong | 74 |  | mean | -0.159794 | [-0.275614, -0.0566382] |
| delta \| step in [72,80) | wrong | 75 |  | mean | -0.174051 | [-0.282268, -0.083226] |
| delta \| step in [80,88) | wrong | 71 |  | mean | -0.163235 | [-0.258137, -0.07816] |
| delta \| step in [88,96) | wrong | 69 |  | mean | -0.0961641 | [-0.195145, -0.0127038] |
| delta \| step in [96,104) | wrong | 66 |  | mean | -0.103482 | [-0.203875, -0.0231988] |
| delta \| step in [104,112) | wrong | 66 |  | mean | -0.156618 | [-0.264605, -0.067573] |

### Sibling gap vs |C_t| (fixed budgets) (fig4)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib | correct | 225 | base_s128 \|C_t\|=2 | mean | 0.0199505 | [0.0188272, 0.021204] |
| js_sib | correct | 246 | best_s128 \|C_t\|=2 | mean | 0.01092 | [0.0101186, 0.0117369] |
| js_sib | wrong | 75 | base_s128 \|C_t\|=2 | mean | 0.0257854 | [0.0234922, 0.0281217] |
| js_sib | wrong | 54 | best_s128 \|C_t\|=2 | mean | 0.0137 | [0.0113251, 0.0164782] |
| delta | correct | 225 | base_s128 \|C_t\|=2 | mean | -0.129502 | [-0.146443, -0.113495] |
| delta | correct | 246 | best_s128 \|C_t\|=2 | mean | -0.111469 | [-0.126224, -0.0971822] |
| delta | wrong | 75 | base_s128 \|C_t\|=2 | mean | -0.163062 | [-0.191035, -0.135272] |
| delta | wrong | 54 | best_s128 \|C_t\|=2 | mean | -0.136656 | [-0.175761, -0.101687] |

### PI shift vs sibling shift (fig5)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| cos_pi_sib | correct | 225 | 47498 | mean | 0.379019 | [0.366999, 0.390799] |
| cos_pi_sib | wrong | 75 | 16218 | mean | 0.342773 | [0.323753, 0.362399] |
| frac(cos>0) | correct | 225 |  | fraction | 0.668744 | [0.662007, 0.675414] |
| frac(cos>0) | wrong | 75 |  | fraction | 0.663522 | [0.65382, 0.673932] |

### B. Coordination gap (fig6)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S | correct | 225 | ctrl_ok tokens | mean | -0.0975567 | [-0.112777, -0.083246] |
| D_T | correct | 225 | ctrl_ok tokens | mean | -0.0753986 | [-0.0878829, -0.0639467] |
| D_S_ctrl | correct | 225 | ctrl_ok tokens | mean | 0.0195765 | [0.01443, 0.0248143] |
| D_T_ctrl | correct | 225 | ctrl_ok tokens | mean | 0.00993339 | [0.005495, 0.014325] |
| D_S | wrong | 75 | ctrl_ok tokens | mean | -0.135864 | [-0.164021, -0.107476] |
| D_T | wrong | 75 | ctrl_ok tokens | mean | -0.105149 | [-0.128596, -0.0822975] |
| D_S_ctrl | wrong | 75 | ctrl_ok tokens | mean | 0.0176268 | [0.0077734, 0.0278289] |
| D_T_ctrl | wrong | 75 | ctrl_ok tokens | mean | 0.0124039 | [0.00194623, 0.0224914] |
| coord | correct | 225 | ctrl_ok tokens | mean | -0.022158 | [-0.028671, -0.0156929] |
| coord_ctrl | correct | 225 | ctrl_ok tokens | mean | 0.00964312 | [0.00534773, 0.0141649] |
| coord | wrong | 75 | ctrl_ok tokens | mean | -0.0307144 | [-0.0425892, -0.0192746] |
| coord_ctrl | wrong | 75 | ctrl_ok tokens | mean | 0.00522288 | [-0.00365248, 0.0136474] |

### C. Sibling vs matched control (paired) (fig7)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S - D_S_ctrl (paired) | correct | 225 | ctrl_ok tokens | mean | -0.117133 | [-0.129711, -0.104809] |
| D_S - D_S_ctrl (paired) | wrong | 75 | ctrl_ok tokens | mean | -0.15349 | [-0.177577, -0.12942] |
| coord - coord_ctrl (paired) | correct | 225 | ctrl_ok tokens | mean | -0.0318011 | [-0.0368404, -0.0271311] |
| coord - coord_ctrl (paired) | wrong | 75 | ctrl_ok tokens | mean | -0.0359373 | [-0.0450933, -0.0269679] |

### A. Pairwise co-gain (distance-matched) (fig8)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| pair co-gain P(both g>0) co-decoded | correct | 225 | 23656 co-decoded pairs with a distance match | fraction | 0.775913 |  |
| pair co-gain distance-matched null | correct | 225 | perm p=0.0020 | fraction | 0.747452 | [0.742938, 0.75224] |
| pair co-gain real - matched expectation | correct | 225 | rollout bootstrap CI | difference | 0.0286402 | [0.026175, 0.0312736] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | correct | 225 | real=0.7759 | fraction | 0.746178 | [0.744548, 0.747446] |
| pair co-gain P(both g>0) co-decoded | wrong | 75 | 8092 co-decoded pairs with a distance match | fraction | 0.71787 |  |
| pair co-gain distance-matched null | wrong | 75 | perm p=0.0020 | fraction | 0.685271 | [0.675729, 0.693589] |
| pair co-gain real - matched expectation | wrong | 75 | rollout bootstrap CI | difference | 0.0324891 | [0.0275285, 0.0373931] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | wrong | 75 | real=0.7179 | fraction | 0.682194 | [0.679022, 0.68516] |

### PI effect types (fig10)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| sharpen | correct | 225 |  | mean | 0.850878 | [0.839141, 0.861909] |
| defer | correct | 225 |  | mean | 0.121984 | [0.112754, 0.131835] |
| redirect | correct | 225 |  | mean | 0.027138 | [0.0245853, 0.0297476] |
| sharpen | wrong | 75 |  | mean | 0.810026 | [0.789508, 0.830568] |
| defer | wrong | 75 |  | mean | 0.154581 | [0.137774, 0.170707] |
| redirect | wrong | 75 |  | mean | 0.0353928 | [0.0299371, 0.0408944] |
| defer\|redirect D_S<0 (conflict) | correct | 225 |  | mean | 0.196053 | [0.182545, 0.210063] |
| defer\|redirect D_S>=0 | correct | 225 |  | mean | 0.132699 | [0.12211, 0.143804] |
| defer\|redirect D_S<0 (conflict) | wrong | 75 |  | mean | 0.24826 | [0.221082, 0.275879] |
| defer\|redirect D_S>=0 | wrong | 75 |  | mean | 0.167899 | [0.149417, 0.186869] |

### Decoding order (fig11)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| overlap | correct | 225 |  | mean | 0.543649 | [0.535526, 0.55164] |
| overlap | wrong | 75 |  | mean | 0.554462 | [0.540532, 0.567737] |
| all | correct | 225 |  | mean | 0.197377 | [0.192301, 0.202641] |
| D_S<0 | correct | 225 |  | mean | 0.242752 | [0.234661, 0.251027] |
| D_S>=0 | correct | 225 |  | mean | 0.181498 | [0.175847, 0.18716] |
| all | wrong | 75 |  | mean | 0.184486 | [0.174133, 0.194819] |
| D_S<0 | wrong | 75 |  | mean | 0.241526 | [0.226293, 0.256663] |
| D_S>=0 | wrong | 75 |  | mean | 0.162884 | [0.15129, 0.174325] |
| spearman | correct | 225 |  | mean | 0.373305 | [0.362702, 0.384798] |
| spearman | wrong | 75 |  | mean | 0.384989 | [0.364209, 0.406572] |

## Files

- `summary.csv`: every number above
- `tokens.csv.gz`: one row per revealed token (all runs)
- `steps.csv.gz`: one row per decoding step
- `rollouts.csv`: one row per rollout with question/gold/prediction/completion
- `fig*.png`: figures
