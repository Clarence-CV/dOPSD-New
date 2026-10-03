# PI / intra-step coordination analysis report

generated 2026-10-02T23:56:33, code be724bc

FIXED BUDGET (128 steps), primary = trained step-448. GSM8K, 300 seeded test problems, T=0.9, analysis re-scored in fp16. best = adapter step-448 (picked by greedy sweep: base 78.3%, step-448 82.7%). Teacher = base weights + PI (fixed teacher), as in training.

## Runs

| run | records | rollouts | accuracy | steps | threshold | temperature | gen_length | block_length | adapter | teacher | pi_samples | teacher_retain_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_s128 | /tmp/pipe_gsm/best_s128_t0.9 | 300 | 0.820 | 128 | none (fixed budget) | 0.9 | 256 | 32 | /scratch/11277/yilunai0430/dopsd/runs/gsm_opsd/adapters/step-448 | fixed | 1 | 0.25 |
| base_s128 | /tmp/pipe_gsm/base_s128_t0.9 | 300 | 0.750 | 128 | none (fixed budget) | 0.9 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |

Figures 1-3 and 5-8, 10, 11 use the primary run `best_s128`; figure 4 uses every fixed-budget run; figure 9 uses the threshold run.

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
| g | correct | 246 | 49618 | mean | -0.022333 | [-0.0263413, -0.018361] |
| g | wrong | 54 | 11134 | mean | -0.0309562 | [-0.042218, -0.02073] |
| frac(p_S<0.5 & g>0.5) | correct | 246 |  | fraction | 0.00572373 | [0.00494867, 0.00650352] |
| frac(p_S<0.5 & g>0.5) | wrong | 54 |  | fraction | 0.00799353 | [0.00588265, 0.0104483] |

### Sibling gap (fig2)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| delta | correct | 246 | 49618 | mean | -0.111469 | [-0.126224, -0.0971822] |
| delta | wrong | 54 | 11134 | mean | -0.136656 | [-0.175761, -0.101687] |
| js_sib | correct | 246 | 49618 | mean | 0.01092 | [0.0101186, 0.0117369] |
| js_sib | wrong | 54 | 11134 | mean | 0.0137 | [0.0113251, 0.0164782] |

### Sibling gap vs step (fig3)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| step in [0,8) | correct | 246 |  | mean | 0.0112365 | [0.00895743, 0.013757] |
| js_sib \| step in [8,16) | correct | 246 |  | mean | 0.0163229 | [0.0137445, 0.0191338] |
| js_sib \| step in [16,24) | correct | 246 |  | mean | 0.012653 | [0.010256, 0.0152366] |
| js_sib \| step in [24,32) | correct | 246 |  | mean | 0.0153674 | [0.0126483, 0.0183852] |
| js_sib \| step in [32,40) | correct | 246 |  | mean | 0.0119725 | [0.00951773, 0.0145692] |
| js_sib \| step in [40,48) | correct | 246 |  | mean | 0.00953866 | [0.00760695, 0.0116519] |
| js_sib \| step in [48,56) | correct | 245 |  | mean | 0.00830038 | [0.00609214, 0.010608] |
| js_sib \| step in [56,64) | correct | 245 |  | mean | 0.0107362 | [0.00824597, 0.0134477] |
| js_sib \| step in [64,72) | correct | 236 |  | mean | 0.00655339 | [0.00471235, 0.00864941] |
| js_sib \| step in [72,80) | correct | 237 |  | mean | 0.0112668 | [0.00871797, 0.0140377] |
| js_sib \| step in [80,88) | correct | 207 |  | mean | 0.00877187 | [0.00648661, 0.0113] |
| js_sib \| step in [88,96) | correct | 206 |  | mean | 0.0104994 | [0.00782353, 0.0134679] |
| js_sib \| step in [96,104) | correct | 167 |  | mean | 0.00670835 | [0.00480034, 0.00896803] |
| js_sib \| step in [104,112) | correct | 166 |  | mean | 0.00982695 | [0.00702177, 0.0127008] |
| js_sib \| step in [0,8) | wrong | 54 |  | mean | 0.0109288 | [0.00753869, 0.0145737] |
| js_sib \| step in [8,16) | wrong | 54 |  | mean | 0.0208893 | [0.0151978, 0.0267401] |
| js_sib \| step in [16,24) | wrong | 54 |  | mean | 0.0136783 | [0.00955624, 0.0183964] |
| js_sib \| step in [24,32) | wrong | 54 |  | mean | 0.0128096 | [0.0084761, 0.0177033] |
| js_sib \| step in [32,40) | wrong | 54 |  | mean | 0.00982146 | [0.00574543, 0.0147494] |
| js_sib \| step in [40,48) | wrong | 54 |  | mean | 0.0125421 | [0.00758785, 0.0185212] |
| js_sib \| step in [48,56) | wrong | 54 |  | mean | 0.0124344 | [0.00666989, 0.0193803] |
| js_sib \| step in [56,64) | wrong | 54 |  | mean | 0.0161087 | [0.00900471, 0.0250845] |
| js_sib \| step in [64,72) | wrong | 49 |  | mean | 0.00952735 | [0.00492394, 0.015027] |
| js_sib \| step in [72,80) | wrong | 52 |  | mean | 0.0172715 | [0.0101589, 0.0258118] |
| js_sib \| step in [80,88) | wrong | 46 |  | mean | 0.00760971 | [0.00342591, 0.0131207] |
| js_sib \| step in [88,96) | wrong | 47 |  | mean | 0.0143351 | [0.00744003, 0.0221925] |
| js_sib \| step in [96,104) | wrong | 42 |  | mean | 0.0140981 | [0.00829069, 0.0203092] |
| js_sib \| step in [104,112) | wrong | 42 |  | mean | 0.0203083 | [0.0114659, 0.0310963] |
| delta \| step in [0,8) | correct | 246 |  | mean | -0.0810548 | [-0.126734, -0.0438608] |
| delta \| step in [8,16) | correct | 246 |  | mean | -0.176743 | [-0.241667, -0.113322] |
| delta \| step in [16,24) | correct | 246 |  | mean | -0.0858783 | [-0.134468, -0.0437379] |
| delta \| step in [24,32) | correct | 246 |  | mean | -0.204925 | [-0.274511, -0.140203] |
| delta \| step in [32,40) | correct | 246 |  | mean | -0.108021 | [-0.161572, -0.0621653] |
| delta \| step in [40,48) | correct | 246 |  | mean | -0.0954207 | [-0.144691, -0.0524296] |
| delta \| step in [48,56) | correct | 245 |  | mean | -0.0691644 | [-0.119756, -0.026959] |
| delta \| step in [56,64) | correct | 245 |  | mean | -0.136594 | [-0.194883, -0.0822473] |
| delta \| step in [64,72) | correct | 236 |  | mean | -0.0401644 | [-0.0720837, -0.0140085] |
| delta \| step in [72,80) | correct | 237 |  | mean | -0.142294 | [-0.205591, -0.0874346] |
| delta \| step in [80,88) | correct | 207 |  | mean | -0.0840158 | [-0.135856, -0.0386252] |
| delta \| step in [88,96) | correct | 206 |  | mean | -0.120787 | [-0.183774, -0.0641155] |
| delta \| step in [96,104) | correct | 167 |  | mean | -0.0575447 | [-0.105988, -0.0185851] |
| delta \| step in [104,112) | correct | 166 |  | mean | -0.135107 | [-0.215205, -0.0658284] |
| delta \| step in [0,8) | wrong | 54 |  | mean | -0.106412 | [-0.208562, -0.0250304] |
| delta \| step in [8,16) | wrong | 54 |  | mean | -0.240147 | [-0.378802, -0.117929] |
| delta \| step in [16,24) | wrong | 54 |  | mean | -0.0529965 | [-0.0968639, -0.0169913] |
| delta \| step in [24,32) | wrong | 54 |  | mean | -0.122797 | [-0.247336, -0.0231528] |
| delta \| step in [32,40) | wrong | 54 |  | mean | -0.0408058 | [-0.111925, 0.000131574] |
| delta \| step in [40,48) | wrong | 54 |  | mean | -0.190069 | [-0.353678, -0.0637087] |
| delta \| step in [48,56) | wrong | 54 |  | mean | -0.152877 | [-0.305993, -0.0241363] |
| delta \| step in [56,64) | wrong | 54 |  | mean | -0.161379 | [-0.335829, -0.0451244] |
| delta \| step in [64,72) | wrong | 49 |  | mean | -0.0946533 | [-0.195828, -0.00385099] |
| delta \| step in [72,80) | wrong | 52 |  | mean | -0.0580271 | [-0.144743, 0.0105319] |
| delta \| step in [80,88) | wrong | 46 |  | mean | -0.0354231 | [-0.0961988, 0.00479758] |
| delta \| step in [88,96) | wrong | 47 |  | mean | -0.230331 | [-0.402748, -0.0816379] |
| delta \| step in [96,104) | wrong | 42 |  | mean | -0.103611 | [-0.231805, -0.00910752] |
| delta \| step in [104,112) | wrong | 42 |  | mean | -0.356774 | [-0.607216, -0.145904] |

### Sibling gap vs |C_t| (fixed budgets) (fig4)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib | correct | 246 | best_s128 \|C_t\|=2 | mean | 0.01092 | [0.0101186, 0.0117369] |
| js_sib | correct | 225 | base_s128 \|C_t\|=2 | mean | 0.0199505 | [0.0188272, 0.021204] |
| js_sib | wrong | 54 | best_s128 \|C_t\|=2 | mean | 0.0137 | [0.0113251, 0.0164782] |
| js_sib | wrong | 75 | base_s128 \|C_t\|=2 | mean | 0.0257854 | [0.0234922, 0.0281217] |
| delta | correct | 246 | best_s128 \|C_t\|=2 | mean | -0.111469 | [-0.126224, -0.0971822] |
| delta | correct | 225 | base_s128 \|C_t\|=2 | mean | -0.129502 | [-0.146443, -0.113495] |
| delta | wrong | 54 | best_s128 \|C_t\|=2 | mean | -0.136656 | [-0.175761, -0.101687] |
| delta | wrong | 75 | base_s128 \|C_t\|=2 | mean | -0.163062 | [-0.191035, -0.135272] |

### PI shift vs sibling shift (fig5)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| cos_pi_sib | correct | 246 | 49618 | mean | 0.0156334 | [0.00588502, 0.0255] |
| cos_pi_sib | wrong | 54 | 11134 | mean | 0.0152361 | [-0.00146322, 0.0317951] |
| frac(cos>0) | correct | 246 |  | fraction | 0.320126 | [0.313723, 0.326875] |
| frac(cos>0) | wrong | 54 |  | fraction | 0.361146 | [0.344781, 0.378035] |

### B. Coordination gap (fig6)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S | correct | 246 | ctrl_ok tokens | mean | -0.0890439 | [-0.103412, -0.0756364] |
| D_T | correct | 246 | ctrl_ok tokens | mean | -0.0287776 | [-0.0363829, -0.0212168] |
| D_S_ctrl | correct | 246 | ctrl_ok tokens | mean | -0.0251664 | [-0.0310855, -0.0193492] |
| D_T_ctrl | correct | 246 | ctrl_ok tokens | mean | 0.0117134 | [0.00795061, 0.0153443] |
| D_S | wrong | 54 | ctrl_ok tokens | mean | -0.106171 | [-0.139492, -0.0763476] |
| D_T | wrong | 54 | ctrl_ok tokens | mean | -0.0224353 | [-0.040455, -0.00762515] |
| D_S_ctrl | wrong | 54 | ctrl_ok tokens | mean | -0.0241282 | [-0.0369077, -0.0128649] |
| D_T_ctrl | wrong | 54 | ctrl_ok tokens | mean | 0.0210158 | [0.0125792, 0.0303658] |
| coord | correct | 246 | ctrl_ok tokens | mean | -0.0602662 | [-0.0692156, -0.0519179] |
| coord_ctrl | correct | 246 | ctrl_ok tokens | mean | -0.0368799 | [-0.0427224, -0.0316301] |
| coord | wrong | 54 | ctrl_ok tokens | mean | -0.0837354 | [-0.104854, -0.0642909] |
| coord_ctrl | wrong | 54 | ctrl_ok tokens | mean | -0.0451439 | [-0.0579613, -0.0333925] |

### C. Sibling vs matched control (paired) (fig7)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S - D_S_ctrl (paired) | correct | 246 | ctrl_ok tokens | mean | -0.0638774 | [-0.0744525, -0.0540491] |
| D_S - D_S_ctrl (paired) | wrong | 54 | ctrl_ok tokens | mean | -0.0820426 | [-0.106669, -0.0593088] |
| coord - coord_ctrl (paired) | correct | 246 | ctrl_ok tokens | mean | -0.0233864 | [-0.0296667, -0.0177753] |
| coord - coord_ctrl (paired) | wrong | 54 | ctrl_ok tokens | mean | -0.0385915 | [-0.0521419, -0.0256647] |

### A. Pairwise co-gain (distance-matched) (fig8)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| pair co-gain P(both g>0) co-decoded | correct | 246 | 24631 co-decoded pairs with a distance match | fraction | 0.221509 |  |
| pair co-gain distance-matched null | correct | 246 | perm p=0.0020 | fraction | 0.157513 | [0.153484, 0.161363] |
| pair co-gain real - matched expectation | correct | 246 | rollout bootstrap CI | difference | 0.0638572 | [0.0611974, 0.0666093] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | correct | 246 | real=0.2214 | fraction | 0.149048 | [0.146198, 0.151827] |
| pair co-gain P(both g>0) co-decoded | wrong | 54 | 5540 co-decoded pairs with a distance match | fraction | 0.22491 |  |
| pair co-gain distance-matched null | wrong | 54 | perm p=0.0020 | fraction | 0.16464 | [0.155054, 0.173741] |
| pair co-gain real - matched expectation | wrong | 54 | rollout bootstrap CI | difference | 0.0600426 | [0.05387, 0.0661203] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | wrong | 54 | real=0.2249 | fraction | 0.153401 | [0.148186, 0.158579] |

### PI effect types (fig10)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| sharpen | correct | 246 |  | mean | 0.347555 | [0.337492, 0.35778] |
| defer | correct | 246 |  | mean | 0.634387 | [0.624701, 0.643841] |
| redirect | correct | 246 |  | mean | 0.018058 | [0.016383, 0.0198778] |
| sharpen | wrong | 54 |  | mean | 0.355218 | [0.332408, 0.378273] |
| defer | wrong | 54 |  | mean | 0.621699 | [0.600213, 0.643538] |
| redirect | wrong | 54 |  | mean | 0.0230825 | [0.0179418, 0.029055] |
| defer\|redirect D_S<0 (conflict) | correct | 246 |  | mean | 0.604375 | [0.590937, 0.617866] |
| defer\|redirect D_S>=0 | correct | 246 |  | mean | 0.668754 | [0.658412, 0.678707] |
| defer\|redirect D_S<0 (conflict) | wrong | 54 |  | mean | 0.627846 | [0.596556, 0.661588] |
| defer\|redirect D_S>=0 | wrong | 54 |  | mean | 0.651241 | [0.629039, 0.673402] |

### Decoding order (fig11)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| overlap | correct | 246 |  | mean | 0.539741 | [0.532466, 0.547364] |
| overlap | wrong | 54 |  | mean | 0.540669 | [0.52532, 0.556727] |
| all | correct | 246 |  | mean | 0.181426 | [0.176805, 0.186082] |
| D_S<0 | correct | 246 |  | mean | 0.237788 | [0.229039, 0.246371] |
| D_S>=0 | correct | 246 |  | mean | 0.162303 | [0.15744, 0.167457] |
| all | wrong | 54 |  | mean | 0.178552 | [0.168566, 0.187985] |
| D_S<0 | wrong | 54 |  | mean | 0.231945 | [0.214376, 0.25] |
| D_S>=0 | wrong | 54 |  | mean | 0.158189 | [0.147558, 0.168081] |
| spearman | correct | 246 |  | mean | 0.434852 | [0.425659, 0.444091] |
| spearman | wrong | 54 |  | mean | 0.416579 | [0.398298, 0.434254] |

## Files

- `summary.csv`: every number above
- `tokens.csv.gz`: one row per revealed token (all runs)
- `steps.csv.gz`: one row per decoding step
- `rollouts.csv`: one row per rollout with question/gold/prediction/completion
- `fig*.png`: figures
