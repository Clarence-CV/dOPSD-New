# PI / intra-step coordination analysis report

generated 2026-09-30T13:58:43, code 53c5aeb

SMOKE TEST ONLY: base LLaDA, 20 GSM8K problems, greedy decoding, analysis re-scored in bf16 (the pipeline now uses fp16). Pipeline check, not evidence.

## Runs

| run | records | rollouts | accuracy | steps | threshold | temperature | gen_length | block_length | adapter | teacher | pi_samples | teacher_retain_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s128 | /tmp/sm3/s128 | 20 | 0.750 | 128 | none (fixed budget) | 0.0 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |
| s64 | /tmp/sm3/s64 | 20 | 0.800 | 64 | none (fixed budget) | 0.0 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |
| thr0.9 | /tmp/sm3/thr0.9 | 20 | 0.750 | 128 | 0.9 | 0.0 | 256 | 32 | none (base model) | same-as-student | 1 | 0.25 |

Figures 1-3 and 5-8, 10, 11 use the primary run `s128`; figure 4 uses every fixed-budget run; figure 9 uses the threshold run.

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
| g | correct | 15 | 3032 | mean | 0.0380357 | [0.0241643, 0.0511928] |
| g | wrong | 5 | 1112 | mean | 0.0223707 | [-0.0200139, 0.0666403] |
| frac(p_S<0.5 & g>0.5) | correct | 15 |  | fraction | 0.0161609 | [0.0118894, 0.0213076] |
| frac(p_S<0.5 & g>0.5) | wrong | 5 |  | fraction | 0.0170863 | [0.00719424, 0.0276786] |

### Sibling gap (fig2)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| delta | correct | 15 | 3032 | mean | -0.0236283 | [-0.0600347, 0.00833578] |
| delta | wrong | 5 | 1112 | mean | -0.131011 | [-0.222568, -0.0407622] |
| js_sib | correct | 15 | 3032 | mean | 0.0164098 | [0.0131947, 0.0201872] |
| js_sib | wrong | 5 | 1112 | mean | 0.0221461 | [0.016408, 0.0274358] |

### Sibling gap vs step (fig3)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| step in [0,8) | correct | 15 |  | mean | 0.0147937 | [0.00657139, 0.0246209] |
| js_sib \| step in [8,16) | correct | 15 |  | mean | 0.0218248 | [0.0105914, 0.0353168] |
| js_sib \| step in [16,24) | correct | 15 |  | mean | 0.0349399 | [0.0210476, 0.0505574] |
| js_sib \| step in [24,32) | correct | 15 |  | mean | 0.0206628 | [0.0113926, 0.0311741] |
| js_sib \| step in [32,40) | correct | 15 |  | mean | 0.0154724 | [0.00784111, 0.024801] |
| js_sib \| step in [40,48) | correct | 15 |  | mean | 0.0151229 | [0.00817499, 0.0229997] |
| js_sib \| step in [48,56) | correct | 15 |  | mean | 0.0235551 | [0.0110951, 0.0414861] |
| js_sib \| step in [56,64) | correct | 15 |  | mean | 0.0132177 | [0.00548981, 0.0220843] |
| js_sib \| step in [64,72) | correct | 14 |  | mean | 0.00864273 | [0.00357009, 0.0142749] |
| js_sib \| step in [72,80) | correct | 14 |  | mean | 0.0108155 | [0.00564042, 0.0177083] |
| js_sib \| step in [80,88) | correct | 12 |  | mean | 0.00514578 | [0.00211776, 0.00909654] |
| js_sib \| step in [88,96) | correct | 12 |  | mean | 0.00936701 | [0.00403215, 0.0157909] |
| js_sib \| step in [96,104) | correct | 11 |  | mean | 0.0167563 | [0.00663195, 0.0289913] |
| js_sib \| step in [104,112) | correct | 11 |  | mean | 0.0127503 | [0.00467582, 0.0218095] |
| js_sib \| step in [0,8) | wrong | 5 |  | mean | 0.0140764 | [0.00602516, 0.0227908] |
| js_sib \| step in [8,16) | wrong | 5 |  | mean | 0.0122246 | [0.00485284, 0.0195965] |
| js_sib \| step in [16,24) | wrong | 5 |  | mean | 0.0334655 | [0.019361, 0.0479681] |
| js_sib \| step in [24,32) | wrong | 5 |  | mean | 0.0265201 | [0.00699846, 0.0516639] |
| js_sib \| step in [32,40) | wrong | 5 |  | mean | 0.0331902 | [0.0109421, 0.0603899] |
| js_sib \| step in [40,48) | wrong | 5 |  | mean | 0.0213226 | [0.00914017, 0.036948] |
| js_sib \| step in [48,56) | wrong | 5 |  | mean | 0.0154747 | [0.00692845, 0.0239111] |
| js_sib \| step in [56,64) | wrong | 5 |  | mean | 0.0122327 | [0.00604315, 0.0176387] |
| js_sib \| step in [64,72) | wrong | 5 |  | mean | 0.030287 | [0.00778714, 0.0563323] |
| js_sib \| step in [72,80) | wrong | 5 |  | mean | 0.0312733 | [0.00701611, 0.0641983] |
| js_sib \| step in [80,88) | wrong | 5 |  | mean | 0.0199671 | [0.00511278, 0.0444184] |
| js_sib \| step in [88,96) | wrong | 5 |  | mean | 0.0146858 | [0.00248619, 0.0290944] |
| js_sib \| step in [96,104) | wrong | 5 |  | mean | 0.033202 | [0.00614949, 0.0759992] |
| js_sib \| step in [104,112) | wrong | 5 |  | mean | 0.0127168 | [0.00272085, 0.0219511] |
| delta \| step in [0,8) | correct | 15 |  | mean | -0.080529 | [-0.248921, 0.0246828] |
| delta \| step in [8,16) | correct | 15 |  | mean | -0.0133503 | [-0.161351, 0.0845716] |
| delta \| step in [16,24) | correct | 15 |  | mean | -0.151657 | [-0.335218, -0.0135229] |
| delta \| step in [24,32) | correct | 15 |  | mean | -0.0450114 | [-0.237865, 0.0653629] |
| delta \| step in [32,40) | correct | 15 |  | mean | 0.0314269 | [0.0038099, 0.0616893] |
| delta \| step in [40,48) | correct | 15 |  | mean | 0.0468849 | [0.0227349, 0.0729527] |
| delta \| step in [48,56) | correct | 15 |  | mean | -0.123476 | [-0.418835, 0.0472963] |
| delta \| step in [56,64) | correct | 15 |  | mean | -0.0862838 | [-0.302944, 0.034023] |
| delta \| step in [64,72) | correct | 14 |  | mean | 0.0221619 | [0.00821543, 0.0386649] |
| delta \| step in [72,80) | correct | 14 |  | mean | 0.0187047 | [-0.00925646, 0.0465304] |
| delta \| step in [80,88) | correct | 12 |  | mean | 0.0153476 | [0.00579155, 0.0281612] |
| delta \| step in [88,96) | correct | 12 |  | mean | 0.0162753 | [-0.0077358, 0.0399771] |
| delta \| step in [96,104) | correct | 11 |  | mean | 0.0562224 | [0.0208767, 0.100581] |
| delta \| step in [104,112) | correct | 11 |  | mean | 0.0390089 | [0.0146129, 0.0663843] |
| delta \| step in [0,8) | wrong | 5 |  | mean | 0.039632 | [0.0212209, 0.0603592] |
| delta \| step in [8,16) | wrong | 5 |  | mean | 0.0334235 | [0.00491185, 0.0630908] |
| delta \| step in [16,24) | wrong | 5 |  | mean | -0.111891 | [-0.288124, 0.064231] |
| delta \| step in [24,32) | wrong | 5 |  | mean | -0.499863 | [-1.07409, 0.00472219] |
| delta \| step in [32,40) | wrong | 5 |  | mean | -0.114189 | [-0.33395, 0.0429409] |
| delta \| step in [40,48) | wrong | 5 |  | mean | 0.0621883 | [0.0233539, 0.106628] |
| delta \| step in [48,56) | wrong | 5 |  | mean | 0.0498092 | [0.0185638, 0.082393] |
| delta \| step in [56,64) | wrong | 5 |  | mean | 0.029587 | [0.00906014, 0.0501139] |
| delta \| step in [64,72) | wrong | 5 |  | mean | -0.250514 | [-0.826564, 0.0880304] |
| delta \| step in [72,80) | wrong | 5 |  | mean | -0.216791 | [-0.744729, 0.0752952] |
| delta \| step in [80,88) | wrong | 5 |  | mean | -0.23206 | [-0.714163, 0.0293731] |
| delta \| step in [88,96) | wrong | 5 |  | mean | -0.101841 | [-0.352659, 0.0453024] |
| delta \| step in [96,104) | wrong | 5 |  | mean | -0.572902 | [-1.7769, 0.060009] |
| delta \| step in [104,112) | wrong | 5 |  | mean | 0.0219391 | [0.0026552, 0.047197] |

### Sibling gap vs |C_t| (fixed budgets) (fig4)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib | correct | 15 | s128 \|C_t\|=2 | mean | 0.0164098 | [0.0131947, 0.0201872] |
| js_sib | correct | 16 | s64 \|C_t\|=4 | mean | 0.0641201 | [0.0538482, 0.0759136] |
| js_sib | wrong | 5 | s128 \|C_t\|=2 | mean | 0.0221461 | [0.016408, 0.0274358] |
| js_sib | wrong | 4 | s64 \|C_t\|=4 | mean | 0.0817866 | [0.060368, 0.103556] |
| delta | correct | 15 | s128 \|C_t\|=2 | mean | -0.0236283 | [-0.0600347, 0.00833578] |
| delta | correct | 16 | s64 \|C_t\|=4 | mean | -0.229625 | [-0.30911, -0.156734] |
| delta | wrong | 5 | s128 \|C_t\|=2 | mean | -0.131011 | [-0.222568, -0.0407622] |
| delta | wrong | 4 | s64 \|C_t\|=4 | mean | -0.258877 | [-0.478602, -0.104076] |

### PI shift vs sibling shift (fig5)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| cos_pi_sib | correct | 15 | 3032 | mean | 0.408668 | [0.372499, 0.438361] |
| cos_pi_sib | wrong | 5 | 1112 | mean | 0.371357 | [0.323168, 0.41312] |
| frac(cos>0) | correct | 15 |  | fraction | 0.680079 | [0.658957, 0.698391] |
| frac(cos>0) | wrong | 5 |  | fraction | 0.673561 | [0.641964, 0.705379] |

### B. Coordination gap (fig6)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S | correct | 15 | ctrl_ok tokens | mean | -0.0189567 | [-0.0548678, 0.0116588] |
| D_T | correct | 15 | ctrl_ok tokens | mean | -0.0199846 | [-0.05964, 0.0118822] |
| D_S_ctrl | correct | 15 | ctrl_ok tokens | mean | 0.0442558 | [0.0356619, 0.0546797] |
| D_T_ctrl | correct | 15 | ctrl_ok tokens | mean | 0.0149263 | [-0.00104688, 0.0315516] |
| D_S | wrong | 5 | ctrl_ok tokens | mean | -0.0956749 | [-0.145859, -0.0468623] |
| D_T | wrong | 5 | ctrl_ok tokens | mean | -0.100411 | [-0.161865, -0.0317501] |
| D_S_ctrl | wrong | 5 | ctrl_ok tokens | mean | 0.0325415 | [0.0141511, 0.0527575] |
| D_T_ctrl | wrong | 5 | ctrl_ok tokens | mean | -0.00454608 | [-0.022991, 0.0145952] |
| coord | correct | 15 | ctrl_ok tokens | mean | 0.00102788 | [-0.0128948, 0.0120017] |
| coord_ctrl | correct | 15 | ctrl_ok tokens | mean | 0.0293295 | [0.016098, 0.0422079] |
| coord | wrong | 5 | ctrl_ok tokens | mean | 0.00473619 | [-0.0150283, 0.0335392] |
| coord_ctrl | wrong | 5 | ctrl_ok tokens | mean | 0.0370876 | [0.00938999, 0.0618222] |

### C. Sibling vs matched control (paired) (fig7)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| D_S - D_S_ctrl (paired) | correct | 15 | ctrl_ok tokens | mean | -0.0632125 | [-0.0989431, -0.0342823] |
| D_S - D_S_ctrl (paired) | wrong | 5 | ctrl_ok tokens | mean | -0.128216 | [-0.167071, -0.0755801] |
| coord - coord_ctrl (paired) | correct | 15 | ctrl_ok tokens | mean | -0.0283017 | [-0.0421447, -0.017617] |
| coord - coord_ctrl (paired) | wrong | 5 | ctrl_ok tokens | mean | -0.0323514 | [-0.0475031, -0.0174306] |

### A. Pairwise co-gain (distance-matched) (fig8)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| pair co-gain P(both g>0) co-decoded | correct | 15 | 1505 co-decoded pairs with a distance match | fraction | 0.827907 |  |
| pair co-gain distance-matched null | correct | 15 | perm p=0.0040 | fraction | 0.802481 | [0.786711, 0.819269] |
| pair co-gain real - matched expectation | correct | 15 | rollout bootstrap CI | difference | 0.0256999 | [0.0139761, 0.0387757] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | correct | 15 | real=0.8279 | fraction | 0.802518 | [0.797658, 0.806645] |
| pair co-gain P(both g>0) co-decoded | wrong | 5 | 554 co-decoded pairs with a distance match | fraction | 0.741877 |  |
| pair co-gain distance-matched null | wrong | 5 | perm p=0.0279 | fraction | 0.707949 | [0.672338, 0.74102] |
| pair co-gain real - matched expectation | wrong | 5 | rollout bootstrap CI | difference | 0.0337573 | [0.0261065, 0.0423892] |
| REF all-up rate (within-block shuffle, adjacency-confounded) | wrong | 5 | real=0.7419 | fraction | 0.708213 | [0.697608, 0.720307] |

### Sibling gap vs natural |C_t| (threshold) (fig9)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| js_sib \| \|C_t\| in 2 | correct | 14 | thr0.9 | mean | 0.00545527 | [0.00353817, 0.0077802] |
| js_sib \| \|C_t\| in 3-4 | correct | 15 | thr0.9 | mean | 0.00613267 | [0.00508399, 0.00712413] |
| js_sib \| \|C_t\| in 5-8 | correct | 15 | thr0.9 | mean | 0.00789245 | [0.00644974, 0.00925951] |
| js_sib \| \|C_t\| in 9+ | correct | 15 | thr0.9 | mean | 0.00773393 | [0.0067589, 0.0086383] |
| js_sib \| \|C_t\| in 2 | wrong | 5 | thr0.9 | mean | 0.00769397 | [0.00605409, 0.0101946] |
| js_sib \| \|C_t\| in 3-4 | wrong | 5 | thr0.9 | mean | 0.00678225 | [0.00467228, 0.00868529] |
| js_sib \| \|C_t\| in 5-8 | wrong | 5 | thr0.9 | mean | 0.00714068 | [0.00534081, 0.0086143] |
| js_sib \| \|C_t\| in 9+ | wrong | 5 | thr0.9 | mean | 0.00598419 | [0.00382871, 0.00967848] |
| D_S \| \|C_t\| in 2 | correct | 14 | thr0.9 | mean | 0.0121005 | [0.00449554, 0.0183069] |
| D_S \| \|C_t\| in 3-4 | correct | 15 | thr0.9 | mean | 0.017454 | [0.0146529, 0.0204971] |
| D_S \| \|C_t\| in 5-8 | correct | 15 | thr0.9 | mean | 0.0219735 | [0.0166044, 0.0269522] |
| D_S \| \|C_t\| in 9+ | correct | 15 | thr0.9 | mean | 0.022098 | [0.0195621, 0.024543] |
| D_S \| \|C_t\| in 2 | wrong | 5 | thr0.9 | mean | 0.0245028 | [0.0214065, 0.0317227] |
| D_S \| \|C_t\| in 3-4 | wrong | 5 | thr0.9 | mean | 0.0185205 | [0.0136886, 0.0214017] |
| D_S \| \|C_t\| in 5-8 | wrong | 5 | thr0.9 | mean | 0.0173767 | [0.0110291, 0.0226664] |
| D_S \| \|C_t\| in 9+ | wrong | 5 | thr0.9 | mean | 0.015194 | [0.0104221, 0.0202948] |

### PI effect types (fig10)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| sharpen | correct | 15 |  | mean | 0.884565 | [0.853269, 0.911471] |
| defer | correct | 15 |  | mean | 0.103892 | [0.0802873, 0.130889] |
| redirect | correct | 15 |  | mean | 0.0115435 | [0.00641145, 0.0179415] |
| sharpen | wrong | 5 |  | mean | 0.831835 | [0.758036, 0.894784] |
| defer | wrong | 5 |  | mean | 0.148381 | [0.0899281, 0.221429] |
| redirect | wrong | 5 |  | mean | 0.0197842 | [0.0133929, 0.0264599] |
| defer\|redirect D_S<0 (conflict) | correct | 15 |  | mean | 0.14191 | [0.107437, 0.185453] |
| defer\|redirect D_S>=0 | correct | 15 |  | mean | 0.106673 | [0.0793351, 0.137841] |
| defer\|redirect D_S<0 (conflict) | wrong | 5 |  | mean | 0.239482 | [0.149007, 0.342857] |
| defer\|redirect D_S>=0 | wrong | 5 |  | mean | 0.140722 | [0.0906033, 0.198473] |

### Decoding order (fig11)

| quantity | group | n_rollouts | note | stat | value | 95% CI |
|---|---|---|---|---|---|---|
| overlap | correct | 15 |  | mean | 0.562541 | [0.52746, 0.596872] |
| overlap | wrong | 5 |  | mean | 0.519713 | [0.488393, 0.560469] |
| all | correct | 15 |  | mean | 0.199868 | [0.182826, 0.214896] |
| D_S<0 | correct | 15 |  | mean | 0.262599 | [0.223857, 0.300909] |
| D_S>=0 | correct | 15 |  | mean | 0.179104 | [0.162063, 0.19601] |
| all | wrong | 5 |  | mean | 0.205935 | [0.163321, 0.238393] |
| D_S<0 | wrong | 5 |  | mean | 0.236246 | [0.201923, 0.292517] |
| D_S>=0 | wrong | 5 |  | mean | 0.194271 | [0.147541, 0.235669] |
| spearman | correct | 15 |  | mean | 0.352438 | [0.31172, 0.39118] |
| spearman | wrong | 5 |  | mean | 0.346788 | [0.265839, 0.443386] |

## Files

- `summary.csv`: every number above
- `tokens.csv.gz`: one row per revealed token (all runs)
- `steps.csv.gz`: one row per decoding step
- `rollouts.csv`: one row per rollout with question/gold/prediction/completion
- `fig*.png`: figures
