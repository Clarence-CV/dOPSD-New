# dLLM OPSD 开源仓库（检索于 2026-11-10，仅 dLLM）

结论：公开代码、且是"同一模型 + 特权信息 teacher"的推理任务自蒸馏，dLLM 上只有两个：THU d-OPSD（LLaDA）和 NUS dOPSD（Dream）。两者都没有 license、没有公开日志/wandb/checkpoint。其余 dLLM 自蒸馏仓库都是加速蒸馏或外部 teacher。

## A. THU d-OPSD（论文 A）

- 仓库 github.com/xingzhejun/d-opsd-code（本仓库 remote `thu`，分支 `THUdOPSD`）；arXiv 2606.18195 "Learning from the Self-future"；README 自称 NeurIPS 2026（arXiv 仍标 preprint，未独立核实）。
- TRL GRPOTrainer 分支（d1 系）；需改装 `trl/trainer/grpo_trainer.py` 第 424 行和 `trl/trainer/utils.py`（仓库提供替换文件）。LLaDA-8B-Instruct 4-bit + LoRA r128/α64。任务 GSM8K / MATH500 / Countdown / Sudoku。`accelerate.yaml` num_processes=4（LS6 需改 3）；README：A100/H100 需 `BATCH_DIVIDE=8`。
- 解码：每步 2 token（diffusion_steps = 长度/2，block 32），代码断言每步正好 2 个监督位置。
- 原文动力学：OPSD "prone to policy collapse after achieving peak performance"；不裁剪的 run 约 step 150 开始崩，step 500 降到 69.37；每 25 步评测取最优。消融（GSM8K）：裁剪 81.0 vs 不裁剪 77.0（Tab. 9）；冻结 teacher 81.0 vs 当前 79.7（Tab. 10）；只用正确 81.0 vs 全部 80.3（Tab. 11）。
- **teacher 默认是冻结的**：`opsd.yaml` 写 `fixed_teacher: False`，但 `run/*/opsd.sh` 设 `fixed_teacher=True`；`d_opsd_trainer.py` 约 253 行 `disable_adapter()`。所以原文的崩溃大概率在冻结 teacher 下发生（本地核实）。`sync_ref_model`/`ref_model_sync_steps: 64` 是 GRPO 遗留，大概率未用。
- 开关：`--fixed_teacher`；`--jsd_token_clip`（数学 0.04，默认 0.05，`generalized_jsd_loss` 约 150–195 行，记录 `clip_ratio`、`top_k_overlap`）；`--top_k_loss 20`；`--beta`（1 = reverse KL）。
- 过滤：最多 `--passk 8` 次、`--passk_temperature 0.9` 重采直到 `get_all_parsed_answer` 判对（约 450–525 行）；错的 loss 乘 0（约 366 行）。评分器：`d-opsd/math500_utils.py`、`reward_func.py`、`utils.py`；评测 `eval/parsers.py`、`eval/parse_and_get_acc.py`（d1 的 Hendrycks 字符串匹配，无截断规则，见 `doc/lit_graders_dllm_rl.md`）。
- PI：student 自己的最终答案，揭示 `--teacher_retain_ratio 0.25`；`--add_ref` 换成标准答案。
- 已记录：`completion_length`、`iter_num`（找到正确 rollout 需要的次数，H1/H4 代理）。
- 需补丁：每 N 步刷新 teacher（周期性快照 adapter）。

## B. NUS dOPSD（论文 B）

- 仓库 github.com/tuandattt/dOPSD；arXiv 2607.04428；无 venue。本仓库最初即 fork 自它（文件结构相同），但它是 NUS 的代码与论文。
- TRL GOLD trainer + DeepSpeed ZeRO-2。Dream-v0-Instruct-7B（默认，sdpa）或 LLaDA-8B（`STUDENT_BACKEND=llada`）。训练数据 `horseee/MixChain-Z-PRM12K`；评测 GSM8K / MATH500 / HumanEval / MBPP。原文 8 卡、LoRA r32、lr 2e-5、3 epoch。
- 解码：每步 1 token（gen_steps = gen_max_new_tokens = 256），`--gen_temperature 0.2`。
- 原文动力学：没有自身的准确率–步数曲线（报最优 ckpt）；Fig. 2(b) off-path 消融"steadily degrades GSM8K"，文中说 full-solution OPSD "collapses"；验证消融（Dream 四任务均值）60.11 有验证 / 58.64 无 / 57.59 原模型；forward KL 全面优于 reverse。
- **teacher 默认是当前模型**（`--fixed_teacher` 默认 False，需 `--use_peft`）。`--traj_teacher_gap`、`--traj_teacher_view snapshot|all_future` 控制 teacher 看多远。
- 过滤：`--filter_wrong_rollouts`（默认开）、`--use_grpo`（默认开）、`--grpo_num_rollouts`、`--grpo_coef`；答案检查在 `data_collator_dllm_trajectory.py` / `opsd_dllm_trajectory_trainer.py`。裁剪 `--jsd_token_clip` 默认 0（关）。
- 入口 `opsd_dllm_trajectory_train.py`，启动 `scripts/run_opsd_dllm_trajectory_7b.sh`（环境变量旋钮）。

两篇互补：底座 LLaDA vs Dream；每步 2 token vs 1 token；默认 reverse vs forward KL；默认冻结 vs 当前 teacher。

## 其它 dLLM 自蒸馏（不用于 H1–H4）

| 仓库 / 论文 | 情况 | 不适用原因 |
|---|---|---|
| T3D（github.com/Tyrion58/T3D，2602.12262，自称 NeurIPS 2026） | 轨迹自蒸馏，SDAR-4B/1.7B，有 ckpt 与轨迹数据 | 离线 teacher 轨迹，少步加速，非特权信息 OPSD |
| dParallel（czg1225/dParallel，ICLR 2026，MIT） | certainty forcing，LLaDA-8B / Dream-7B | 加速方法；本仓库 `eval/` 即 fork 自它 |
| CForce（inclusionAI/dFactory，2608.13925） | 并行解码一致性 | 未找到 CForce 具体文件；16B |
| TraceRL（Gen-Verse/dLLM-RL，ICLR 2026） | 轨迹 RL | RL 非自蒸馏，可作 GRPO 基线 |
| SelFusion（2608.22898，ACL 2026） | 高/低掩码视图互蒸馏 | 非 on-policy，无特权 teacher |
| OPTD 2608.02942、TSD 2609.15177、TOPD 2607.16872、Hesitation-Aware OPD 2609.33301 | — | 无代码（后两者外部 teacher） |

未核实：THU 的 NeurIPS 录用、GPU 型号、Fig. 12 数值；两仓库能否在 3×A100-40GB / 1×GH200 跑；CForce 在 dFactory 的位置。
