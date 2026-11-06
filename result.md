# 最终结果：Selective Draft / Joint-reference OPSD（soft draft 方案）

日期：2026-11-05。本文件是 INF 385T（Post-Training of LLMs）课程的最终汇报。这条方案到此**正式结束**。

**一句话结论**：用"早层 soft draft 读出 + 注入器"让同一步并行提交的 token 互相看到对方的草稿，再用留一（LOO）的 Teacher 做 on-policy 自蒸馏。在两个模型上，准确率都只有不显著的小幅上升，LLaDA2.0-mini 上的加速也很小：MATH-500 准确率 74.8 → 76.2%（p=0.25），步数少 4.9%，延迟没有变化。

## 1. 研究问题

掩码扩散语言模型（dLLM）在同一步里并行提交多个 token 时，每个 token 都是按自己的边缘分布独立选出来的，最终预测并不以同伴这一步实际选中的 token 为条件。我们想做的是：让同一步的 token 之间先"通个气"，改善它们的协调性。判断标准是准确率和实际生成时延。

## 2. 方法

```text
真实去噪状态 S → 冻结主干到第 l 层
  → readout adapter → 原 final norm → 共享 LM head → 草稿分布 r_i
  → 按置信度选出组 G；soft message m_i = sum_v r_i(v) * E[v]
  → h'_i = h_i + Injector(h_i, m_i)（所有消息同时注入）
  → 冻结的剩余层 → 最终 logits → 正常阈值解码
```

1. **阶段一（readout）**：只训练读出器，目标是 `KL(q0 || r)`，让第 l 层就能读出接近末层的分布。旁路不改变原模型的输出（parity 已验证）。
2. **阶段二（Joint）**：冻结主干和 readout，只训练注入器。训练时从 prompt 出发 rollout（T=0.9，最多 8 次，首次答对即停），从答对的轨迹里抽一个有同伴组的状态。Teacher 揭开组内其他位置的参考 token，第 i 个位置保持 MASK（LOO），所有位置都通过 `argmax(q_i) == pi_i` 认证后才拿来训练。损失是完整词表上的 forward KL。

详细设计见 [doc/Selective_Draft_OPSD.md](doc/Selective_Draft_OPSD.md)、[doc/llada2-mini_engineering_plan.md](doc/llada2-mini_engineering_plan.md)；实现在 `selective_draft/`。

## 3. 三轮实验汇总

| 轮次 | 模型 / 评测 | 准确率 base → ours | 速度 | 显著性 | SU |
|---|---|---|---|---|---|
| Selective Draft v3（job 3511314） | LLaDA-8B，单源 soft message，MATH-500 前 200 题，阈值 0.9 | 41.0 → 37.0% | 步数 −2%，延迟 +6% | p=0.13（变差） | 8.60 |
| Joint v1（job 3515124） | LLaDA-8B，组大小 4，MATH-500，阈值 0.9，256 tok | 34.4 → 35.0% | 步数 −18%，延迟 −11% | p=0.80 | 16.61 |
| **LLaDA2-mini Joint v3（job 3545392）** | LLaDA2.0-mini，组按置信度 > 0.8 选，MATH-500，阈值 0.95，2048 tok | **74.8 → 76.2%** | 步数 −4.9%，TPF 2.87 → 2.95，延迟持平 | p=0.25 | 68.03 |

v1 的详细记录见 [results/joint/math_v1.md](results/joint/math_v1.md)，v3 的详细记录见下文第 4 节。

## 4. LLaDA2-mini Joint v3（job 3545392）

### 设置

- **模型**：`inclusionAI/LLaDA2.0-mini`（revision dad945c），使用 fused MoE（DMax/VeOmni kernel）。
- **解码**：block 32，greedy，置信度阈值 0.95，最多 2048 个新 token，DMax 的 boxed prompt 和 scorer。
- **readout**：放在第 17 层，这是手动指定的（自动规则选的是 19 层）；rank 64，训练 200 步。
- **Joint**：注入器 rank 64，lr 1e-4，200 次更新 × 3 卡，每 25 步存一个 checkpoint。
- **选检查点**：在 dev_fast 128 题上选准确率最高的；平手时选平均步数最少的，再平手选最早的。
- **最终评测**：MATH-500 test 全部 500 题，base 和选中的检查点做配对比较。
- **算力**：LS6 `gpu-a100`，1 节点 3×A100，实际运行 22.68 h，计 68.03 SU。
- **代码**：commit b73929a（运行时是 ea8496f 加上未提交的修改，这些修改随后作为 b73929a 提交）。

### dev_fast 128 题检查点扫描

| 检查点 | 准确率 | 平均步数 | TPF | 截断率 | 中位延迟 (s) |
|---|---:|---:|---:|---:|---:|
| base | 78.9 | 377.0 | 2.92 | 21.9% | 11.17 |
| joint 25 | 78.9 | 391.5 | 2.92 | 23.4% | 12.14 |
| **joint 50（选中）** | **78.9** | **370.6** | **2.96** | 22.7% | 11.31 |
| joint 100 | 78.1 | 366.9 | 3.05 | 23.4% | 11.06 |
| joint 150 | 75.0 | 347.7 | 3.25 | 23.4% | 11.20 |
| joint 200 | 76.6 | 345.8 | 3.38 | 25.8% | 9.80 |

训练越往后越快（step 200 时 TPF +16%），但准确率掉了 2–4 个点。这是拿准确率换速度，并不是协调性变好了。

readout-only（只挂读出器、不注入）在前 32 题上与 base **逐字相同**（27/32 对，平均步数完全一致），parity 成立。日志里那个 84.4% 只是因为前 32 题比较简单。

### MATH-500 test（配对比较，500 题）

| | base | joint step 50 |
|---|---:|---:|
| 准确率 | 74.8% | **76.2%** |
| 修好 / 改错 | | 17 / 10（McNemar p=0.25） |
| 平均步数 | 390.0 | 371.0（−4.9%） |
| 平均 NFE | 400.6 | 382.1 |
| TPF | 2.87 | 2.95 |
| 截断率（触及 2048 上限） | 25.8% | 24.6% |
| 平均延迟 (s) | 27.78 | 27.73 |

作为参照：同一套设置下，用 HF 原版 MoE 循环得到的 base 是 75.9%（job 3526876，494 题，[results/llada2_math_v2/baseline_summary.md](results/llada2_math_v2/baseline_summary.md)），DMax 报告的是 75.8%。本次 base 用的是 fused MoE，与 HF 循环并非逐比特一致，所以只和本次 run 自己的 base 做配对比较。

### 训练诊断

- **readout**：确认集上选中的是 step 150，但没有达到参考水平（agreement、fidelity 两项未通过，只记录不拦截）。L17 的读出质量本身就有限。
- **Joint**：共 615 次题目访问，认证通过 420 个状态。以 rank0 为例：205 题里有 179 题至少一次答对，抽出 179 个状态，其中 37 个被 LOO 拒绝，6 个参考无效，136 个通过；每次更新平均只有约 2 个有效状态。
- **训练曲线**：loss 没有明显下降（每 40 步段的均值依次为 0.33、0.47、0.38、0.40、0.38、0.98），grad_norm 从 0.46 一路涨到 1.79。同时注入幅度 delta_norm 的每 40 步均值从 3.1 涨到 77.8；组外 KL 从 0.001 涨到 0.032；组外 token 被改动的次数从每个状态 0.04 次涨到 0.49 次。这和 LLaDA-8B 上 v1 的情况一样：注入器越训越"响"，但没有收敛到更好的协调。

## 5. 为什么放弃

1. 在两个模型、三轮实验里，准确率变化都在噪声范围内（p 值 0.13 / 0.80 / 0.25），方向也不一致。
2. 有速度收益的那一次（v1，−18%）出现在 LLaDA-8B 的 256 token、阈值 0.9 设定下；换到 LLaDA2-mini 的官方设定（阈值 0.95）之后，只剩 −5% 的步数、0 延迟收益。更激进的检查点能快 16%，但准确率要掉。
3. 主要瓶颈不在注入本身：早层读出质量有限（L17 没通过确认），可监督的状态少（每步约 2 个），而且 Teacher 目标几乎是 one-hot，本质上接近"复制已经答对的轨迹"。
4. 判断：soft draft 的思路以后可能还会用，但不会再用"早层读出 + 注入器 + LOO Joint 目标"这种形式。

## 6. 文件与数据

| 内容 | 位置 |
|---|---|
| 本次 run 的评测、日志、配置、选中的权重（git 内，31 MB） | [results/llada2_math_v3/](results/llada2_math_v3/) |
| 逐题输出（base / joint 选中 test 500） | `results/llada2_math_v3/eval/*_test_dmax_boxed/rows_thr0.95.jsonl` |
| 检查点选择 / 配对结果 | `results/llada2_math_v3/eval/joint_selection.json`、`paired_test_dmax_boxed.json` |
| 训练日志（每 rank） | `results/llada2_math_v3/{readout,joint}/train/log_rank*.jsonl` |
| 选中权重（readout step 150、joint step 50，含 config） | `results/llada2_math_v3/{readout/train/checkpoint-step-150,joint/train/checkpoint-step-50}/snapshot/` |
| SLURM stdout | `results/llada2_math_v3/logs/llada2-3545392.out` |
| 完整拷贝（含 450 MB 状态库和全部检查点，不在 git 内） | 本地 `outputs/llada2_math_v3/`；LS6 `$SCRATCH/dopsd/llada2/llada2_math_v3/`，备份 `$WORK/inf385t/results/llada2/llada2_math_v3` |
| 全部作业与 SU 记账 | [results/EXPERIMENTS.md](results/EXPERIMENTS.md) |

git 里没有放的部分：状态库 `states/`，以及非选中检查点的 optimizer/rng 文件。这些只用于断点续跑，复现评测不需要。
