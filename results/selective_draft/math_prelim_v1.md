# math_prelim_v1 — Selective Draft OPSD 首轮真机运行（MATH）

**状态：已设计，待提交（2026-10-14）。** 本文件是这次运行的唯一记录：设计、预算、提交信息、检查清单、结果与问题都写在这里。自动生成的数字报告在 `$SCRATCH/dopsd/selective_draft/math_prelim_v1/REPORT.md`（备份 `$WORK/inf385t/results/selective_draft/math_prelim_v1/`）。

## 1. 目的

这是初步测试，不追求定论。要回答四件事：

1. 新流水线在 LLaDA-8B 真模型上能否端到端跑通（parity、采集、两阶段训练、续跑、评测）。
2. L28 早期读出在 MATH 状态上的可靠性：原生 LM head 与训练后适配器的一致率、0.8/0.9 门控覆盖率与校准。
3. Teacher 目标是否有信号：mixture、argmax、realized 三种目标相对无条件 q⁰ 移动多少，方向是否一致；L24/L28 patch 作为机制探针。
4. 阶段二训练后，同一步提交 token 的一致性指标与准确率–NFE 曲线相对 base 的变化。

## 2. 设置

| 项 | 值 |
|---|---|
| 模型 | GSAI-ML/LLaDA-8B-Instruct，bf16，冻结；新增模块约 131 万参数 |
| 数据 | `ankner/math-500` 的 train split（7500 行，字段 problem/level/type/solution，无 answer 字段），按 seed 42 划出互不重叠的 train / dev / analysis（6000/500/500）；gold = 解答中最后一个 `\boxed{}`；test split（500 行）不碰 |
| Prompt | `eval/math500.py` 的系统提示逐字复制 + `<reasoning>` |
| 判分 | 最后一个 `\boxed{}`，否则 `<answer>`；`eval/parser_helper.is_equiv`。与旧评测的差别：无 `\boxed` 时不再把整段文本当作预测 |
| 解码 | 512 tokens，block 32，固定窗口，block_end 停止，greedy，阈值 0.9 为主协议 |
| 指导 | L28（blocks[27] 输出），门控 max r_A ≥ 0.8，full-soft 消息，A 不锁定 |
| 阶段一 | 80 道 train 题的 base 状态，800 次更新 × 4 状态，lr 3e-4 |
| 阶段二 | on-policy，每次更新 2 条 rollout × 每条最多 4 个门控状态，只用答对的 rollout，Teacher = realized，100 次更新，lr 3e-4，每 25 步存 checkpoint |
| 诊断 | analysis 64 题（batch 1 采集）；readout（原生 vs 训练后）、teacher（L28 训练后 / L28 原生 / L24 原生，各 150 状态）、consistency（base vs guided） |
| 评测 | dev 64 题，batch 1 计时；base 与 guided 各扫阈值 0.7/0.8/0.9/0.95；readout_only 只在 0.9 测开销 |

lr 从计划的 1e-4 提到 3e-4，因为只有 100 次更新，想先看到消息幅度是否会动起来；日志里有 `delta_norm / source_hidden_norm` 可直接判断。

## 3. 预算

队列 `gpu-a100-small`（1×A100-40GB，1.5 SU/节点·小时，CLAUDE.md B.3，抓取于 2026-09-20，提交前用 `qlimits` 复核）。选它是因为历史上 small 队列当天就能开始，`gpu-a100` 曾排队 48–57 小时。

| 阶段 | 估计（分钟） |
|---|---|
| 元数据、数据划分、inspect、parity smoke | 10 |
| 采集 train 80 + analysis 64 | 35 |
| 阶段一 + readout 诊断 | 13 |
| teacher 三组 + consistency base | 14 |
| 阶段二 100 次更新 | 60 |
| guided 采集 + consistency | 20 |
| 评测 base / guided 各 4 个阈值 + readout_only | 130 |
| 约 17 次模型加载 | 25 |
| **合计** | **约 5.2 小时** |

估计基于每次前向约 0.1 秒（约 750 tokens，8B，bf16），每条 512-token 阈值解码约 130 步、约 15 秒。申请 7:15（约 1.4 倍）。**最多 1 × 7.25 × 1.5 ≈ 10.9 SU**，低于 30 SU 的上限。如果实际前向慢一倍，评测阶段会被截断；所有阶段可续跑，用同一个 RUN 重新提交即可。

## 4. 运行方式

```bash
# 本地：同步代码到 $WORK/inf385t/dOPSD-sd（独立目录，不碰 THUdOPSD 的工作树）并提交
bash tacc/sync_and_submit_math.sh            # DRY=1 只同步不提交
# 续跑：同一 RUN 重新提交即可，已完成阶段跳过，训练从最新 checkpoint 恢复
```

## 5. 拿到结果后先看什么

按优先级：

1. `stages.tsv`：哪些阶段 ok / FAIL，各自耗时，用来校准下次预算。
2. Parity smoke：readout_only 与零初始化注入必须与 off 逐字一致；不一致说明 hook 改变了计算路径，后面全部作废。
3. `meta/model_revision.txt`：实际用的 LLaDA snapshot 是否等于 pinned revision。
4. Readout 表：原生 lens 与训练后适配器在 analysis 状态上的 top-1 一致率、门控覆盖率、`source_agree_given_gate_0.8`。
5. Teacher 表：三种目标的 `kl_*_vs_q0` 量级与差距；`vstar_eq_realized`（门控源的候选与最终写入是否一致，就是"A 最终改变"的频率）；oracle 恢复率 L24 vs L28。
6. 阶段二：loss 是否下降，`delta_norm / source_hidden_norm` 是否离开 0，`reselect_mismatch_total` 应接近 0，答对率与跳过次数。
7. 评测：同阈值下 paired 对比（base only / guided only），以及 steps/row、latency 的变化；readout_only 与 off 的时延差就是早期读出的开销。
8. Consistency：base 与 guided 的 psa、pmi、regret、induced_fidelity，按 doc/Selective_Draft_OPSD_Review.md §7 的收紧口径解读。

## 6. 已知局限

- 64 道 dev 题的准确率标准误约 6 个百分点，只能看大方向；paired 对比比绝对值可靠。
- 阶段二只有 100 次更新，主要验证"能训、能动"，不是收敛结果。
- train 状态是 batch 4 采集的，padding 下 bf16 数值与 batch 1 有微小差异；只影响阶段一的训练数据，不影响诊断。
- realized 目标下，A 在本步未提交时用的是它之后实际写入的 token。

## 7. 提交记录与结果

（提交后填写：job ID、实际开始/结束、`sacct` 计费、结论与问题。）
