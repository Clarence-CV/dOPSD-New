# Selective Draft OPSD 首轮真机运行（MATH）

本文件是这次运行的唯一记录：设计、预算、提交信息、检查清单、结果与问题都写在这里。自动生成的数字报告在 `$SCRATCH/dopsd/selective_draft/<RUN>/REPORT.md`（备份 `$WORK/inf385t/results/selective_draft/<RUN>/`）。

## 0. 版本

| RUN | job | 状态 | 说明 |
|---|---|---|---|
| math_prelim_v1 | 3510902 | 2026-10-14 排队中取消，未运行，0 SU | 512 tokens、阈值解码为主、dev 评测。用户要求对齐 THU d-OPSD 的训练与评测协议后作废 |
| math_prelim_v2 | 3511305 | 2026-10-14 排队中取消，未运行，0 SU | 误用了 THU 的固定预算解码；用户要求固定阈值 |
| math_prelim_v3 | 3511314 | 完成（2026-10-15），8.60 SU | 固定阈值解码 + THU 的其余训练/评测设置 |

## 1. 目的

这是初步测试，不追求定论。要回答四件事：

1. 新流水线在 LLaDA-8B 真模型上能否端到端跑通（parity、采集、两阶段训练、续跑、评测）。
2. L28 早期读出在 MATH 状态上的可靠性：原生 LM head 与训练后适配器的一致率、0.8/0.9 门控覆盖率与校准。
3. Teacher 目标是否有信号：mixture、argmax、realized 三种目标相对无条件 q⁰ 移动多少，方向是否一致；L24/L28 patch 作为机制探针。
4. 阶段二训练后，同一步提交 token 的一致性指标和准确率、步数、时延相对 base 的变化。

## 2. 与 THU d-OPSD 的对齐

核对来源：`d-opsd/opsd.yaml`、`d-opsd/run/math/opsd.sh`、`d-opsd/d_opsd_trainer.py`（rollout 与 pass@k 循环）、`d-opsd/d_opsd_config.py`、`d-opsd/utils.py::get_all_parsed_answer`、`eval/run/math/opsd.sh`。

| 项 | THU | 本次 |
|---|---|---|
| 训练题池 | `ankner/math-500` train（7500） | 同，按 seed 42 划出 train / dev / analysis（6000/500/500） |
| 生成长度 / block | 256 / 32 | 同 |
| 解码 | 固定预算 128 步（每步 2 token） | **不同：固定阈值**（末层概率 > 阈值的位置同步提交，至少 1 个，步数自适应）。这是方法设计，用户明确要求；THU 的准确率数字因此不能直接对比 |
| 训练 rollout | 温度 0.9（`temperature` 默认值），答错用 0.9 重采，每题最多 8 次，答对即停 | 同 |
| 8 次全错 | loss × 0，不产生信号 | 丢弃该题 |
| 每题 rollout 数 | 1 条有效 | 同 |
| 训练 gold | 解答中最后一个 `\boxed{}`，`is_equiv` 判等 | 同 |
| 评测 | MATH-500 test（HuggingFaceH4），greedy，256 tokens，固定预算 128 步 | MATH-500 test 前 200 题，greedy，256 tokens，固定阈值 0.7 / 0.8 / 0.9 |
| 学习率与优化器 | LoRA r=128：lr 5e-6、weight decay 0.1、梯度裁剪 0.2、beta2 0.99、constant_with_warmup | **不照搬**：本方法训练的是两个零初始化的新小模块，5e-6 在 100 步内几乎不会让注入消息离开 0。用 AdamW lr 3e-4、weight decay 0、裁剪 1.0；靠 `delta_norm / source_hidden_norm` 判断是否合适 |
| 主干 | 4-bit + LoRA | bf16 冻结主干，不加 LoRA（方法设计） |
| 混合精度 | bf16 权重 + fp16 autocast | bf16，不开 autocast（阶段二要反传，fp16 无 loss scaling 有溢出风险） |
| Prompt | 训练用 GSM 风格系统提示、评测用 MATH500 提示 + `<reasoning>` | 训练与评测统一用评测 prompt |
| 损失 | top-20 JSD + clip、future PI teacher | 本方法：全词表 forward KL，realized reveal teacher |
| 停止 | 完整解码 256 个位置 | block_end：EOS 所在 block 解完即停。EOS 之前的输出与完整解码相同，只减少前向次数 |

阶段一的 train 状态、analysis 诊断状态仍用 greedy 采集：诊断里的反事实续跑需要确定性。

## 3. 设置

| 项 | 值 |
|---|---|
| 模型 | GSAI-ML/LLaDA-8B-Instruct，revision 08b83a6f，bf16，冻结；新增模块约 131 万参数 |
| 判分 | 最后一个 `\boxed{}`，否则 `<answer>`；`eval/parser_helper.is_equiv`。无 `\boxed` 时不再把整段文本当作预测 |
| 指导 | L28（blocks[27] 输出），门控 max r_A ≥ 0.8，full-soft 消息，A 不锁定、不占提交额度 |
| 阶段一 | 80 道 train 题的 greedy base 状态（batch 4 采集），800 次更新 × 4 状态，lr 3e-4 |
| 阶段二 | 100 次更新；每次 2 道题，每题按 THU 规则最多采 8 条直到答对；每条有效 rollout 最多取 4 个门控状态；Teacher = realized；每 25 步 checkpoint |
| 诊断 | analysis 64 题（batch 1 采集）；readout（原生 vs 训练后）、teacher（L28 训练后 / L28 原生 / L24 原生，各 150 状态）、consistency（base vs guided） |
| 解码 | 固定阈值：采集与训练 rollout 用 0.9；评测扫 0.7 / 0.8 / 0.9 |
| 评测 | MATH-500 test 前 200 题，batch 1 计时；base 与 guided 各扫 3 个阈值；readout_only 在阈值 0.9 下跑前 64 题测早期读出开销 |

## 4. 预算

队列 `gpu-a100-small`（1×A100-40GB，1.5 SU/节点·小时；qlimits 2026-10-14 复核：1 节点、48 h 上限）。

| 阶段 | 估计（分钟） |
|---|---|
| 元数据、数据划分、inspect、parity smoke | 8 |
| 采集 train 80 + analysis 64 | 15 |
| 阶段一 + readout 诊断 | 12 |
| teacher 三组 + consistency base | 12 |
| 阶段二 100 次更新（每题平均约 3 条 rollout） | 65 |
| guided 采集 + consistency | 13 |
| 评测：base / guided 各 3 个阈值 × 200 题、readout_only 64 题 | 110 |
| 约 18 次模型加载 | 25 |
| **合计** | **约 4.3 小时** |

估计基于每次前向约 0.07 秒（约 500 tokens），每条 256-token 阈值解码约 80 步、约 6 秒，MATH 上正确率按三到四成估。申请 5:45（约 1.3 倍），**最多约 8.6 SU**。阶段二的耗时取决于温度 0.9 下的正确率，是最大的不确定项；超时则用同一个 RUN 重新提交续跑。

## 5. 运行方式

```bash
RUN=math_prelim_v3 bash tacc/sync_and_submit_math.sh   # 本地；DRY=1 只同步不提交
```

## 6. 拿到结果后先看什么

1. `stages.tsv`：哪些阶段 ok / FAIL，各自耗时，用来校准下次预算。
2. Parity smoke：readout_only 与零初始化注入必须与 off 逐字一致。
3. `meta/model_revision.txt`：实际 snapshot 是否等于 pinned revision。
4. Readout 表：原生 lens 与训练后适配器的 top-1 一致率、门控覆盖率、`source_agree_given_gate_0.8`。
5. Teacher 表：三种目标 `kl_*_vs_q0` 的量级与差距；`vstar_eq_realized`（A 最终改变的频率）；oracle 恢复率 L24 vs L28。
6. 阶段二：loss 走势、`delta_norm / source_hidden_norm`、`attempts_per_problem` 与有效题比例、`reselect_mismatch_total`（应接近 0）。
7. 评测：每个阈值下 guided vs base 的配对对比（各自独有的答对题数）、steps/row 与时延，以及准确率随阈值的变化；readout_only vs base 的时延差即早期读出开销。
8. Consistency：base 与 guided 的 psa、pmi、regret、induced_fidelity，按 doc/Selective_Draft_OPSD_Review.md §7 的收紧口径解读。

## 7. 提交记录与结果

| 项 | 值 |
|---|---|
| job | 3511314，gpu-a100-small，申请 5:45:00 |
| 排队 | 2026-10-14 23:10 → 2026-10-15 06:48（约 7.6 h，不计费） |
| 运行 | 06:48:42 → 12:32:38，Elapsed 05:43:56（20636 s），COMPLETED 0:0；离申请上限只剩约 1 分钟 |
| 计费 | 1 × 5.732 h × 1.5 = **8.60 SU** |
| 输出 | `$SCRATCH/dopsd/selective_draft/math_prelim_v3/`；本地 `outputs/math_prelim_v3/`；报告副本 `results/selective_draft/math_prelim_v3_REPORT.md` |
| 峰值显存 | 阶段二 18.1 GB，评测 16.6 GB（40 GB 卡） |

阶段耗时与估计的差距：阶段二 143 分钟（估 65），因为温度 0.9 下每题平均要采 4.7 条（估 3）；inspect 和 smoke 各约 7–8 分钟，基本是模型加载。

### 结论

1. **流水线在真模型上端到端跑通。** 18 个阶段全部成功；parity 3/3 逐字一致，readout_only 在 64 道评测题上与 base 逐字一致（64/64）；阶段二源位置重选不一致 0 次；日志无报错。
2. **早期读出适配器有效。** analysis 状态上，与末层 argmax 的一致率：

   | 指标 | 原生 LM head | 训练后适配器 |
   |---|---|---|
   | 全部 MASK 位置 top-1 一致率 | 0.307 | 0.420 |
   | 门控 0.8 下源位置一致率 | 0.731 | 0.903 |
   | 门控 0.9 下源位置一致率 | 0.775 | 0.936 |
   | 门控 0.8 覆盖率 | 0.743 | 0.795 |

   校准大体单调，max r ≥ 0.9 的桶一致率 0.945。训练日志末段 top-1 一致率 0.644，高于 analysis 上的 0.420，可能是在 80 道题上过拟合，原因未查。
3. **L24 原生读出完全不开门。** 150 个状态里 max r 从未 ≥ 0.8，L24 的三种目标和 patch 探针没有数据。要测 L24 必须先训练 L24 的读出适配器。
4. **Teacher 目标：mixture 没有退化成 q⁰。** L28 训练后读出、门控开启的 123 个状态上，各目标相对 q⁰ 的 KL（nats，按 Receiver 平均）：mixture(MC) 0.172、top-K 0.148、argmax 0.163、realized 0.163；realized 与 mixture 之间 0.071。门控源的候选 v\* 与最终写入一致 90%。审阅文档 §2.2 的退化假设在这批数据上**不成立**：揭示一个模型已有 94% 把握的 token，Receiver 分布仍移动约 0.16 nats，说明模型的条件分布与边缘分布并不自洽，三种目标的差别反而很小。
5. **patch 探针**：L28 只替换 h_A，原生尾层恢复 realized 条件响应的 16–18%，与 GPTRecord 的 19% 一致。
6. **阶段二训练动起来了，消息幅度偏大且仍在增长。** loss 0.24 → 0.17（波动大）；`delta_norm / source_hidden_norm` 从 0.007 涨到 0.32，最后 20 步仍在涨；指导后 A 自身的末层置信度略降（0.844 → 0.834）。240 道题、1130 条 rollout，只有 131 道在 8 次内答对；20 次更新因两道题都全错而跳过。
7. **评测：guided 没有提升准确率，各阈值都略低于原始 LLaDA，单个阈值都不显著。**

   | 阈值 | base 准确率 | guided 准确率 | 只有 base 对 | 只有 guided 对 | McNemar p | Δ 步数/题 | Δ 时延/题 |
   |---|---|---|---|---|---|---|---|
   | 0.7 | 0.375 | 0.345 | 15 | 9 | 0.31 | −1.4 | +0.23 s |
   | 0.8 | 0.390 | 0.380 | 12 | 10 | 0.83 | −2.0 | +0.26 s |
   | 0.9 | 0.410 | 0.370 | 15 | 7 | 0.13 | −1.7 | +0.34 s |

   步数每题少约 2%，但早期读出与消息的开销（readout_only 对 base 约 +0.32 s，约 6%）大于省下的步数，实际时延反而更长。base 自身随阈值降低：0.9 → 0.7 步数少 33%，准确率掉 3.5 个点。
8. **同步一致性在阈值 0.9 下没有可改善的空间。** 150 个状态里只有约 40 个有 ≥2 个同步提交；base 的 psa = 1.0、regret = 0，揭示 A 后的 log-prob 变化均值约 0。guided 产生 13 个"消息诱导的提前提交"，全部与关闭指导的续跑写入相同 token（fidelity 1.0），平均省 1.5 步。

### 问题与下一步建议

- **0.9 阈值下几乎没有协调失败可修**（结论 8），收益只能来自更低阈值。下轮的 consistency 诊断应在 0.7 / 0.8 的状态上做，并以此为主协议。
- **消息幅度可能过大**（结论 6）：相对范数 0.32 且未收敛，同时准确率略降。可选：学习率降到 1e-4 或更低、对 delta 加范数惩罚或可学习的缩放上限、或训练更久看是否回落。
- **阶段二数据效率低**：温度 0.9 下 MATH 只有 55% 的题能在 8 次内答对，每题平均 4.7 条 rollout，占了作业 40% 的时间。
- **三种 Teacher 目标差别小**（结论 4），选哪个对首轮结果影响不大；更值得查的是消息为什么让准确率下降。
- **样本量**：200 题下单个阈值的差异都不显著，三个阈值方向一致但并非独立样本。
- **L24**：需先训练 L24 读出适配器才能比较 L24 和 L28。
- **流程**：模型每阶段重载一次，19 次加载约 25 分钟；把多个阶段合进一个进程可以省下来。显存峰值 18 GB，评测还能加大 batch。
