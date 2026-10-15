# Selective Draft OPSD 首轮真机运行（MATH）

本文件是这次运行的唯一记录：设计、预算、提交信息、检查清单、结果与问题都写在这里。自动生成的数字报告在 `$SCRATCH/dopsd/selective_draft/<RUN>/REPORT.md`（备份 `$WORK/inf385t/results/selective_draft/<RUN>/`）。

## 0. 版本

| RUN | job | 状态 | 说明 |
|---|---|---|---|
| math_prelim_v1 | 3510902 | 2026-10-14 排队中取消，未运行，0 SU | 512 tokens、阈值解码为主、dev 评测。用户要求对齐 THU d-OPSD 的训练与评测协议后作废 |
| math_prelim_v2 | 见 §7 | 当前 | THU 协议，见下 |

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
| 生成长度 / 步数 / block | 256 / 128 固定预算 / 32 | 同 |
| 训练 rollout | 温度 0.9（`temperature` 默认值），答错用 0.9 重采，每题最多 8 次，答对即停 | 同 |
| 8 次全错 | loss × 0，不产生信号 | 丢弃该题 |
| 每题 rollout 数 | 1 条有效 | 同 |
| 训练 gold | 解答中最后一个 `\boxed{}`，`is_equiv` 判等 | 同 |
| 评测 | MATH-500 test（HuggingFaceH4），greedy，256 / 128 / 32 | 同（取前 200 题）；另加阈值 0.9 作为第二协议 |
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
| 评测 | MATH-500 test 前 200 题，batch 1 计时；base 与 guided × {固定预算 128 步, 阈值 0.9}；readout_only 固定预算前 64 题测早期读出开销 |

## 4. 预算

队列 `gpu-a100-small`（1×A100-40GB，1.5 SU/节点·小时；qlimits 2026-10-14 复核：1 节点、48 h 上限）。

| 阶段 | 估计（分钟） |
|---|---|
| 元数据、数据划分、inspect、parity smoke | 8 |
| 采集 train 80 + analysis 64 | 20 |
| 阶段一 + readout 诊断 | 12 |
| teacher 三组 + consistency base | 12 |
| 阶段二 100 次更新（每题平均约 3 条 rollout） | 105 |
| guided 采集 + consistency | 13 |
| 评测：固定预算 2 × 200、阈值 2 × 200、readout_only 64 | 110 |
| 约 19 次模型加载 | 25 |
| **合计** | **约 5.1 小时** |

估计基于每次前向约 0.07 秒（约 500 tokens），每条 256-token 固定预算解码 128 步约 9 秒，MATH 上 greedy 正确率按三到四成估。申请 6:45（约 1.3 倍），**最多约 10.1 SU**。阶段二的耗时取决于温度 0.9 下的正确率，是最大的不确定项；超时则用同一个 RUN 重新提交续跑。

## 5. 运行方式

```bash
RUN=math_prelim_v2 bash tacc/sync_and_submit_math.sh   # 本地；DRY=1 只同步不提交
```

## 6. 拿到结果后先看什么

1. `stages.tsv`：哪些阶段 ok / FAIL，各自耗时，用来校准下次预算。
2. Parity smoke：readout_only 与零初始化注入必须与 off 逐字一致。
3. `meta/model_revision.txt`：实际 snapshot 是否等于 pinned revision。
4. Readout 表：原生 lens 与训练后适配器的 top-1 一致率、门控覆盖率、`source_agree_given_gate_0.8`。
5. Teacher 表：三种目标 `kl_*_vs_q0` 的量级与差距；`vstar_eq_realized`（A 最终改变的频率）；oracle 恢复率 L24 vs L28。
6. 阶段二：loss 走势、`delta_norm / source_hidden_norm`、`attempts_per_problem` 与有效题比例、`reselect_mismatch_total`（应接近 0）。
7. 评测：两种协议下 guided vs base 的配对对比（各自独有的答对题数）、steps/row 与时延；readout_only vs base 的时延差即早期读出开销。
8. Consistency：base 与 guided 的 psa、pmi、regret（固定预算下不计算 induced_fidelity，它只支持 greedy 阈值轨迹），按 doc/Selective_Draft_OPSD_Review.md §7 的收紧口径解读。

## 7. 提交记录与结果

（提交后填写。）
