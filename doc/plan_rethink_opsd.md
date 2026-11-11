# Rethinking On-Policy Self-Distillation for Diffusion LLMs

> 计划草案，2026-11-10。结构：分析 → 结论 → 设计原则 → 方法。只讨论 dLLM。
> 文献与仓库细节：`doc/lit_opsd_repos.md`、`doc/lit_eagle_spec_decoding.md`、`doc/lit_sibling_remask.md`。

## 0. 术语

- **状态 S**：解码中的半成品（prompt + 已提交 token + 其余 MASK）。
- **Teacher / Student**：同一个网络。teacher = 冻结的原始模型（LoRA 关），student = 原始模型 + LoRA。
- **PI**：teacher 比 student 多看到的信息。
- **候选组 C(S)**：student 在 S 上 p ≥ 1/2 的位置，ĉ_i 为其 argmax。与旧的 early-readout draft 无关（该方向已放弃）。
- **sibling**：同一步一起提交的 token。

---

## 1. 核心论点

dLLM 上的 OPSD 先升后崩，是结构问题：**监督标签经由 PI 依赖于 student 自身，训练消耗的是产生监督的信息差。** 只冻结 teacher 权重不够。据此，并行解码训练应当让一个**独立于 student 的判定者**去**审核** student 在**它自己会走到的状态**上的并行决策——这是投机解码的结构，而不是自蒸馏的结构。

---

## 2. 分析：OPSD 的监督为什么越训越差

### 2.1 对象

dLLM 上公开代码的特权信息自蒸馏的两篇：

| | THU d-OPSD（2606.18195） | NUS dOPSD（2607.04428） |
|---|---|---|
| 底座 / 每步提交 | LLaDA-8B / 2 token | Dream-7B / 1 token |
| teacher 权重 | **冻结**（启动脚本默认） | 当前模型 |
| PI | student 自己的最终答案（揭示 25%） | student 自己轨迹的未来步 |
| 原文报告 | 不裁剪约 step 150 开始崩 | 报最优 ckpt，无训练曲线 |

### 2.2 统一视角

OPSD 的监督信号 = teacher 与 student 在同一位置上的分歧，而分歧只来自 PI。两篇的 PI 都是 **student 自己生成的内容**，所以信号的大小和方向都是 student 参数的函数。

### 2.3 三个待验证的结论

**F1 信号是消耗品。** PI 有价值，是因为 student 的 rollout 里存在比它的众数更好的路径（pass@k > pass@1）。蒸馏把 student 推向这些路径，同时消耗掉 pass@k − pass@1；信息差耗尽后，teacher–student 分歧趋于零。
*可观测*：pass@1、pass@8、teacher–student KL 随训练步。*预期*：KL 单调衰减，pass@8 先于 pass@1 下降。

**F2 冻结权重不能冻结标签。** THU 冻结了 teacher 权重，仍然崩溃。因为标签还经由 PI 随 student 移动：teacher 看到 student 的答案后去预测 student 的 token，本质是自我确认。
*干预*：①权重：当前 / 每 N 步刷新 / 冻结；②PI 来源：student 的 rollout vs 冻结原模型事先采好的 rollout。*预期*：只冻结权重仍崩；权重与 PI 都独立于 student 才不崩，代价是峰值被原模型锁住——即"稳定性–上限"的权衡曲线，回应 THU 的 future work。

**F3 答案级过滤只约束终点。** 信号耗尽后，梯度主要沿过滤器看不见的方向（长度、重复、格式）推动 student 走向低熵。
*可观测*：输出长度、重复率、截断率随训练步，以及它们与 F1 中 KL 衰减点的先后关系。*预期*：漂移在信号衰减之后加速。

### 2.4 分析实验（两篇各做）

| 编号 | 内容 | 检验 |
|---|---|---|
| A1 | 原版复现到崩溃之后，每 25 步记录上述可观测量 | F1、F3 |
| A2 | 权重 × PI 来源的两条干预 | F2 |
| A3 | F2 不变，抬高 rollout 多样性（温度 / 次数），看崩溃是否推后 | F1 的因果性 |

---

## 3. 从分析到设计

| 结论 | 原则 | 对方法的修改 |
|---|---|---|
| F2 标签经由 PI 跟随 student | **标签由独立于 student 的判定者给出** | teacher 冻结；teacher 不复制 student 的输出，而是**判断** student 的提议，不同意即否决 |
| F1 信号来自 student 的探索并被耗尽 | **不监督 student 本该自己探索的东西** | 不教"答什么"，只教"这一步能一起定几个"；内容始终由原模型决定 |
| F3 过滤只约束终点 | **监督单元与要学的决策对齐** | 位置级判定取代答案级过滤，不需要标准答案 |

此外采纳投机解码文献（EAGLE-3、HASS、Nemotron）的共同做法：**状态 on-policy，标签冻结**——在 student 自己会走到的状态上，由冻结的 target 给标签，既修正曝光偏差，又不引入反馈。

---

## 4. 方法

### 4.1 设定（精确到每个数值）

**解码器（训练与评测共用，dInfer 规则）**：块长 32，只在当前块内解码；每步把置信度 ≥ τ 的 MASK 位置一起提交为其 argmax，置信度 = 对 logits 做 softmax（温度 1）后的最大概率；若没有位置 ≥ τ，提交置信度最高的那一个；T = 0（贪心）。最大生成长度：LLaDA2-mini 2048，LLaDA-8B / Dream-7B 按各自上限。

**Student**：原模型 + LoRA（r = 32，attention 的 qkv 与 dense，lr 2e-5）。有两个用途，设置不同：

| 用途 | 设置 |
|---|---|
| **产生训练状态** | 用上面的解码器、**τ = 0.95、T = 0**（与部署完全相同），每题解一条轨迹，沿途随机截取 8 个中间状态 S。状态每 N = 100 次更新用当前 student 重新采一批（on-policy 刷新）。 |
| **给出候选组** | 在 S 上一次前向，p_i = softmax(logits_i)。C(S) = {i ∈ 当前块的 MASK 位置 : max p_i ≥ 1/2}，ĉ_i = argmax p_i。1/2 的含义：只有过半数的 argmax 才算一个明确的提议；低于 1/2 的位置 student 自己都没有候选。不是调出来的数。 |

**Teacher**：同一个网络、LoRA 关闭，训练全程不变。只做一件事：对 S ⊕ ĉ_{C∖i}（S 里把候选组中除 i 以外的位置填上各自的 ĉ）做一次前向，取位置 i 的 q_i = softmax(logits_i)（温度 1，不采样）。整个候选组的 LOO 只需一次 teacher 前向（`sibling_opsd.loo_teacher` 已实现）。

**PI**：teacher 看到的 S ⊕ ĉ_{C∖i} 比 student 看到的 S 多出来的就是候选组里其它位置的 ĉ。teacher 不看答案、不看轨迹的未来。

**刹车目标 t_i 的估计**：从 student 的 p_j（j ∈ C∖i，温度 1）独立采 K 组 c_{−i}，每组一次 teacher 前向，取 q 的平均；默认 K = 1（损失对 t 线性，单样本无偏）。

**每次更新**：1 题 × 8 个状态 = 8 个 (S, C, V, t) 组；损失见 4.2；AdamW，grad clip 1.0，warmup 10 步。

**部署**：同一个解码器，T = 0，τ ∈ {0.95, 0.9, 0.8, 0.7}，无任何额外模块。

### 4.2 目标

student 在 S 上给出候选组 C(S)。把其它位置的 ĉ 填入 S，问 teacher：位置 i 你还选 ĉ_i 吗？

  V(S) = { i ∈ C(S) : argmax_v q(v | S ⊕ ĉ_{C∖i}, i) = ĉ_i }

  L = (1/|V|) Σ_{i∈V} [ log τ − log p_i(ĉ_i) ]₊ + (1/|C∖V|) Σ_{i∈C∖V} KL( t_i ‖ p_i )

- **油门**（V）：teacher 看到其它候选后仍选 ĉ_i，这一组在 i 上相容。把 p_i(ĉ_i) 推到 τ 即停：不改变 argmax，只让它这一步就过阈值。
- **刹车**（C∖V）：teacher 看到其它候选后改了主意。拉回 t_i(·) = E_{c_{−i}~p}[ q(·|S ⊕ c_{−i}, i) ]，即 teacher 对 sibling 取平均后的看法，使 student 在冲突位置不比原模型更自信。
- 单用边缘化 KL 不能加速：由全概率公式 t_i 约等于 teacher 自己的边缘，只会把 student 拉回原模型的置信度。加速必须来自油门这个独立的决策项。
- 无新的连续超参；τ 是部署阈值。

### 4.3 为什么能加速，为什么预期不掉点

- **加速**：V 中的 token 从后续步骤提前到当前步骤提交。
- **不掉点（预期）**：油门只推 teacher 认可的 token，不改 argmax；刹车阻止油门泛化到冲突位置。
- **低阈值部署**：原模型在低 τ 下会强行提交冲突组（τ = 0.5 时约 8% 的位置）。刹车若足够强，可在更低 τ 下保持准确率——第二个速度来源，也是与纯锐化方法的分界。

### 4.4 与最接近工作的区别

**OPTD**（2608.02942）已有"on-policy 状态 + 冻结 teacher + 推到阈值"的骨架，但只有油门：未通过验证的位置不压置信度，作者承认不能全面保持准确率（四基准平均 −3.4）。本文的差别是**刹车**，以及**位置级 sibling 判定**（一次 teacher 前向，而非逐前缀的反事实 rollout）。WINO+（2605.16941）有 defer，但它是均匀摊开的熵最大化，带阈值超参，teacher 非冻结。

**必须回答的问题**：本文的 PI 也来自 student，为何不落入 F2？因为 teacher 对 PI 的用法是审核而非复制。训练中记录 teacher 同意率 |V|/|C|：若它随训练涨到 1 且 |C| 持续增大，则 F2 对本文同样成立。

---

## 5. 风险

1. 位置级判定只看当前一步，漏掉跨步或多位置的联合冲突 → 报告与块级判定的一致率，必要时补一次联合复核。
2. 刹车效应小于评测噪声 → 配对检验，GSM8K 补样本量。
3. 推理时 LOO（COVER）同算力下追平 → 定位退为零推理开销版本。
4. OPTD 公开代码后在我们的设定下不掉点 → 差异只剩刹车带来的低阈值区间与分析部分。

---

## 6. 2026-11-11 修订：正确性过滤与内容项（方案 2，已实现）

目标改为**加速与涨点都要**。代码 `selective_draft/drafter_opsd.py`（`sibling.target=drafter`），配置 `configs/llada2_mini_drafter.json`。

- **状态来源**：当前 student 在部署阈值 τ=0.95 下解码。第 1 次贪心（就是部署解码器），第 2 到 4 次 T=0.9 采样。取第一条**答对、以 EOS 结束、含 \boxed** 的 rollout；4 次都不行，这题不产生梯度。
- **每个状态三项损失**（cap = 0.95）：
  - 油门：teacher 认可的候选，hinge 推到 cap。
  - 刹车：teacher 否决的候选，KL 拉向"从 student 采一组 sibling 后的 LOO teacher"（单样本无偏估计）。
  - 内容：该 rollout 在这一步提交的 token，capped CE 推到 cap。贪心答对时它只是速度项；贪心错、采样答对时，它把贪心路径不会产生的 token 教进去，这是 pass@k 转入 pass@1 的部分，也是涨点来源。
- **必做消融**：`sibling.content_weight=0`（无内容项）对比 1；无 LOO（全部候选只加油门）；随机否决。
- **lr 5e-6**（sib_t05_v1 与 path_v1 为 2e-5）。
- **训练日志第一眼看**：`agree_frac`（= 1 − LOO 违反率，在 student 自己的状态上）、`greedy_accept`（训练题上贪心 pass@1）、`greedy_truncated`、`greedy_tpf`、`kl_p0_block`。
