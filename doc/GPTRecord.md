# 从 Sibling Conditioning 到 Depth-wise Commitment：dLLM 步内并行决策内化研究记录（v3）

> **更新日期：2026-10-11**  
> **基础模型：** LLaDA-8B-Instruct（已有实验）；LLaDA2.0-mini / mini-CAP（后续外部验证候选）。  
> **研究状态：** Motivation 已获得多组初步实验证据；最终 Method **尚未锁定**。近期优先完成 **L24/L28 Early Anchor Feasibility**，然后判断是否值得做 **Selective Depth-wise Commitment / Partial Loop**。  
> **与 v2 的关系：** 保留原版 sibling gap、on-policy PI 的问题定义及有价值的理论讨论；纳入后续实际实验。**不再将全位置 L24 Draft Feedback、L24 Early Distillation 或 L32 外挂 Refiner 视为默认最终方法。**

---

## 摘要

扩散语言模型（diffusion language models, dLLMs）通过双向 Transformer 在一次去噪前向传播中同时预测多个被掩码位置，但原生 confidence-based decoding 通常根据逐位置预测分布选择并提交多个词元。同一步的不同位置虽能共享隐藏上下文，却不显式条件于彼此**本次实际确定的 token identity**。我们首先通过真实 on-policy 轨迹、受控 Sibling Reveal、逐层 Logit Lens 和 Activation Patching 研究这一差异。已有结果显示：揭开正确 Anchor 会提高 Receiver 的目标 log-prob，而错误 Anchor 会造成明显损害；原始 LM Head 对 Anchor 的身份直到 Transformer 后段才具有稳定可读性；较早注入真实揭开后的 Anchor hidden state，可更大程度地恢复 Receiver 的条件预测变化。与此同时，一个 500 题 training-free L24 Draft Feedback Pilot 表明，未经蒸馏的 Draft 与最终预测仅约 18% 一致，直接注入反而将 GSM8K 准确率由 30.8% 降至 26.6%。这些发现共同指向一种可能的 **prediction–communication timing mismatch**，但尚未证明早期 hidden states 不含词元身份，也尚未证明显式 Draft Feedback 必然有效。

在方法层面，我们重新区分 **全位置早期自条件化** 与 **实际 token 决策驱动的 sibling conditioning**。前者本质上更接近已有 layer-wise prediction / soft-embedding feedback；后者要求在 Transformer 的计算过程中产生一个可被后续层利用、且与最终提交语义一致的 Anchor 决策。目前最值得探索的是：先在 L24/L28 诊断高可靠、高条件价值的早期 Anchor 是否存在；若存在，再考虑 **Selective Depth-wise Commitment**，并通过 **Partial Loop** 补偿可靠决策出现较晚导致的剩余交互深度不足。最终目标是在 dLLM **自身内部**学习步内条件协调，而不是在最终 logits 后外挂一个修正器。

**关键词：** Diffusion Language Models；Parallel Token Commitment；Sibling Dependence；On-policy Privileged Information；Depth-wise Commitment；Looped Transformer。

---

## 1. 背景与问题定义

### 1.1 双向表征并不等于联合提交

令 $S_t$ 表示第 $t$ 次去噪时的 masked sequence，$\mathcal M_t$ 为当前仍被掩码的位置集合。$L$ 层双向 Transformer 的输出为

$$
H^{(L)}=F_{1:L}(S_t),\qquad
p_\theta(x_i\mid S_t)=\operatorname{softmax}(W_{\rm LM}\operatorname{Norm}(h_i^{(L)})).
$$

原生 confidence-based decoder 根据各位置预测与置信度形成当步提交集合 $\mathcal C_t\subseteq\mathcal M_t$，并写入其选定 token。其核心限制**不是**“位置之间没有双向 attention”，而是：当 $A,B\in\mathcal C_t$ 同时提交时，$B$ 的本次最终 logits 并没有经过一次以 **A 此次实际选定的词元 $a$** 为输入条件的后续更新。逐位置选择得到的结果可能与真正揭开 $A$ 后的条件判断不一致。必须区别表示层的双向交互与决策层的显式条件化。

这里的研究目标是改善**并行提交的协调性及最终任务表现**，不应未经验证就声称模型能在单次独立 logits 输出中精确实现任意联合采样。

### 1.2 Sibling Revealing 与 PI 的定义

对于同一原生去噪状态 $S_t$，选取待提交位置 $A,B$，令 $a$ 为 Anchor 的实际/指定候选词元，构造只揭开 A 的反事实输入

$$
S_t^{+A=a}=S_t[A\leftarrow a],\qquad S_t^{+A=a}[B]=[\mathrm{MASK}].
$$

冻结原模型给出

$$
q_B^0=p_{\bar\theta}(x_B\mid S_t),\qquad
q_B^a=p_{\bar\theta}(x_B\mid S_t^{+A=a}).
$$

$S_t^{+A=a}$ 是**训练/诊断可用、常规并行预测中不可用的 post-commit sibling-conditioned view**。它与传统 dOPSD 利用较远未来 rollout 状态不同：这里只改变**同一步一个 Anchor 的确定词元**，其余位置不随意揭示。这个区别必须在论文中说清楚，避免把一种局部条件干预笼统地称为完整 future reasoning。

---

## 2. 动机：从 Sibling Gap 到深度上的计算错位

### 2.1 原始研究起点：并行提交并不总是相互兼容

原 v2 中，LLaDA-8B-Instruct 采用 THU 风格 GSM8K **300 题**协议（128 steps / 256 generated tokens / block length 32；固定预算平均每步提交 2 tokens），在真实同一步提交的 sibling pair 上测量

$$
D_S(B\leftarrow A)
=\log p_{\bar\theta}(\hat y_B\mid S_t,A=\hat y_A)
-\log p_{\bar\theta}(\hat y_B\mid S_t).
$$

该实验报告固定预算条件下 $D_S$ 均值约 **−0.130 nat**、$|D_S|$ 均值约 **0.211**、约 **26%** 的位置 $D_S<0$；在另一套 confidence-threshold decoding 下，均值接近 **+0.017～+0.02**、负向比例约 **7%**。它表明激进同批提交与保守阈值提交呈现不同的 sibling response 统计，提示并行性—协调性之间存在潜在 trade-off。

**重要限定：** $D_S$ 是**模型在反事实揭示前后的目标 token log-prob 差**。除非逐位置条件分布来自同一个一致的联合分布，否则不能把它无条件等同于严格定义的 PMI，也不能由负均值直接断言“所有并行联合生成必然出错”。该数值也不是 ground-truth accuracy 增益。不同解码设置下的分布不同，不应直接互换。

### 2.2 新实验：正确/错误 Sibling Reveal 的受控对照

**问题。** Receiver B 的预测变化究竟来自 A 的**具体 token identity**，还是仅仅因为移除了一个 MASK？

**做法。** 在已有 on-policy decoding state 上，选取同一步待提交的 A、B，保持 B 及其他上下文不变；分别用冻结模型计算 masked baseline、正确揭开 A、错误揭开 A 的完整前向。对 B 比较目标词元 log-prob、Top-1 正确率、wrong-to-right（Repair）和 right-to-wrong（Damage）。这是输入条件干预而非模型训练；错误 A 的具体替换/抽样规则和 B 目标 token 的评估定义，仍须与原始实验脚本核对后在论文最终版披露。

**结果。** 从精度百分比反推，本轮统计对应 **128 个 Receiver 样本**：

| 条件 | B 正确率 | 与 Mask baseline 相比 | 其他观测 |
|---|---:|---:|---|
| A 仍为 MASK | **78.13%**（100/128） | — | 原始条件预测 |
| 正确揭开 A | **79.69%**（102/128） | **+1.56 pp** | 2 次 Repair、0 次 Damage |
| 错误揭开 A | **45.31%**（58/128） | **−32.82 pp** | 目标 log-prob 大幅下降 |

正确 Reveal 使 B 的平均目标 log-prob 改变 **+0.099**，报告的 **95% CI 为 [+0.034,+0.169]**；错误 Reveal 对应 **−2.095**。

**结论。** 对 B 而言，A 的 token identity 是有实质影响的条件变量；错误条件尤其可能造成强干扰。正确 Reveal 的平均 log-prob 收益具有正向信号，但 **Top-1 只净修复了 2/128 个位置**，不能将其表述成已经证明的大幅任务准确率改善。这个不对称性还意味着：方法不只要学习“使用 Anchor”，更要解决**何时足够信任 Anchor**。

### 2.3 新实验：逐层 Token Readout（原生 LM Head）

**问题。** 如果中间层已经包含可读的 A token，能否及时让其他位置利用？

**做法。** 在原始 masked forward 中截取不同深度的 $h_A^{(\ell)}$，通过原来的 final normalization 与共享 LM Head 输出 Draft，比较其与所采用的目标 A token 的一致性；不训练 probe、不提前揭示 A。

**结果。** 原实验报告的分层“可读率”为：

| 深度 | LM Head token 可读率 |
|---|---:|
| L0–L24 | **0%** |
| L28 | **55.47%** |
| L31 | **97.66%** |
| 首次可读层 | 平均 **28.06 / 32** |

**结论。** 在该诊断口径下，原生 LM Head 对 A 的词元身份直到后层才具有强可读性（**late lexical readout**）。这**不能证明**早期 hidden states 不包含任何可恢复的 token identity：未训练的原 LM Head 读不出，与通过重新训练的 linear probe 读不出，是两种不同命题。

### 2.4 新实验：Activation Patching 定位条件传播能力

**问题。** 如果在中间层提前向模型提供“已经知道 A 是什么”的表示，剩余 Transformer 层能否把它转化为 B 的预测变化？

**做法。** 两次冻结前向，分别计算 masked state 和只揭开 A 的 state。固定中间层 $\ell$，在原始 masked forward 的层边界**仅将 A 的 hidden state** 从 $h_A^{(\ell),\mathrm{mask}}$ 替换为 $h_A^{(\ell),\mathrm{reveal}}$；B 和其他位置在该边界的 hidden states 保持 masked 版本，再运行剩余层，测量 B 恢复了多少完整 Reveal 的条件收益。

| Patch boundary | 恢复 Reveal 收益比例 |
|---|---:|
| L8 | **104%** |
| L16 | **74%** |
| L24 | **37%** |
| L28 | **19%** |
| L31 | **6%** |

这些百分比是**相对于完整揭示效应的恢复比例**，不是准确率涨幅；具体归一化统计口径及接近零分母的处理须以实验代码为准。104% 并不代表精度提高 104%，可能来自非线性响应或归一化波动。

**结论。** 较早注入真实 revealed representation 通常能让后续层体现更强的条件变化，说明**后续层具有利用显式条件信息的计算能力**。但不能直接推出“收益下降完全由剩余层数不足导致”：早晚层的表示分布、其他位置已传播的信息、以及 patch 引发的分布外效应都可能影响结果。

### 2.5 综合现象：有吸引力但尚未闭环的 Mechanistic Hypothesis

现有三组受控实验支持以下逻辑：

$$
\underbrace{\text{Sibling identity influences B}}_{\text{Reveal}}
+\underbrace{\text{LM-head token readout emerges late}}_{\text{Logit Lens}}
+\underbrace{\text{Earlier privileged injection is more effective}}_{\text{Patching}}
$$

由此提出工作假设：**Prediction–Communication Timing Mismatch**。词元身份在原始可用读出空间里出现得较晚，而有意义的 sibling-conditioned computation 可能需要一定剩余深度。这里“mismatch”是实验启发的假设，**不是**现阶段已经严格证明的因果机制，更不能自动推出“中间 Draft Feedback 就是正确答案”。

---

## 3. 已完成的 Training-Free Pilot：为什么我们暂缓全位置 Draft

### 3.1 实验设计

在 **LLaDA-8B-Instruct** 上，不进行任何训练，取 **L24** hidden states，经原 LM Head 得到每个 MASK 位置的 hard Draft $\hat x_i^{24}$。将 Draft 的 token embedding 以手工控制的 residual 形式反馈到该位置：

$$
\Delta_i=E(\hat x_i^{24})-E([\mathrm{MASK}]),\qquad
\widetilde h_i^{24}=h_i^{24}+\alpha\frac{\operatorname{RMS}(h_i^{24})}{\operatorname{RMS}(\Delta_i)+\epsilon}\Delta_i.
$$

此操作用于诊断模型对未训练 Draft 的响应：$\alpha$ 仅是**手工注入强度**，不是学到的条件协调策略。反馈后继续运行 L25–32，最终仍按原生 confidence-based decoding 提交。已报告设置：$\alpha=0.1$、`conf > 0.9`、GSM8K **500 题**。**$\alpha=0$ 与原始 logits 的 parity 精确通过**，并确认模型权重未改变。

### 3.2 结果与解释

| 条件 | 正确数 | GSM8K 准确率 | 相对 baseline |
|---|---:|---:|---:|
| 同设置原始 LLaDA | **154/500** | **30.8%** | — |
| L24 Hard Draft Feedback（$\alpha=0.1$） | **133/500** | **26.6%** | **−4.2 pp** |

该反馈实验报告平均每题 **110.4 次 forward**、**5.06 秒**，平均 **2.51 tokens/step**；在抽查的相同 masked states 中，L24 Draft 与完整 L32 最终 Top-1 的一致率仅约 **18%**。

这一结果否定了当前具体的 **training-free L24 hard-draft + 手工残差注入** 方案的有效性，但无法区分两种潜在原因：(i) Draft 本身不可靠；(ii) token embedding 与 L24 residual 表示不对齐，注入破坏了隐藏表示。$\alpha=0$ parity 证明实现的开关行为正确，**不等于整个 GSM8K baseline 已经跨协议对齐**。

**数据口径警告。** L0–24“0% 可读率”与本实验“18% 的 L24–L32 Top-1 agreement”来自**不同抽样与指标定义/统计口径**，不能直接对照为同一种准确率。前者应重新核对是否只衡量 Anchor 目标词元，后者衡量更广的 masked positions。同理，这里的 **30.8%** 与其他历史实验中约 **79.3%** 的 confidence-decoding GSM8K 数字不能直接比较；必须核对 checkpoint、prompt、生成长度、block、steps、threshold、停止规则和答案抽取。

### 3.3 原定 L24 Early Distillation：从“即将开训”改为“候选对照”

此前曾构造冻结 Teacher 的 L32 分布监督 Student L24 的方案：

$$
\mathcal L_{\rm early}=\mathbb E_{S_t,i\in\mathcal M_t}\,\mathrm{KL}(p_{\bar\theta}^{32}(x_i\mid S_t)\Vert p_\theta^{24}(x_i\mid S_t)),
$$

并考虑用最终层保持损失 $\mathcal L_{\rm keep}=\mathrm{KL}(p_{\bar\theta}^{32}\Vert p_\theta^{32})$ 防止原模型退化。这可以验证“把中间层训成早退预测器”的可行性，但**不是现阶段必需的主线实验**：即使能将 L24 agreement 由 18% 提升，也只证明预测被提前，并未证明实际的 sibling-conditioned decision 在同一步被利用。将所有 MASK 的 Draft 广播给后续 attention，更像已有的 layer-wise self-conditioning，因此该训练计划目前**暂缓，而非宣称已经执行**。

---

## 4. 核心分析：为什么早期全位置 Draft 与真正的 Sibling 决策不同

### 4.1 全位置 Draft 只是表示变换，不自动产生决策顺序

对于任意中间层 $\ell$，设 Draft 分布 $\pi_i^{(\ell)}=g(h_i^{(\ell)})$、消息 $m_i=f(\pi_i^{(\ell)})$。如果对**所有** MASK 位置都注入 $m_i$，再经过剩余 self-attention，则 $m_i$ 只是 $H^{(\ell)}$ 的确定性变换：它不提供条件于 $S_t$ 的新外部信息，也没有明确规定哪个 token 的**实际决策**应该先于另一个 token。

这种机制仍可能因为 token-space bottleneck / 优化偏置而改善计算，但仅靠“增加早期 LM Head + Feedback”不足以建立真正的 sibling-conditioned commitment，且与 DSLP、DiscoLoop 等有显著结构重合。

### 4.2 在中间层提前选完整 Commit Set 会造成闭环矛盾

假设 L24 提前预测本步应提交 $\mathcal C_{24}=\{s_1,s_4,s_6\}$。一旦将其中 token 的决策反馈给 L25–32，这些层的 logits、置信度、排序都可能被改变，故一般没有理由保证

$$
\mathcal C_{24}=\mathcal C_{32}.
$$

若坚持最终由 L32 选择，L24 的定向协调可能对错位置；若强制采用 L24 集合，则等于把重要的调度决策提前到尚未充分成熟的层，且**位置选择正确**不等于**对应 token identity 正确**。因此不建议直接从“L24 预测完整提交顺序”开始。

### 4.3 更合理的分解：Anchor 与 Receiver 分工不同

可以只要求中间层**先确定极少数可靠 Anchor**，而不预先决定全部 Receiver。Anchor 的 token 作为后续层的显式条件，其余 MASK 位置保留可修改状态；最终由末层按原生规则选 Receiver。这样实现的是**一次 forward 内部的局部先决策、后条件化**，而不是让全体 token 同时互相猜测。

但有一个不可回避的设计抉择：

- **真正 Early Commitment：** Anchor 的 token 一经中间层确定，后续最终提交必须保持一致。这给出明确依赖语义，也带来提前锁错 token 的风险。
- **仅 Early Proposal：** 中间层 token 允许在 L32 被更改；灵活但 Receiver 可能受未被采纳的 Anchor 影响，本质上又接近 Draft Self-conditioning。

不能一边宣称使用已确定的 Anchor 做条件决策，一边允许最终 Anchor 无代价地变化。正式 Method 必须规定状态语义、何时 lock、失败时如何处理，以及在固定配额/阈值解码下 Anchor 是否占用提交额度。

### 4.4 PI Distillation 的本质限制

如果 Teacher 给出 $q_B^a=p_{\bar\theta}(x_B\mid S_t,A=a)$，但 Student 在相同 $S_t$ 下根本无法观察或内部确定 $a$，仅对 $p_\theta(x_B\mid S_t)$ 做 KL，**不能凭空学到针对不同已实现 $a$ 的条件响应**。对于随机 Anchor，它倾向拟合不同条件分布的平均；如果 $a$ 是 $S_t$ 的确定函数，Student 可以学习其特定条件分布，但这也不等于模型获得了真实随机联合采样能力。

因此，我们重新明确研究对象：**是否能在 dLLM 自身内部产生与训练/推理一致的 Anchor 决策，并把冻结 Teacher 的 sibling-conditioned behavior 蒸馏到这一真正存在的决策路径上。**

---

## 5. 最新方法候选：Selective Depth-wise Commitment 与 Partial Loop

> **本节是最新研究假设 / 待验证 Method，不是已实现或已取得收益的模型。**

### 5.1 路线 A：单次前向 Selective Early Anchor（优先诊断）

在深度 $\ell\in\{24,28\}$ 上，先通过轻量 readout 在部分高可靠 MASK 位置中选出最多一个或少数 Anchor $A$，确定其词元 $a$；随后将其 token identity 以中间层消息注入 A 自己的 residual stream，让其他位置通过后续原生双向 attention 读取：

$$
H^{(\ell)}=F_{1:\ell}(S_t),\quad
(A,a)=\operatorname{SelectAnchor}_\ell(H^{(\ell)}),\quad
\widetilde H^{(\ell)}=\operatorname{Inject}(H^{(\ell)},A,a),
$$

$$
P^{\rm final}=\operatorname{Head}(F_{\ell+1:L}(\widetilde H^{(\ell)})).
$$

除 Anchor 外的其他 MASK 不需要全员提早生成 token Draft。最终的 Receiver 集合可由末层决定；如果 Anchor 提前 lock，提交策略需显式确保它被包含在本次提交结果中，否则条件化与实际写入不一致。

**研究假设：** 少数 token 可能比全部 MASK 更早具备可高精度选出的身份，而且提前决定这些 token 对其他待提交位置有真正的条件价值。这个假设尚未在同一批样本上实证成立。

**核心挑战：** 可靠 Anchor 未必对 Receiver 有帮助；高置信位置与高 conditional utility 并非同一回事；选择本身需要校准；中间锁定可能改变原有 confidence scheduling。

### 5.2 L24 与 L28 的核心 trade-off

| 边界 | 已报告原 LM-head readout | 剩余层数 | Privileged Activation Patch 恢复比例 | 当前判断 |
|---|---:|---:|---:|---|
| L24 | 旧分层实验中接近 0%；独立 Pilot agreement 约 18% | 8 | 37% | 传播深度更多，预测可靠性困难 |
| L28 | 55.47%（旧实验口径） | 4 | 19% | 预测更成熟，传播空间有限 |
| L31 | 97.66%（旧实验口径） | 1 | 6% | 接近最终结果，但几乎没有后续协调深度 |

此表只是将**不同机制实验的观测量并列**，不能把 readout、patch 比例直接相乘当成预期收益。必须在**同一 held-out on-policy state** 上衡量高精度 Anchor 的覆盖率、与末层 commit set 的交集以及条件价值。若 L28 能找到少量强 Anchor，即便只有 4 层也值得先做小规模机制验证；若 L28 仍无法可靠选 Anchor，则不建议无止境向 L29/L30/L31 扫描。

### 5.3 路线 B：Partial Loop 补偿“晚成熟—短传播”的矛盾

如果 L28 的 Anchor 相对可靠，但只剩 L29–32 四层不足以协调 Receiver，可以考虑**只循环部分已有层**，不是重新训练一整套 looped backbone：

$$
\begin{aligned}
H^{24}&=F_{1:24}(S_t),\\
H^{28}&=F_{25:28}(H^{24}),\\
(A,a)&=\operatorname{SelectAnchor}(H^{28}),\\
\bar H^{28}&=\operatorname{Inject}(H^{28},A,a),\\
H^{\rm loop}&=F_{25:28}^{\rm reused}(\bar H^{28}),\\
P^{\rm final}&=\operatorname{Head}(F_{29:32}(H^{\rm loop})).
\end{aligned}
$$

计算路径是 **L1–24 → L25–28 → 选 Anchor / 注入 → 重用 L25–28 → L29–32**。相对于原 32 层，多计算 4 个 Transformer blocks，理想化层数成本 **36/32=1.125（+12.5%）**，还需中间 LM Head、Anchor selection 和额外访存；真实 latency 必须实测。

这个设计允许“先以较成熟的中间预测确定 Anchor，再保留约 8 个 block 的后续条件计算”，无需强迫 L24 直接预测高质量 token。与 CoCommit 的**在完整 L32 之后选择集合并重算尾层**不同，该方案希望在一次主干推理内部、最终预测之前进行 token decision；与 LoopMDM 的一般 latent refinement 不同，它拟使 loop 的收益由**可解释的条件决策**驱动。

**重大实现风险：** 预训练的 L25–28 并不保证能直接作用于自身输出 $H^{28}$，中间归一化尺度、RoPE、残差分布、位置信息与 attention cache 均需核查；简单复用这些层可能完全失效。不能把它描述成已验证可工作的无训练方案。尤其需要比较**相同额外 FLOPs 的无条件 loop**，否则无法证明收益来自 Anchor conditioning 而非纯粹加深计算。

### 5.4 如果未来训练，PI 应怎样服务于真正的决策通路？

训练仍然围绕真实 on-policy $S_t$，但**Anchor 在 Student 路径中必须实际被选择/指定**，与教师揭示语义匹配。冻结原模型 Teacher 在 $S_t^{+A=a}$ 上输出 Receiver 条件分布；Student 在原始 $S_t$ 上通过内部 Anchor 注入与后续层/Loop 输出：

$$
q_{B}^{a}=p_{\bar\theta}(x_B\mid S_t,A=a),\qquad
p_{\theta,B}^{\rm cond}=p_\theta(x_B\mid S_t;\operatorname{Anchor}_\ell=(A,a)),
$$

$$
\mathcal L_{\rm coord}=\mathbb E_{(S_t,A,B,a)}\mathrm{KL}(q_B^a\Vert p_{\theta,B}^{\rm cond}).
$$

另外需要一个 **Anchor Fidelity / Selective Reliability** 目标，以约束早期 token identity、选择与最终 commit 的一致性；是否采用 teacher final logits KL、选择性 CE、或只训练少数高精度 Anchor，尚未决定。为避免“Student 忽略消息，单靠原始 state 拟合条件 KL”的 shortcut，候选机制检验是在**相同 $S_t$ 上只改变 Anchor 消息**，查看 Receiver 是否产生与 Teacher 相同方向的条件响应。注意用 hard candidate intervention 训练时，必须考虑与自然推理时选定候选的分布匹配问题。

当前**不锁定最终总 Loss，不启动多目标训练**。先用下一节无训练实验确定存在足够的可靠 Anchor 与 conditional utility，再决定 Method 是否成立。

### 5.5 暂不作为主线的路线

| 路线 | 现阶段判断 | 理由 |
|---|---|---|
| **All-mask L24 Draft + Early KL + Feedback** | 暂缓，保留作对照 | 更像 layer-wise self-conditioning；未直接定义提交顺序，且已有 training-free 负结果 |
| **L24 预测整个 commit set** | 不建议直接做 | 中间选择改变后续 logits，早层/末层的集合一致性无法保证 |
| **L32 完整预测后外挂 Refiner** | 不作为目标方法 | 虽易于训练 teacher conditional KL，但与 ADJUST/MRP 等重合、并未把条件决策内化到 backbone |
| **L32 后重算若干层** | 作为强基线，而非创新起点 | 与 CoCommit 高度接近；若采用必须证明成本/机制优势 |
| **纯 Loop、无 Anchor** | 必做 matched-compute 消融 | 可以改善表示，但不能据此归因于 sibling-conditioned decision |

---

## 6. 下一步：优先做的三个小而决定性的实验

### 6.1 Experiment I — Early Anchor Feasibility（最优先；无需训练）

**目标：** 是否存在能在 L24 或 L28 被可靠锁定、又能帮助 Receiver 的 Anchor？

**数据与协议：** 冻结 LLaDA-8B-Instruct，沿用真正运行时的 confidence decoder，保存测试集之外或独立分析集上的真实 $S_t$、原生最终提交集合 $\mathcal C_t^{32}$、各位置的 logits/confidence；在同一个 state 上截取 L24/L28 readout。必须明确训练分析与正式评测数据的划分，同时保留正确与错误 rollout，避免只分析容易样本。

主要指标：

1. **Early Identity Reliability**：L24/L28 高置信候选与 **L32 最终拟提交 token** 的 agreement；另独立衡量与任务参考 token 的正确性（若可对齐）。这两者不可混用。
2. **Selective Precision–Coverage**：只取每步最高分 1 个/最多 2 个 Anchor 时，置信度阈值—覆盖率—错误锁定率曲线。重点不是所有 MASK 的平均 Top-1 agreement，而是**是否存在可用的高可靠子集**。
3. **Commit Consistency**：L24/L28 所选 Anchor 是否真的属于 L32 原生 $\mathcal C_t^{32}$；如果不属于，记录提前 lock 对提交配额与阈值逻辑的潜在冲突。
4. **Sibling Conditional Utility**：对所选 Anchor $A=a$ 只揭开 A，测量 Receiver $B$ 的 $\Delta\log p$、Repair/Damage/Net Gain；分别报告正确和错误 Anchor。定义候选 Receiver 为 L32 原生同批提交位置，但在后续真正部署时不能把这个 oracle 集合当作早期可见条件。
5. **Jointly Useful Anchor Rate**：同时满足预先指定的 Anchor 可靠性与条件收益标准的状态占比。作为诊断可使用 Reveal-derived utility，但**不得在推理时用它挑 Anchor**，因为它需要额外 privileged forward。

**核心可视化：** 一张 L24 vs L28 的 precision–coverage 图，以及“可靠 Anchor 的 sibling conditional utility”分桶图。每个结果报告样本数与置信区间，而不只画平均 KL。

### 6.2 Experiment II — Loop/Injection Feasibility（只有实验 I 有信号才做）

冻结同一模型，在**同一个 masked state** 比较：原始 32 层、L28 选择 Anchor 后剩余 4 层注入、L28 + Partial Loop（额外重用 L25–28）、同等 4 层开销但**无 Anchor** 的纯 loop。由于预训练模型未适配 loop，先做 tensor/scale/parity 检查和小规模成对的 Receiver 方向性诊断；如 loop 直接失效，不做大规模跑分，不解释为研究方向已失败。

首要观察是：**以真实选定 Anchor 为条件的预测响应，是否比同等额外计算的无条件 refinement 更贴近 Reveal Teacher？** 再比较最终 accuracy、tokens/forward、实际 wall-clock latency 与错误 Anchor 的负面传播。

### 6.3 Experiment III — 训练方案的 Go/No-Go 决策

只有当实验 I 证明“可靠且有价值的 early Anchor”并且实验 II 证明“剩余层/loop 可以消费 Anchor 信息”，才考虑训练 Anchor Fidelity + Conditional PI Distillation。届时应把以下四组作为最小对照：**base / pure loop / early Anchor without PI / early Anchor with PI**；再比较 CoCommit、ADJUST、MRP 等强相关方法的准确率—时延 Pareto。

若实验 I 中 L28 仍找不到有效 Anchor，则优先讨论是否必须学习**选择性早期决策器**或转向“层间决策状态”而非强行训练全部位置的 early token head；不要机械地继续向 L29–31 扫描。

---

## 7. 机制与理论边界：已知、未知、需要修正的旧推导

**已支持的结论。** (1) Sibling token identity 对 Receiver 的反事实预测有实质影响；(2) 原始 LM Head 可读性集中于后层；(3) privileged revealed state 越早 patch 通常越能恢复条件变化；(4) training-free L24 全位置 hard feedback 在目前 500 题设置下有负收益。

**尚未支持的结论。** (1) 早期 hidden state 完全没有 identity；(2) 随层数衰减的 patch 收益纯粹由层数决定；(3) L28 有可可靠早锁且对 Receiver 有帮助的 Anchor；(4) Loop 必然比追加一个原生去噪 step 更有效；(5) 所有位置 Draft Feedback 优于现有双向 attention；(6) PI KL 必然逼迫 Student 使用特定 Anchor message。

**对 v2 数学叙述的修正。**

- $D_S$ 与严格 PMI 的相等需要**一致联合概率模型**的假设；普通 masked conditional 网络未必满足。把 $D_S$ 称为 reveal response / conditional log-prob shift 更审慎。
- “前向只能产生乘积分布，所以任何一次前向都不能改善联合采样”是过强表述。若模型有明确内部决策与随机变量传递机制，单次计算中也可能实现非平凡耦合；限制具体取决于最终是否仍独立采样每位置 logits。我们目前不声称 exact joint sampling。
- “按置信度选择 Anchor”**不能自动保证**同步更新具有 Gauss–Seidel 的单调性：该结论要求明确的共同联合目标及真实顺序条件更新，在近似 teacher、深层修正和预测错误下均需重新验证。
- 单一 $\mathrm{KL}(q_B^a\Vert p_S(B\mid S_t))$ 不意味着学生真的使用了 A 的 token 信息；只有 Student 内存在对应条件路径，并通过改变条件的干预测试验证依赖，才有资格称为 conditional response distillation。
- L24/L28 token identity 与末层 commit set 的关系是**需要实验测量的问题**，不能依据 pooled LM-head agreement 自动推出 early commit 可用。

---

## 8. 相关工作与明确的 Novelty 边界

### 8.1 中间预测与 token-space feedback：不能重复宣称为新模块

**DSLP** [1]（AAAI 2022）在非自回归翻译中用 layer-wise prediction、深监督与预测 embedding feedback，直接覆盖“中间 LM Head 读出 → token embedding → 后续层交互”这一架构祖先。**DiscoLoop** [2]（2026）在 looped Transformer 中联合连续 hidden state 与离散 embedding channel，指出 token 可读性与 embedding-space 可用性可能错位。这两篇对解释我们早期全位置 Draft 路线的 novelty 尤其重要：**Draft Feedback 本身不是贡献**。

### 8.2 并行提交与条件修正：已有方法占据哪些位置

**CoCommit** [3]（2026）在完整预测选出待提交 bundle 后，利用 marker 指示集合并部分重算 backbone 尾层以协调多个 token；它是 partial recomputation / joint commitment 最重要的直接对照。**ADJUST** [4]（2025）采用一个轻量 sampler 逐个近似条件采样，使一次完整 Backbone Forward 可以接多个轻量条件前向。**MRP** [5]（2026）用 residual predictor 近似揭开 token 后下一 denoising state 的 logits，并可用于 speculative / remasking。它们表明“揭开 sibling 后条件分布 → 蒸馏成廉价预测器”并非自动新颖；我们的潜在差异必须在**原生 backbone 内部的决策路径**上体现。

**DAPD** [6]（ICML 2026）根据 attention 导出的 dependency graph 挑选弱耦合位置并行提交，属于**避免强依赖同批提交**；**AXON** [7]（2026）选择能够帮助其他未知位置的 confident anchor，属于**支持性揭示与调度**。因此“找有帮助的 Anchor”这一思想本身也已有先例；我们需要回答的新增问题是**能否把 Anchor 决策前移至 Transformer 深度维度，并内化 sibling conditioned computation**。**Learning Unmasking Policies** [8]（ICML 2026）进一步说明学习 commit/unmask scheduling 本身已有强 baseline。

### 8.3 Loop / Recurrent Depth：最需要防守的竞争路线

**LoopMDM** [9]（2026）在 masked diffusion LM 中选择性循环早中层，报告参数不变情况下的训练效率和性能收益，且通过 attention 分析讨论 MASK 位置交互。我们不能把“dLLM 循环一些层改善 masked-token interaction”作为 novelty。**Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models** [10]（ICML 2026）展示 recurrent-depth 模型在循环推进中并行细化 latent states / 引入新 token，说明“深度循环 + token decisions”已属于相关研究。**DiscoLoop** [2] 还覆盖 discrete-token feedback + recurrence 的具体实现。

因此当前 **Partial Loop + Anchor** 的真正待证明贡献只可能是：在原生 dLLM 中使用**选择性、可靠的内部 token decision**作为条件，将同一步 sibling response 通过 on-policy PI 精确监督，并在与 matched-compute loop 和 CoCommit 对比中显示明确优势。若只有“多循环四层、任务分数上涨”，论文创新不足。

### 8.4 On-policy PI、深度诊断与吞吐相关工作

**Learning from the Self-future: On-policy Self-distillation for dLLMs** [11]（2026）与 **dOPSD: On-Policy Self-Distillation for Diffusion Language Models** [12]（2026）通过后续去噪/自未来轨迹构造 Teacher PI，是我们数据采样与教师构造的重要背景，但二者的 self-future PI 不等于只揭开一个同批 sibling 的局部 PI。**Steering Without Breaking** [13]（2026）给出离散扩散模型内部属性形成时机与机制分析的参考，LLaDA 的逐层可读性必须以**我们自己的测量**为准。**DFlash** [14]（ICML 2026）是 block-diffusion draft 用于 AR speculative decoding 的系统路线：其 draft/verification 目标不同，适合在讨论“方法为何坚持 dLLM backbone 内化，而非外挂复杂 speculative pipeline”时引用，而非作为同一任务的直接 apples-to-apples baseline。

**模型扩展提醒。** **LLaDA2.0-mini / mini-CAP** 是更现代的强模型候选，mini-CAP 属于 confidence-aware parallel training；它的层数/ MoE 特征与 32 层 LLaDA-8B 不同，不能将 L24/L28 生搬硬套，需按真实结构重新确定分层诊断位置与原生 decoder。[15]

---

## 9. 下一轮研究讨论：四个必须先回答的开放问题

**Q1 — Early decision 还是 early proposal？** 如果中间层只是“猜到” A，L32 可以推翻它，Receiver 对旧猜测的响应会与最终 commit 不一致；但如果中间层真的锁定 A，就可能让错误 Anchor 带来连锁损害。我们需要一种清晰、可实现且评测可归因的 commitment 语义，而不是用“Draft/Anchor”混称两者。

**Q2 — Reliability 与 Utility 能否同时满足？** 最高置信度位置通常最容易预测，却不一定最能帮助其他位置。AXON/DAPD 已研究位置价值/依赖；我们应先验证在 L24/L28 能否找到**早可确定、会被最终选中、且对同批 Receiver 有显著条件作用**的交集。交集为空时，选择策略再复杂也难有基础。

**Q3 — Loop 是真正改善条件依赖，还是仅仅增加深度？** L28 + 重用 L25–28 的方案很直观，但也最容易退化成 LoopMDM/CoCommit。需要 matched-extra-FLOPs 纯 loop、same-layer injection、post-final partial recompute、额外 denoising step 的严格对照，还需关注 loop 表示分布外问题。

**Q4 — PI 是否提供了传统 CE/KL 无法替代的训练信号？** 当 Teacher $q_B^a$ 依赖已决定 Anchor 时，Student 是否真的经由 A 的内部消息调整 B？应对同一 state 做不同 Anchor 的反事实响应检验，并在相同架构、预算下对比 **无 PI / 普通监督 / Sibling PI**。这是未来 Method 的学术区分度，而不是多加一个 KL 符号。

**当前执行结论：** 先做 **L24 vs L28 Early Anchor Reliability + Commit Consistency + Sibling Utility**（同一真实 on-policy state；无需训练）。根据结果决定是否进入 **L28 Selective Anchor + Partial Loop** 的小规模机制实验。暂时不启动全位置 L24 Draft Distillation；也不将 L32 + 外挂 Refiner 作为论文主线。

---

## 参考文献

> 优先列出与方法重合度高、在讨论中真正决定设计选择的工作；以下题目和链接已与公开论文页核对。本文自己的实验统计来自研究对话提供的结果和原 v2 记录，最终投稿前需回查原始 trace、代码与评测配置。

[1] Huang, C., Zhou, H., Zaïane, O. R., Mou, L., & Li, L. **Non-Autoregressive Translation with Layer-Wise Prediction and Deep Supervision**. AAAI 2022. https://arxiv.org/abs/2110.07515

[2] Fu, H., et al. **DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning**. arXiv:2607.00341, 2026. https://arxiv.org/abs/2607.00341

[3] Yao, L. **Don't Commit Alone: Joint Token Commitment in Diffusion Language Models** (CoCommit). arXiv:2607.04469, 2026. https://arxiv.org/abs/2607.04469

[4] Bansal, P., & Sanghavi, S. **Enabling Approximate Joint Sampling in Diffusion LMs** (ADJUST). arXiv:2509.22738, 2025. https://arxiv.org/abs/2509.22738

[5] Xu, Y., et al. **Multi-Token Residual Prediction** (MRP). arXiv:2605.18817, 2026. https://arxiv.org/abs/2605.18817

[6] Kim, B., Jeon, D., Jeon, M., & No, A. **DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs**. ICML 2026. https://proceedings.mlr.press/v306/kim26t.html

[7] Abu Ayoub, G., et al. **Supportive Token Revealing for Fast Diffusion Language Model Decoding** (AXON). arXiv:2606.04236, 2026. https://arxiv.org/abs/2606.04236

[8] Jazbec, M., et al. **Learning Unmasking Policies for Diffusion Language Models**. ICML 2026. https://proceedings.mlr.press/v306/jazbec26a.html

[9] Lee, S., et al. **Looped Diffusion Language Models** (LoopMDM). arXiv:2605.26106, 2026. https://arxiv.org/abs/2605.26106

[10] Geiping, J., Yang, X., & Su, G. **Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models**. ICML 2026. https://proceedings.mlr.press/v306/geiping26a.html

[11] **Learning from the Self-future: On-policy Self-distillation for dLLMs**. arXiv:2606.18195, 2026. https://arxiv.org/abs/2606.18195

[12] Dat, P. T., Li, Q., & Wang, X. **dOPSD: On-Policy Self-Distillation for Diffusion Language Models**. arXiv:2607.04428, 2026. https://arxiv.org/abs/2607.04428

[13] Zhou, H., Roy, S., & Gangadharaiah, R. **Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models**. arXiv:2605.10971, 2026. https://arxiv.org/abs/2605.10971

[14] Chen, J., Liang, Y., & Liu, Z. **DFlash: Block Diffusion for Flash Speculative Decoding**. ICML 2026. https://proceedings.mlr.press/v306/chen26bv.html

[15] inclusionAI. **LLaDA2.0-mini-CAP Model Card**. https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP

---

### 附录 A：各批实验不可混用的口径速查

| 批次 | 规模/配置 | 主要结果 | 主要用途 |
|---|---|---|---|
| 原 v2 sibling-gap | GSM8K 300 题；128 steps / 256 tokens / block32；固定 2 tokens/step，与另一个阈值设置比较 | 固定预算 $D_S=-0.130$，负比例约 26%；阈值均值约 +0.017～+0.02、负比例约 7% | 并行提交条件响应的初步证据 |
| 受控 Sibling Reveal | 128 个 B 样本；相同 state 的 masked / correct-A / wrong-A 前向 | 78.13% → 79.69%（+2 Repair）；wrong-A 45.31%；$\Delta\log p$ +0.099（95% CI [+0.034,+0.169]）/ −2.095 | 确认 token identity 的反事实影响 |
| Layer Readout + Patching | 32 层 LLaDA；按层读取/替换 A 表示 | L28 55.47%、L31 97.66%；patch L8 104%、L16 74%、L24 37%、L28 19%、L31 6% | 时间错位的机制证据 |
| Training-free L24 Feedback | GSM8K 500 题；$\alpha=0.1$、conf>0.9 | baseline 154/500 → 133/500；L24–L32 agreement 抽查约 18%；110.4 forward/题，5.06 秒/题 | 无训练中间 Draft 的负结果 |

**附录中的历史数值均是研究过程中报告的结果，不代表在同一 evaluator / sampling configuration 下重新统一复现。** 论文正式实验应先完成配置审计与同协议验证。
