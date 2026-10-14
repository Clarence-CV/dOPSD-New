# Selective Draft OPSD：训练数学审阅与可衡量性分析

**日期：2026-10-14（同日修订，见文末 §7）。性质：对 [Selective_Draft_OPSD.md](./Selective_Draft_OPSD.md) 与 [工程计划](./Selective_Draft_OPSD_Engineering_Plan.md) 的审阅意见；结论用于决定代码默认值，不是实验结果。**

记号沿用方法文：状态 $S_t$，当前 block 的 MASK 集合 $M_t$，中间层 $\ell$（默认 28），早期读出 $r_i$，源位置 $A$，门控 $z_t=\mathbf 1[\max r_A\ge0.8]$，消息 $m_A=\sum_v r_A(v)E(v)$，末层分布 $p^{\rm final}$，冻结 Teacher 的边缘 $q_B^0=T_B(S_t)$ 与条件 $q_B^v=T_B(S_t^{+A=v})$。

## 0. 结论先行

1. **阶段一（早期读出适配器）在数学上是成立的、标准的。** 它就是 Tuned Lens：用 forward KL 把第 $\ell$ 层经过小适配器后的 LM-head 读出对齐到末层分布。它不改变模型输出路径，优化目标明确；训练后的校准、准确率与泛化仍须在独立状态上实测（见 §7）。
2. **阶段二按方案原文的目标 $\bar q_B=\sum_v r_A(v)\,q_B^v$ 训练，在理想情况下学不到任何东西。** 对一个自洽的掩码扩散模型，$\sum_v q_A^0(v)\,q_B^v = q_B^0$ 是恒等式（全概率公式）；阶段一越成功（$r_A\to q_A^0$），$\bar q_B$ 就越接近 $q_B^0$，而 $J_\psi$ 零初始化时 $p_B^{\rm final}=q_B^0$，于是损失从一开始就在最小值附近，剩下的梯度只来自模型自身的不自洽和读出误差，不是 sibling 依赖。**这是方案里必须改的一点，不是可选的诊断项。** §2 给出推导和修正。
3. **软消息 $m_A$ 在信息上是冗余的。** $m_A$ 是 $h_A^\ell$ 的确定函数，$J_\psi(h_A,m_A)$ 等价于 $J'_\psi(h_A)$：阶段二实际训练的是"在最高置信位置的 residual 上加一个 rank-64 的、按位置的可学习扰动"。它不向网络注入任何新信息，只是换一种后 4 层更容易用 attention 读到的表示形式。这不否定方案，但决定了：(a) 收益的天花板由"末尾几层能否把 A 的单点扰动转化为 B 的条件变化"决定，而我们已经有这个量的 oracle 估计（L28 activation patch 只恢复 reveal 效应的 19%）；(b) 计划 §11.3 的"换消息看 Student 是否响应"对照不能单独证明机制，因为 $J$ 可以绕过 $m_A$ 直接用 $h_A$。
4. **能否提高整体能力：在固定阈值 0.9 协议下，准确率收益的量级预期接近零；速度收益（更多位置提前过阈值）在理论上存在但很小且伴随过度自信风险。** 现有数据（阈值协议下 sibling 效应 +0.017 nat、负向仅 7%；L28 patch 恢复 19%）把 B 的可修复量压到 $10^{-3}$ nat 量级。要看到"sibling 通信是否有用"的信号，应把实验设计成**准确率–NFE 曲线**（扫 $\tau_{\rm commit}$，含 0.7/0.8）而不是在 0.9 一个点上比，并先做一个**不需要训练**的 oracle 天花板实验（§4）。
5. **衡量"同一步 token 一致性是否提高"有清晰可算的量**，核心是"并行提交 = 顺序提交"的一致率（§3），都能从已有的 state 录制离线算出，不需要改生成循环。

## 1. 阶段一：早期读出 = Tuned Lens，成立

目标 $\mathcal L_{\rm draft}=\operatorname{mean}_{i\in M_t}\mathrm{KL}(q_i^0\Vert r_i)$，只更新 $g_\phi$。

- $r_i=\operatorname{softmax}(W\,\mathrm{Norm}(g_\phi(h_i^\ell)))$ 是 $h_i^\ell$ 的按位置函数。forward KL 在"任意依赖 $h$ 的分布族"上的最小元是条件期望 $\mathbb E[q_i^0\mid h_i^\ell]$；rank-64 残差瓶颈限制容量，但初始化为恒等（上投影零初始化）只保证起点就是原生 logit lens（L28 原生一致率 55%），不保证训练后校准或泛化一定改善。
- forward KL 是 mode-covering 的：$r_i$ 的熵不会系统性低于 $q_i^0$。这只是一个直觉：forward KL 并不能直接推出门控覆盖率的上界，覆盖率以实测为准。
- 它完全在旁路上，关闭消息时与原模型逐位一致，这个 parity 是整套方法最干净的性质：**Student（消息关）≡ Teacher ≡ base**。它消掉了旧 d-OPSD 分析里"训练后 student 与 base teacher 权重不同"的混淆（见 `results/pipeline_gsm_20261002/SUMMARY.md` 读表须知），后面的一致性度量因此可以用同一个模型既当被测者又当 oracle。
- 只用 top-1 位置但对全部 $M_t$ 训练，是对的：若只训练被选中的位置，其余位置的读出会失控，top-1 选择本身就会漂移。

## 2. 阶段二：目标函数决定一切

### 2.1 消息不携带新信息

阶段二冻结 $g_\phi$，于是 $r_A=f(h_A^\ell)$、$m_A=E^\top f(h_A^\ell)$、$\delta_A=J_\psi(h_A^\ell,m_A)=J'_\psi(h_A^\ell)$。注入后 $\widetilde h_A^\ell=h_A^\ell+\delta_A$，其他位置不变，后 $L-\ell$ 层只能通过 attention 读到 A 的这个改变。

所以阶段二的真实假设是：**"存在一个只依赖 $h_A^\ell$ 的低秩扰动，使冻结的后 4 层在 B 处表现得像 A 已被揭示"**。这和 GPTRecord §2.4 的 activation patching 是同一个通道，只是 patch 用的是自然的 $h_A^{\ell,\rm reveal}$，这里换成学出来的 $\delta_A$。学出来的 $\delta$ 可以比自然 patch 更好（它是为下游效果直接优化的），但 19%（L28）/ 37%（L24）是该通道当前唯一的量化先验。

### 2.2 mixture 目标退化为边缘分布

方案 §4.2：$\bar q_B=\sum_v r_A(v)q_B^v$，$\mathcal L_{\rm cond}=\mathrm{KL}(\bar q_B\Vert p_B^{\rm final})$。

对真实数据分布 $\pi$，全概率公式给出 $\sum_v\pi(x_A=v\mid S_t)\,\pi(x_B\mid S_t,x_A=v)=\pi(x_B\mid S_t)$。掩码扩散模型的训练目标就是让 $T_i(S)$ 逼近 $\pi(x_i\mid S)$，所以对一个训练良好的模型

$$
\sum_v q_A^0(v)\,q_B^v \;=\; q_B^0+\varepsilon_{AB},\qquad \varepsilon_{AB}:=\text{模型自洽误差}.
$$

把 $r_A=q_A^0+\eta_A$（$\eta_A$ 为读出误差）代入：

$$
\bar q_B = q_B^0 + \varepsilon_{AB} + \sum_v \eta_A(v)\,q_B^v .
$$

而 $J_\psi$ 零初始化时 $p_B^{\rm final}=q_B^0$。于是 **$\mathcal L_{\rm cond}$ 的初始梯度只由 $\varepsilon$（模型缺陷）和 $\eta$（读出误差）驱动，阶段一做得越好 $\eta$ 越小，目标越接近"什么都别改"。** 门控选出的恰恰是 $r_A$ 与 $q_A^0$ 都接近 one-hot 的位置，两者在同一个 $v^\*$ 上，$\eta_A$ 在这些位置最小。

计划 §7.2 已经指出"mixture KL 的梯度等于各 hard 条件 KL 的加权平均"：$\nabla\mathrm{KL}(\bar q\Vert p)=\sum_v w_v\nabla\mathrm{KL}(q^v\Vert p)$。这正好说明问题出在权重 $w$ 上：**$w=r_A\approx q_A^0$ 是唯一会把条件信号平均掉的选择。** 用 MC 采样 $v\sim r_A$ 做 hard KL，期望上同样退化，只是多了采样噪声。

这不是实现细节，而是 GPTRecord §4.4 那条限制的精确形式：Student 在 $S_t$ 下观察不到 A 的实现值，就只能拟合条件分布的平均；平均正好等于它已经输出的边缘。

### 2.3 有信号的目标：条件于 A 实际写入的 token

sibling 依赖的定义本来就是"B 条件于 A **这一次实际确定**的 token"（AGENTS.md、GPTRecord §1.2）。相应的目标是 one-hot 权重：

- **realized（建议默认）**：$a$ = 该 rollout 里 A 最终写入的 token（A 在本步提交时就是本步的 $x_A$；否则是后续步写入的值）。目标 $q_B^{a}$，只需 **1 次** Teacher 前向。这是 OPSD 的 privileged-information 逻辑限制在同批 sibling 上：Teacher 知道 A 将成为什么，Student 不知道但可以从 $h_A^\ell$ 推断。贪心 + 阈值解码下 A 几乎总在本步提交（$\max r_A\ge0.8$ 的位置末层通常更尖），此时 $a$ 是 $S_t$ 的确定函数，Student 能学到针对它的条件响应（GPTRecord §4.4 的允许情形）。
- **argmax**：$v^\*=\arg\max r_A$，目标 $q_B^{v^\*}$。与 realized 的差别只在 $v^\*\ne a$ 的样本上；这个不一致率本身是必须报告的量（它就是"错误 Anchor"风险的频率）。
- **mixture（mc / top-K）**：保留为诊断，用来**实测 §2.2 的退化**：在同一批状态上报告 $\mathrm{KL}(\bar q_B\Vert q_B^0)$、$\mathrm{KL}(q_B^{a}\Vert q_B^0)$、$\mathrm{KL}(q_B^{v^\*}\Vert q_B^0)$。预期前者比后两者小一个量级以上。若不是，说明模型自洽误差 $\varepsilon$ 很大，那是另一个值得单独研究的现象，但不是 sibling 协调。

Student 侧的消息定义不必改：$m_A$ 仍是 full-soft embedding。因为 $m_A$ 只是 $h_A$ 的再编码，"软消息 + 硬目标"没有任何语义矛盾；方案 §4.2 认为"Student 用的是 $r_A$ 而非硬 token，所以目标也要 soft"这一推理不成立。

**风险的正确记账**：训练到 $q_B^{a}$ 后，推理时若 A 的最终 token $\ne$ 模型内部"假定"的值，B 被错误条件化。受控 Reveal 给出的不对称（正确 +0.099 nat，错误 −2.095 nat）意味着 0.8 的门控远远不够：若按这两个数字线性外推，命中率需要 $>95\%$ 才是正期望。所以阶段二之前必须测 $P(v^\*=a\mid \max r_A\ge0.8)$ 和 $P(v^\*=a_{32})$，门控阈值应由这个校准曲线定，而不是预设 0.8。

### 2.4 其他需要进入日志的副作用

- 注入改变 A 自己的末层分布。若 $c_A$ 因此跌破 $\tau_{\rm commit}$，A 反而延后提交，这是直接的速度损失。记录 $\Delta c_A$（消息开/关）。
- 阈值协议下 B 只有两种命运：本步 $c_B>0.9$ 直接提交（此时 sibling 效应本来就小），或者等下一步在**输入**里看到 A 的真实 token（完整 32 层条件化，免费）。消息唯一能改变的是"把一些 B 从第二种拉进第一种"。所以**速度收益 = 这类事件的频率 × 1 步**，而**准确率风险 = 这类提前提交与下一步本应得到的决定不一致的比例**。两者必须同时报告（§3 的 early-commit fidelity）。
- 阶段二是 on-policy 的，消息改变提交集合，于是改变 $S_{t+1}$ 的分布。计划 §8.3 的 refresh 逻辑正确；但 Stage 1 的 $g_\phi$ 是在 base 轨迹上训的，阶段二状态分布漂移后读出校准可能变化。至少在阶段二的开发集上复测门控校准。

## 3. 怎么衡量"同一步解码 token 之间的一致性"

所有量都在**固定的分析状态集**上算（消息关 = base，可做受控对比），再在各自 on-policy 轨迹上算一遍。下面 $C_t$ 为本步提交集合，$|C_t|\ge2$ 的步才有定义。

| 指标 | 定义 | 读法 |
|---|---|---|
| **并行–顺序一致率 PSA** | 对 $A\in C_t$（取源位置；无门控时取末层最高置信者）和每个 $B\in C_t\setminus\{A\}$：$\mathbf 1[x_B=\arg\max_x T_B(S_t^{+A=x_A})]$ | 并行写下的 B 是否等于"先写 A 再决定 B"会写的 B。100% = 并行提交没有协调损失。这是最直观的"一致性"。 |
| **sibling 条件间隙 $D_S$ / KL 版** | $\log T_B(x_B\mid S_t^{+A=x_A})-\log T_B(x_B\mid S_t)$；或 $\mathrm{KL}(T_B(\cdot\mid S_t^{+A=x_A})\Vert p_B^{\rm stu}(\cdot\mid S_t))$ | 已在 `analysis/pi_sibling_analysis.py` 实现。训练后 Student 自己的输出应更接近条件分布：KL 版下降、$D_S$ 负向比例下降。 |
| **提交对 PMI** | $\log T_B(x_B\mid S_t^{+A=x_A})-\log T_B(x_B\mid S_t)$ 在 $C_t$ 内所有对上求和 | 并行解码隐含独立性假设；PMI 之和是这个假设的误差。期望训练后分布向 0 或正收缩。 |
| **一步回看后悔率** | 对 $S_{t+1}$ 再前向一次，$\mathbf 1[x_B\ne\arg\max T_B(S_{t+1}\text{ with }B\text{ re-masked})]$ | 看到全部 sibling 之后模型是否还会选同一个 token。消息若只是让 B 更自信而不更正确，后悔率会上升。 |
| **early-commit fidelity** | 只对"消息开时过阈值、消息关时不过阈值"的 B：其 token 是否等于消息关闭时下一步会写的 token | 这是速度收益对应的唯一风险项，必须和 NFE 一起报。 |
| **门控校准** | 按 $\max r_A$ 分桶的 $P(v^\*=a_{32})$、$P(v^\*=a)$ | 决定 $\tau_{\rm msg}$，也决定 §2.3 的期望符号。 |
| **消息依赖性** | 固定 $S_t,A$，把 $m_A$ 换成次优 token / 随机 token 的 embedding，看 $\Delta p_B^{\rm stu}$ 与 $\Delta q_B$ 的相关 | 有参考价值，但因 §2.1，零响应不等于机制失效；需配合"把同一 $\delta_A$ 加到随机其他 MASK 位置"的对照。 |

任务层面：准确率、NFE、端到端时延、tokens/step，**按 $\tau_{\rm commit}\in\{0.7,0.8,0.9,0.95\}$ 画 base 与 Student 的准确率–NFE 曲线**。曲线整体右上移才算"要么更准、要么更快"。单点 0.9 的比较噪声大于效应。

## 4. 先做不训练的天花板实验

在阶段二任何训练之前，用已有 collect 的状态做三件事，总成本不超过一次 `gpu-a100-small` 短作业：

1. **退化实测**（§2.3 三个 KL 并列），确认 mixture 目标确实 $\approx q^0$。
2. **oracle 天花板**：在第 $\ell\in\{24,28\}$ 层边界把 $h_A^\ell$ 换成 $S_t^{+A=a}$ 前向得到的 $h_A^{\ell,\rm reveal}$，其他位置不动，测 B 处恢复了多少 $\mathrm{KL}(q_B^{a}\Vert q_B^0)$。这是"单点中间层扰动"通道的上界估计；若 L28 在阈值协议状态上仍只恢复 ~20% 且 $\mathrm{KL}(q_B^a\Vert q_B^0)$ 本身只有 $10^{-2}$ 量级，阶段二在 L28 的任务级效应不可测，应把默认层前移或改变实验协议（更低 $\tau_{\rm commit}$ 或固定预算，那里 sibling 冲突大一个量级）。
3. **门控校准曲线**（§3 最后一行），决定 $\tau_{\rm msg}$。

只有 1 确认退化、2 给出可测的天花板、3 给出 $>90\%$ 的高置信桶，再开阶段二训练。

## 5. 对工程计划的默认值修改（已在代码中落实，均可配置回原方案）

| 项 | 计划原文 | 本次默认 | 原因 |
|---|---|---|---|
| `teacher.mode` | `mc`（samples=4） | `realized`；`argmax` / `mc` / `topk` 可选 | §2.2–2.3；同时便宜 4 倍 |
| diagnose | teacher 诊断报告 mixture vs $q^0$ | 额外并列报告 realized / argmax vs $q^0$，以及 oracle patch 恢复率 | §4 |
| evaluate | 单一 `commit_threshold` | 支持阈值列表扫描，输出准确率–NFE 表 | §3 |
| 一致性诊断 | 无 | `diagnose --kind consistency`：PSA、PMI、后悔率、early-commit fidelity，离线跑在 state 录制上 | §3 |
| `guidance.early_layer` | 28 | 28 不变，但 smoke 与 oracle 诊断同时跑 24 | §2.1 的 19% vs 37% |

其余语义（单源、0.8 门控、full-soft 消息、A 不锁定、阈值优先、冻结 base、零初始化注入、hook 实现、parity 要求）保持计划原样。

## 6. 对"最终能否提高能力"的判断

- **能证伪的、便宜的部分先做**：§4 三项不需要训练任何东西。它们要么给出一个可测的天花板，要么直接说明 L28 单源 soft guidance 在阈值协议下没有可测空间。
- **最可能的正结果形态**：不是 0.9 阈值下准确率上升，而是在更低阈值（0.7–0.8）下 Student 的准确率衰减比 base 慢，也就是用同样的 NFE 保住更多准确率。这与用户目标里"增加同一步解码数量"一致。
- **最可能的负结果形态**：门控开启的位置末层本来就会提交、Receiver 要么本来就过阈值要么下一步免费拿到完整条件；学到的 $\delta_A$ 只让 B 更自信，PSA 不变、后悔率上升。这种情况下应回到 GPTRecord §5 的判断：要么前移注入层（L24，付出读出校准代价），要么承认需要 early commitment 语义（A 锁定为 $v^\*$，这样 §2.3 的目标与推理严格一致，但要解决错误锁定）。
- 不建议在没有 §4 结果之前扩大训练规模或增加 Partial Loop。


## 7. 2026-10-14 修订（采纳用户意见）

以下几处原文说得过头或不准确，已在正文改写或在此更正；§2.2 的退化论证保留为一个**待实测的假设**，不作为结论：dLLM 的条件分布并不满足理想的联合一致性，退化程度要看 `diagnose --kind teacher` 里 mixture / argmax / realized 三种目标相对 $q^0$ 的实测差距。

- **恒等初始化**：只保证初始点明确，不保证训练后读出的校准、准确率或泛化改善。
- **门控覆盖率上界**：不能由 forward KL 直接推出，以实测为准。
- **"B 下一步免费看到 A"**：不准确。下一次完整前向正是要节省的成本；A 未必本步提交；指导还可能改变 token 内容、阻止不可靠提交、改变后续轨迹。速度收益不能写成"提前提交事件频率 × 一步"，只能用实测 NFE 与时延回答。
- **指标解释收紧**：
  - 并行–顺序一致率 100% 只表示与所选 Teacher 这一次的顺序预测一致，两者可能一起答错。
  - 揭示 A 后的 log-prob 变化是条件响应；只有满足联合分布一致性等条件才等于严格 PMI。
  - 回看时 B 不变表示输出稳定，错误答案也可能稳定。
  - NFE 减少只是前向次数减少，实际加速要扣除中间 LM head、soft message 等开销。
- **核心待解问题**（用户指出）：Student 这一轮针对的是 A 的哪个候选作出响应，该候选与最终写入的 A 如何对应。允许 A 最终改变就保留了条件不一致的风险。首轮实验因此并列比较 mixture、argmax、realized 三种目标改变 Receiver 的方向与幅度，L24/L28 patch 仅作机制探针；有信号再继续投入阶段二。
