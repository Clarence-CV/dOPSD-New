# dLLM + OPSD 并行推理项目讨论总结

## 0. 当前项目一句话

我们想研究：**能否利用 on-policy self-distillation（OPSD）让 diffusion LLM 在保持并行解码优势的同时，学会更可靠、更协调的并行 reasoning，而不是只提升每个 token 各自的边缘分布。**

当前最重要的问题不是再设计一个复杂 loss，而是找到一个真正成立的 motivation：

> **现有 dOPSD 主要利用 future PI 改善 token-level prediction，但 dLLM 真正独特的困难是：同一个 denoising step 中多个 token 是并行决定的，而这些 parallel decisions 之间缺少直接的 decision-level supervision。**

---

# 1. 项目背景

我们关注 diffusion language model（dLLM）的 post-training，尤其是 OPSD / dOPSD。

dLLM 在一个 denoising step 中，会对多个 masked position 同时得到 token distribution：

\[
p_\theta(x_i\mid S_t), \qquad i\in M_t
\]

其中：

- \(S_t\)：第 \(t\) 个 denoising step 的完整 noisy state；
- \(M_t\)：当前仍然 masked 的位置集合；
- decoder 会根据 confidence / entropy 等规则，从中选择若干 token 一次性 reveal；
- 同一步真正被 commit 的位置集合记作 \(C_t\subseteq M_t\)。

并行解码是 dLLM 的核心优势，但也带来一个基本问题：

\[
p(x_{C_t}\mid S_t)
\neq
\prod_{i\in C_t}p(x_i\mid S_t)
\]

一般情况下，同一步 token 之间可能存在依赖，而标准输出形式主要提供 position-wise marginals。

需要注意：

> dLLM 的 Transformer backbone 并不是“不知道 token 之间有联系”。

每个 masked position 的 hidden state 可以看到 prompt、已经 reveal 的 token、其它 mask 的位置等全局上下文。真正缺失的是：

> **在同一个 forward 中，两个即将同时被生成的 token 都还是 `[MASK]`，因此它们看不到彼此最终 realized value。**

所以更准确的说法是：

> **dependency 可能已经存在于模型内部表示中，但不一定被充分表达到同一步并行采样所使用的 token-wise marginals 中。**

---

# 2. 两篇主要 dOPSD baseline

我们一直围绕两篇工作讨论。

## 2.1 THU: Learning from the Self-future / d-OPSD

核心思想：

- student 在当前 noisy state 上预测；
- teacher 额外看到模型自己最终生成结果中的一部分 token；
- 用这种 self-generated future / suffix privileged information 去监督当前 prediction；
- 主要是 step-level / token-level distillation。

简单理解：

> **当前 state 偷看自己最后生成出来的一部分答案，再教回当前 state。**

我们后来检查了 released code，发现两个重要实现细节：

### 解码并不是单纯“confidence 超阈值就 reveal”

代码里会先根据 mask 数量和剩余 step 数计算每一步固定要 transfer 多少 token，然后在当前 confidence 里选 top-k。

也就是更接近：

\[
\text{fixed quota per step} + \text{confidence ranking}
\]

而不是纯 threshold decoding。

### 实际 distillation 使用 top-20 vocabulary truncation

公开配置中：

```text
TOP_K_LOSS=20
```

也就是 teacher distribution 的 distillation 实际只保留 teacher top-20 vocabulary token，而不是严格意义上的 full-vocabulary distillation。

这说明论文层面的 objective 和 released implementation 之间存在 operational difference。

---

## 2.2 NUS: dOPSD

核心思想：

- student 的真实 intermediate denoising state 作为当前 state；
- teacher 使用同一条 rollout 中更晚、更完整的 denoising states；
- later state 为 earlier state 提供 privileged information；
- 通常只保留最终正确的 rollout。

简单理解：

> **用自己真实 future denoising trajectory 来教当前 state。**

我们认为 NUS 比 THU 更“dLLM-native”，因为 PI 直接来自真实 denoising trajectory。

但也注意到一个重要现象：

> NUS released recipe 中经常设置 `gen_steps == gen_max_new_tokens`，实际上尽量避免非常 aggressive 的 multi-token-per-step commit。

因此它虽然研究 dLLM trajectory，但并没有真正把“同一个 denoising step 内多个强相关 token 的并行 decision”作为核心问题来处理。

---

# 3. 最初的 PI 思路，以及为什么后来放弃

我们一开始想把 self-future PI 拆成：

\[
F
\approx
F_{\text{reasoning}}
+
F_{\text{previous/state-known}}
+
F_{\text{noise}}
\]

然后想：

> previous-state 已知信息是冗余的，noise 没用，所以通过 contrastive 方法把这些东西去掉，只留下 reasoning-relevant PI。

后来我们认为这个 motivation 不够成立。

原因：

1. **redundant 不等于 harmful**；
2. 如果一部分 PI 真的是当前 \(S_t\) 的确定函数，那么它对目标的 conditional mutual information 可以是 0，但这只能说明“没有额外信息”，不能说明“去掉后模型会更好”；
3. 没有一般理论保证：
   \[
   \text{cleaner PI} \Rightarrow \text{better reasoning}
   \]
4. NUS 的 horizon ablation 反而显示更多 future 往往更好；
5. THU 也有现象说明更强 teacher 不一定对应更好 distillation，但这只能说明“teacher strength 和 student gain 不单调”，不能证明 previous-state information 有害。

所以：

> **“self-future 里面混了很多无用信息，所以要 purification”目前没有足够证据，不适合作为主 motivation。**

RLCSD 的 contrastive PI 是合理的，因为它先观察到一个明确 pathology：privileged solution 引起 style drift，然后再用 correct-vs-wrong contrast 消除 style component。我们目前没有类似证据证明 self-future 的 state-known component 有害。

---

# 4. 我们后来转向的核心问题：parallel marginal / same-step coordination

我们逐渐认为，更 dLLM-native 的问题是：

> **同一步并行生成的 token，单独 marginal 可能都合理，但组合起来可能不协调。**

例如：

\[
p(x_i\mid S_t),\quad p(x_j\mid S_t)
\]

都很合理，但真正希望的是：

\[
p(x_i,x_j\mid S_t)
\]

标准并行采样近似为：

\[
p(x_i,x_j\mid S_t)
\approx
p(x_i\mid S_t)p(x_j\mid S_t)
\]

如果 \(x_i,x_j\) 强相关，就会有 factorization error。

需要再次强调：

> 这不是说 backbone 完全没有 dependency，而是同一步两个 realized token 还不存在，因此它们不能显式互相 condition。

如果一个 token 先在前一步 reveal：

\[
p(x_j\mid S_t,x_i)
\]

下一步就可以显式利用 \(x_i\)。  
所以 dependency 往往只能跨 denoising steps 被显式利用。

---

# 5. 我们讨论过但认为过于接近已有工作的方向

## 5.1 “什么时候应该 reveal token”

曾经提出：

> high confidence 不等于 safe-to-parallelize。

定义 sibling sensitivity：

\[
D\left(
p(x_i\mid S_t),
p(x_i\mid S_t,\text{sibling})
\right)
\]

如果 reveal sibling 后当前 token distribution 变化很大，就说明不适合和 sibling 同时 commit。

进一步想让 student confidence 学成：

\[
\text{high confidence}
\approx
\text{high correctness}
+
\text{low sibling sensitivity}
\]

后来认为这条路线和已有工作太接近：

- PUNT：conditional-independence testing；
- SPEED：context sensitivity / safe parallel group；
- DAPD / DEMASK：dependency-aware decoding；
- Disentangled Decoding：dependency-aware self-distillation + special decoding；
- dParallel：训练 certainty/confidence。

所以：

> **如果我们的目标变成“学什么时候 reveal / 哪些 token 能安全并行”，无论 insight 还是 method 都容易撞车。**

这条不作为主方向。

---

## 5.2 直接 sibling-conditioned KL

曾经提出：

把同一步 token 分成 \(A,B\)，teacher 看到 \(A\)，预测 \(B\)：

\[
p_T(B\mid S_t,A)
\]

student 只看到：

\[
p_S(B\mid S_t)
\]

然后做：

\[
D_{\rm KL}(p_T\Vert p_S)
\]

后来发现一个很重要的理论问题。

如果对所有可能的 \(A\) 平均：

\[
\sum_A p(A\mid S)p(B\mid S,A)=p(B\mid S)
\]

所以简单把所有 conditional teacher distribution 压回一个 unconditional marginal，可能平均回原来的 marginal。

也就是说：

> student 在 inference 看不到 sibling 的 realized value，不能凭空恢复任意 conditional dependency。

因此：

> **“训练时强行给 teacher sibling，再让 student 模仿”本身不能保证 inference 获得真正的 intra-step conditional ability。**

这是目前最重要的理论约束之一。

---

# 6. Shared latent / coordination state 方向

我们后来问：

> 如果 inference 时 sibling 不存在，怎样才能让多个 parallel token 真正协调？

一个自然答案是给它们一个 inference 时共同可见的 shared latent：

\[
z_t\sim p_\theta(z\mid S_t)
\]

然后：

\[
p(x_{C_t}\mid S_t)
=
\int p(z_t\mid S_t)
\prod_{i\in C_t}
p(x_i\mid S_t,z_t)\,dz_t
\]

这样所有 token 虽然仍然并行，但共享同一个 \(z_t\)。

如果：

\[
z_t=A
\]

所有 token 都围绕 reasoning mode A 生成，就不会出现：

\[
x_i\sim A,\quad x_j\sim B
\]

这种 mode mixing。

但我们随后调研发现：

> **shared latent 用来解决 parallel token dependency / factorization 已经有人做。**

例如 VADD、DiLaDiff 等 latent-variable diffusion model，明确使用 shared latent 来建模 inter-token / inter-dimensional correlation。

因此：

> “给 dLLM 加 shared latent 解决并行 dependency”本身不新。

如果继续这个方向，区别必须来自：

> **用 dOPSD / successful on-policy reasoning rollout 去学习一个 reasoning-specific coordination mechanism，而不是重新设计 generic latent generative model。**

---

# 7. 当前最重要的 epistemic constraint：PI 能否在 inference 被内化

我们反复讨论后认为，这是整个项目的生死点。

训练时 teacher 可以看到：

\[
F_t
\]

或者 sibling / future / final response。

但 inference 时 student 只有：

\[
S_t
\]

所以并不是所有 PI 都能被 distill。

如果 PI 中的信息完全不能从 \(S_t\) 预测：

\[
z \not\approx f(S_t)
\]

那么 student 不可能在 inference 凭空恢复它。

所以 OPSD 真正能 work 的前提应该是：

> **PI 帮 teacher 暴露了某种本来就隐含在当前 state 中、但 student 没有充分提取或利用的结构。**

这可能是：

- reasoning mode；
- step-level coordination state；
- uncertainty structure；
- parallel decision quality；
- transition direction。

目前我们还没有决定是哪一种。

---

# 8. 为什么“PI 让 token 概率更高”不是正确研究目标

我们后来明确意识到：

> PI 对某个 token 的 probability shift 本身不说明这个 PI 是好是坏。

例如 PI 可能：

- 提高正确 token 概率；
- 降低一个过度自信 token 的概率；
- 提高 entropy，让它暂时不要 commit；
- 改变 candidate ranking；
- 改变整个后续 denoising trajectory；
- 改变多个 token 的联合 reasoning mode。

所以：

\[
p_T(y_i)>p_S(y_i)
\]

不是“PI 有用”的充分条件。

一个好的 PI 可能反而告诉模型：

> 当前这个 token 不该这么确定，应该保留 uncertainty，等更多 context。

因此我们现在更愿意把 PI 的作用定义为：

> **让当前 denoising decision / transition 更合理，而不是简单让某个 token 更 confident。**

---

# 9. 当前最集中的 motivation

经过多轮收缩，我们目前认为最稳的 motivation 是：

> **现有 dOPSD 主要把 privileged future 转化为 token-level prediction supervision，但 dLLM 的核心优势和核心风险都发生在 parallel decision level：一个 denoising step 中模型要同时做多个相互关联的 token decision。单个 token prediction 更好，并不等价于整个 parallel decision 更好。**

可以写成：

\[
\boxed{
\text{token correctness}
\neq
\text{parallel decision quality}
}
\]

现有 dOPSD 更接近：

\[
\text{future PI}
\rightarrow
\text{better token prediction}
\]

我们真正想探索：

\[
\boxed{
\text{future / rollout PI}
\rightarrow
\text{better parallel decision}
}
\]

一句话版本：

> **Existing dOPSD distills future information into individual token predictions; we want to use on-policy privileged information to teach the model how to make better coordinated parallel decisions.**

这目前是最集中的主线。

---

# 10. PI 现在应该怎样理解

目前最合理的态度不是直接规定“future 就是 PI”，也不是规定“sibling 就是 PI”。

更合理的是：

> **PI 应该由我们希望修复的 failure mode 反推。**

如果目标是提升 parallel reasoning，那么 PI 应该能够提供：

> **当前并行 decision 在训练时不可见、但事后能够获得的额外 supervision。**

self-future、successful rollout、resolved sibling、final outcome 都可以是 PI 的 source。

但我们最终需要的是：

\[
\boxed{
\text{PI that reveals something about parallel decision quality}
}
\]

而不是“future 越多越好”或“future 越纯越好”。

目前一个可能的定义是：

> **successful self-future 是 raw PI source；从中提取出的 parallel coordination / decision-level information 才是真正用于训练的 PI。**

但这仍然没有最终定死。

---

# 11. “Teacher 应该教什么”——目前最关键的开放问题

我们已经明确：

### 不能只教：

\[
p_T(x_i)\rightarrow p_S(x_i)
\]

因为这仍然是 token-level。

### 也不能简单教：

\[
p_T(x_i\mid sibling)\rightarrow p_S(x_i)
\]

因为 inference 没有 sibling，且 conditional average 可能回到 marginal。

### 更理想的是：

teacher 利用 privileged rollout，给 student 提供某种 **parallel-decision-level supervision**。

我们讨论过几种可能形式：

1. joint/bundle quality；
2. parallel decision ranking；
3. step-level transition target；
4. shared reasoning/coordination state；
5. inference-time persistent coordination mechanism。

但目前都还没有定，因为很多版本看起来 trivial 或与已有工作过近。

当前真正的问题是：

> **Teacher 到底应该给 student 什么 supervision，才能让 student 在 inference 没有 PI 的情况下，真的表现出更准确、更协调的 parallel prediction？**

这仍是核心未解问题。

---

# 12. Pass@k >> Pass@1 为什么重要

我们讨论过一个重要现象：

如果模型：

\[
\text{pass@1}\approx31\%
\]

但：

\[
\text{pass@k}>50\%
\]

这说明：

> **模型已经能够生成正确 reasoning trajectory，只是这些成功 mode 在单次 rollout 中概率不够高。**

这使得 positive on-policy self-distillation 有合理性：

\[
\text{successful modes exist}
\]

但：

\[
\text{they are underweighted}
\]

所以 OPSD 不一定是在“教模型它不会的 reasoning”，也可能是在：

> **把模型已经偶尔能找到的成功 reasoning mode，从 tail 拉向主要 mode。**

但是：

> 仅仅把“正确 rollout”再学一次，并不能自动解决 parallel coordination。

这只能解释为什么 positive rollout 有学习价值，不能直接证明我们的新方法成立。

---

# 13. 我们正在跑的现象记录

目前已经开始基于 THU 代码做复现和 logging。

这些指标的定位是：

> **phenomenon probe，而不是用来预设某一个固定理论。**

主要记录：

## 13.1 基础 trajectory 信息

每条 rollout 每个 denoising step：

- 当前 state \(S_t\)；
- mask ratio；
- 本轮 reveal set \(C_t\)；
- 各 masked position 的 logits / top-k probability；
- entropy / confidence；
- 本轮实际 selected token；
- 最终 rollout correct / wrong。

## 13.2 PI gain

对 rollout token \(y_i\)：

\[
g_i
=
\log p_T(y_i)-\log p_S(y_i)
\]

但注意：

> \(g_i>0\) 不等于 PI 一定好，\(g_i<0\) 也不等于坏。

这里只是记录 PI 如何改变 student distribution。

## 13.3 Sibling gap

对同一步 \(C_t\) 中某个 token \(i\)：

原 student：

\[
p_S(x_i\mid S_t)
\]

把同一步其它 sibling 的 realized value 填进去：

\[
p_S(x_i\mid S_t,Y_{C_t\setminus i})
\]

记录：

\[
D_{\rm JS}
\left(
p_S(\cdot\mid S_t),
p_S(\cdot\mid S_t,Y_{-i})
\right)
\]

以及 rollout token 的 log-prob shift：

\[
D_S
=
\log p_S(y_i\mid \text{siblings})
-
\log p_S(y_i)
\]

注意：

- \(D_S>0\)：sibling 支持当前 token；
- \(D_S<0\)：sibling 反而否定当前 token；
- 真正的 coordination gap 更应该看 \(|D_S|\) 或 JSD，而不是先假设符号。

## 13.4 Teacher coordination gap

加入 THU PI 后：

\[
D_T
=
\log p_T(y_i\mid \text{siblings},PI)
-
\log p_T(y_i\mid PI)
\]

如果 future PI 已经隐式补了一部分 same-step coordination，那么可能看到：

\[
|D_T|<|D_S|
\]

但这只是待验证 hypothesis。

## 13.5 CoGain

对同一步：

\[
\text{CoGain}_t
=
\frac{1}{|C_t|}
\sum_{i\in C_t}\mathbf 1[g_i>0]
\]

然后和打乱 step 后随机组合的 token group 对比。

目的是观察：

> PI 的作用是否具有 step-level collective structure，而不仅仅是 independent token sharpening。

## 13.6 matched non-sibling control

为 sibling conditioning 找 control：

- 同样多给一个额外 token context；
- 但这个 token 不属于当前同一步 commit group；
- 尽量匹配 distance / reveal timing / confidence。

用来排除：

> “任何 future context 都会改变 distribution”

这种天花板效应。

---

# 14. 我们已经明确不应该做的 claim

以下 statement 当前都不够稳，避免继续沿着它们设计：

### 不要说：

> self-future 中 previous-state information 是无用/有害的。

没有证据。

### 不要说：

> PI 应该单纯降低 entropy。

不对。好的 PI 也可能提高 uncertainty。

### 不要说：

> sibling-conditioned teacher 可以让 student 在 inference 学到真实 causal dependency。

如果 inference 看不到 sibling，这个 claim 太强。

### 不要说：

> 只要 distill successful rollout 就能获得 parallel reasoning ability。

positive rollout 只说明成功 mode 存在，不等于 coordination mechanism 自动学会。

### 不要说：

> shared latent 本身是我们的创新。

已有工作已经用 shared latent 解决 parallel dependency。

### 不要把主问题重新变成：

> 哪些 token 应该什么时候 reveal。

这已经是一个非常拥挤的 decoding/scheduling 方向。

---

# 15. 当前研究问题，最推荐的版本

目前最值得继续的核心问题是：

\[
\boxed{
\textbf{How can on-policy privileged information teach a dLLM to make better coordinated parallel decisions, rather than merely better individual token predictions?}
}
\]

中文：

> **怎样利用模型自己的 on-policy privileged information，让 dLLM 学会更好的并行决策，而不只是把每个 token 单独预测得更好？**

这比“如何提纯 self-future”更稳，也比“如何决定 reveal order”更独特。

---

# 16. 当前可能的贡献位置

如果最后方法成立，理想定位应该是：

### THU / NUS

\[
\text{self-future PI}
\rightarrow
\text{token-level distribution improvement}
\]

### dependency-aware decoding work

\[
\text{dependency detection}
\rightarrow
\text{change reveal schedule / inference strategy}
\]

### latent generative work

\[
\text{new latent architecture}
\rightarrow
\text{explicit joint correlation}
\]

### 我们希望做到

\[
\boxed{
\text{on-policy PI}
\rightarrow
\text{parallel-decision-level post-training}
\rightarrow
\text{better native dLLM inference}
}
\]

理想情况下：

- 不依赖 external teacher；
- 不需要复杂 inference-time dependency test；
- 不只是 confidence calibration；
- 不只是 token KL；
- 不重新设计整个 latent generative architecture；
- 真正让 native dLLM 的 parallel reasoning accuracy 提升。

---

# 17. 目前最需要另一个 AI 帮我们解决的问题

请不要继续从已有 THU/NUS 方法上做局部 patch，也不要简单提出 contrastive loss、bundle ranking、confidence loss 等常规改法。

真正需要解决的是：

> **如果 teacher 在训练时拥有 successful rollout / future / resolved parallel decisions 等 PI，那么什么样的 supervision 或 model mechanism，能够让 student 在 inference 完全看不到这些 PI 时，仍然真正获得更准确、更协调的并行 prediction 能力？**

要求：

1. 必须直接针对 dLLM same-step parallel prediction；
2. 要区别于现有 reveal scheduling / dependency detection；
3. 要区别于简单 token-level dOPSD；
4. 要区别于 generic shared-latent diffusion architecture；
5. 最好是简洁、工程上可实现、能直接在现有 LLaDA/Dream + dOPSD code 上验证；
6. 核心目标首先是 benchmark 涨点；
7. 方法必须解释清楚为什么训练阶段的 PI 能在 inference 消失后仍然留下真正能力，而不是只产生训练时 conditional teacher advantage。

---

# 18. 当前最重要的一句总结

> **我们已经不再把问题理解为“如何找到更干净的 future PI”，而是“如何让 privileged self-rollout 教会 dLLM 做更好的 parallel decisions，并确保这种能力在 inference 没有 PI 时仍然存在”。**

这就是目前整个讨论收敛到的位置。
