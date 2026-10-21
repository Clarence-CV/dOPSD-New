# Selective Draft / Joint OPSD：研究概览

更新：2026-10-18。合并原研究记录、单源方法、工程计划与数学审阅，保留修订后的结论。已运行的是 **LLaDA Joint-v1**；用户已接受的下一版是 **mini + root_success_v2**，尚未实现或训练。历史候选不再视为当前执行要求。

## 1. 要解决的问题

dLLM的双向attention已让位置共享上下文，但同一步并行提交时，Receiver的最终预测未必再条件于同伴本次实际选定的token identity。研究目标是改善这种条件协调，最终用**准确率与实际生成时延**判断价值。

THU self-future PI提供后续去噪/未来block信息；sibling PI只在同一真实状态S中揭开指定同伴，其他位置保持不变。mini是block-causal：更晚block的reveal不能直接影响当前block，因此同block的sibling PI是有结构理由的迁移方向，效果仍需验证。

## 2. 方法演进

| 阶段 | 核心设计 | 当前定位 |
| --- | --- | --- |
| 无训练全位置L24 feedback | 原LM head读出hard draft，手工加embedding残差 | 具体试验负收益，暂停 |
| Early Commitment / Partial Loop | 早锁Anchor，或重用部分层增加条件传播 | 历史候选，未证明有效；不加入当前主线 |
| 单源Selective Draft | L28 readout，最高置信位置以0.8门控发送soft message；单位置条件Teacher | 已实现并有初试；`single_soft`流程保留 |
| LLaDA Joint-v1 | 最多4位置同时发消息；每轨迹抽4状态，每状态续写8次，选择完整tuple并LOO认证 | 已完成100次更新，速度有正向结果，准确率增益不确定 |
| mini / root_success_v2 | confidence>.95决定全组；从prompt最多8次、首次答对即停、抽1状态、使用自身最终tuple | 当前批准的下一版计划；取消主线额外续写与多tuple搜索 |

soft guidance是**proposal**：最终层仍可改变组员token，也可暂不提交。它不是early commitment；若以后锁定硬Anchor，必须保证内部条件与最终写入一致，重新定义错误锁定与提交额度。

## 3. 两阶段结构与Joint目标

```text
真实去噪状态 S → 冻结主干到选定层
  → readout adapter → 原final norm → 共享LM head → r_i
  → 选择G；m_i = sum_v r_i(v) * input_embedding[v]
  → h'_i = h_i + Injector(h_i, m_i)，全部消息先算好再同时注入
  → 冻结的剩余原生层 → 最终logits → 正常解码提交
```

- **阶段一**只训练readout：`mean_i KL(q0_i(S) || r_i(S))`。q0是同状态原模型末层分布，不是LOO；读出旁路不应改变原模型输出。
- **阶段二**冻结readout与主干，只更新injector。历史LLaDA的rank64模块分别有524288和786432参数；mini必须按其维度重建，不能复用旧权重。
- Joint参考pi来自一条完整、答案验证通过且与S相容的轨迹，不能跨轨迹拼接token。对每个i，Teacher只揭开`G除i`的参考token，i仍MASK；所有`argmax(q_i)==pi_i`才接受。
- 优化`mean_i_in_G KL(q_i || p_student_i)`：完整词表、组内位置平均、有效state间平均。Teacher目标detach；Student只接收S与自己的soft messages，不接收pi或成功轨迹尾部。
- LOO认证局部坐标自洽，终局答案由另一verifier判断；均不保证Student实际提交的tuple满足同一性质。Joint loss目前只监督G，组外Receiver虽受attention影响，却没有直接损失。

## 4. 数学审阅：保留的结论与修正

单源`teacher.py`的模式与Joint整组LOO分开：

| 模式 | Teacher在源位置A揭开什么 | 边界 |
| --- | --- | --- |
| `realized`（单源默认） | 本条Student轨迹中A最终得到的token | 可能来自后续步；不保证正确或Student能预测它 |
| `argmax` | 当前readout最大概率token | 可能与最终提交不同 |
| `mc` / `topk` | 多候选条件分布的概率混合 | MC按频数，top-k按保留质量重归一化；不能平均logits |

1. **mixture可能弱化信号，但不是已证明必然无效。** 自洽联合分布满足`sum_a q_A(a|S) q_B(b|S,A=a) = q_B(b|S)`；readout接近q_A时，mixture接近原边缘预测。实际dLLM未必联合自洽，必须实测mixture/argmax/realized相对q0的差异；旧稿“必然学不到”已撤回。detached权重下，平均hard条件KL与mixture KL对Student梯度相同，标量不一定相同。
2. **soft message不是新外部信息。** 它是hidden的确定函数，可能让已有信息更易被尾层使用，也可能被injector忽略。固定hidden及其他输入，只改变消息再测Receiver；loss下降不足以证明同伴信息被利用。
3. **主干冻结仍能学习。** injector参数会改变尾层输入和最终预测。“多采样找到正确参考再学习”是合理收益来源。Teacher很尖锐时，forward KL梯度接近参考token CE；不能把收益全部归因于LOO或协调。
4. **不可见参考有学习限制。** 同一S对应不同参考而Student无法区分时，它可能只能拟合平均目标。soft guidance也不保证最终组员身份与Teacher条件一致。
5. **撤回过强推断。** raw LM head读不出不等于早层没有identity；patch恢复率不是学到的injector能力上限，不能与readout准确率相乘预测收益；下一次前向不是“免费”；不能用事件数替代实测加速，或用小样本Reveal数值推出普适门控阈值。

## 5. 已有证据及范围

| 实验 | 主要观察 | 能说明什么 |
| --- | --- | --- |
| 历史GSM8K sibling response | 固定预算均值约−.130 nat、负向26%；阈值约+.017～.02、负向7% | 协议改变条件冲突分布，不能换算成任务涨点 |
| 受控Reveal，128个Receiver | 正确Anchor：log-prob +.099，100→102答对；错误Anchor：−2.095，降至58答对 | identity有影响，错误条件危害明显；不是大幅任务增益证明 |
| 原LM head逐层读出 | L28 55.47%，L31 97.66% | 该口径下晚层更可读，不证明早层无信息 |
| Activation patch | L8/L16/L24/L28/L31恢复约104%/74%/37%/19%/6% | 条件传播探针，不是accuracy增益或严格上限 |
| 无训练L24 hard feedback，GSM8K500 | 30.8%→26.6%；抽查L24/L32 agreement约18%；零注入parity通过 | 具体方案负收益；draft错误与表示不对齐尚未区分 |
| 单源训练初试，MATH-500前200 | tau=.9时base41.0%→guided37.0%；其他阈值也未见明确增益 | head改善不自动带来任务收益；见[初试报告](../results/selective_draft/math_prelim_v3_REPORT.md) |
| Joint-v1，MATH-500全500 | base34.4%→35.0%；步数100.422→82.232；时延5.533s→4.927s | NFE约−18.1%、时延约−11.0%；32修复/29损伤、净+3题，配对p约.80 |

前几项来自2026-10-11研究记录，原trace/统计定义仍需回查，不能与不同prompt、长度、模型或题集直接比较。早期“L24可读率近0%”与独立抽查“18% agreement”口径不同；30.8%与历史约79%的GSM8K结果也未统一协议。

历史GSM8K背景一并收于此：300题greedy sweep选step448，78.3%→82.7%，最终step1344降至18.0%；同一测试子集选点，不能视为独立held-out效果。T=.9下固定预算75.0%→82.0%，阈值.9为79.3%→79.7%。future PI使兄弟与对照的绝对reveal response分别缩小约32%/38%，兄弟特有部分仅约+.010 nat；师生提交集合重合约54%，是排序变化线索，不能推出已学会协调。单seed、300题与稍后提交的对照token限制因果解释；数值/图保留在[历史结果](../results/pipeline_gsm_20261002/SUMMARY.md)及同目录产物中。

Joint-v1访问327题、178题贡献监督，100updates合计525有效state、2019位置；6000是题池大小。原评分有等价答案误判；相同输出换KLASS scorer为35.2%→36.6%，不能混用新旧分数。详见[Joint分析](../results/joint/math_v1_analysis.md)和[解码审计](decoding_protocol_audit_zh.md)。

Joint加速主要来自fallback每题51.30→31.63，输出长度约244不变；每步成本反而55.1→59.9ms。Teacher参考概率中位数.9984，draft/参考一致率71.4%，支持“学会更确信成功参考”这一待检验解释。checkpoint25/50/75未做任务评测，不能说100步最优或已收敛。逐题修复/损伤案例并入上述Joint分析，完整分组统计保留在[详细指标](../results/joint/math_v1_detail.md)。

TACC[续写审计](../results/joint/rescue_audit.md)：226条自然错轨迹中77条找到LOO通过参考，贡献29.4%的监督位置，却花掉73.8%的续写NFE；每题预算是4×8条额外续写。原本成功轨迹的404个抽样state中，至少287个用自身最终tuple就能通过LOO。这支持简化采样，但不证明删去搜索后能保留全部收益；“救回监督”不等于模型救回相同题目。

## 6. 诊断指标怎么解释

| 指标 | 定义 / 用途 | 边界 |
| --- | --- | --- |
| PSA，并行—顺序一致率 | 同步提交的B是否等于只揭开实际A后Teacher的argmax | 两者可能一起错，不是正确率证书 |
| D_S / reveal response | `log T_B(x_B given S,A=x_A) - log T_B(x_B given S)` | 旧代码键`pmi`保留兼容；无联合一致性假设就不是严格PMI |
| re-mask regret | 写入本步其他token后重新mask B，检查Teacher是否改变预测 | 稳定的错误也能低后悔 |
| induced-commit fidelity | 对guided新触发提交的B，从同一前态关闭guidance继续到B真正提交，比较token | 比较baseline决策，不是ground truth，也不一定只需下一步 |
| readout fidelity / coverage | 与末层KL、argmax一致率、过阈值组覆盖、概率保真 | confidence不是语义正确率 |
| 消息/patch干预 | 固定S/G及Receiver自身消息，改变同伴消息，观察尾层响应 | 非零响应只证明依赖；有益还需自然消息优于off/扰乱对照 |

固定分析state与各自on-policy轨迹分开统计，误差区间按题计算。现有induced-commit fidelity只支持有完整元数据的greedy阈值trace，其他情况显式报告不支持。NFE是调用次数，实际时延必须包含readout、soft embedding、注入与cache成本。

## 7. 实现必须保持的契约

- 原生final norm/head仅应用一次，同位置输出不套AR右移loss；层号指`blocks[layer-1]`输出。backend检查权重版本、hidden/vocab、token IDs与tied head。
- readout初始为恒等残差，injector初始输出零；不能上下投影都零初始化。冻结base参数不等于关闭尾层输入梯度，Teacher才走no_grad。
- scoped hook及时移除，Teacher/off无残留；off/readout-only/zero-injector做真实parity。activation checkpointing或compile另验证重算hook与梯度。
- state保存提交前token、绝对位置、block、attention、G、policy与RNG；同一次更新内固定policy。逐项检查已知token/MASK/EOS相容性，不把未来token送入Student。
- full-soft message和完整目标KL用FP32累加；chunk省显存不改变目标。top-k日志不能冒充完整训练分布。
- 多卡按全局有效state数归一化；空rank仍同步，全局无目标不更新optimizer/scheduler。完整checkpoint含optimizer、各rank RNG、cursor、协议hash；snapshot只供初始化/推理。
- base/方法匹配prompt、题集、长度、block、温度、停止、scorer和backend。固定预算steps与阈值实际NFE分开；计时关闭详细trace，诊断另跑，模型评测在TACC计算节点。

CLI、现有模式和恢复约定见[流水线README](../selective_draft/README.md)，Joint-v1见[JOINT.md](../selective_draft/JOINT.md)。本概览不重复命令和旧默认配置。

## 8. 当前下一版：mini / root_success_v2

1. **先复现官方mini MATH**：block32、T0、confidence>.95，每block steps32，实际NFE自适应；完整prompt/cap/scorer仍有缺项，不用自定2048或旧THU prompt宣称复现93.22%。
2. **重新选层、训练readout**：全层confidence/entropy/identity稳定性诊断，不指定L16/L17；64题验证＋64题一次确认，head拟合q0而非LOO。G为早层所有>.95位置，少于2不通信，无固定4人组。
3. **简化在线采样**：训练root T=.9，最多8次含首次，第一次答案成功即停；成功轨迹均匀抽1个有组state，自身最终tuple做LOO。失败换题，不额外续写、不重抽、不做距离排序；明确成功筛选偏差。
4. **扩大独特题覆盖**：6000题池第一遍不放回，dev/analysis不回流；每25有效updates保存，dev128粗筛、dev500选点，按贡献题与SU决定续训，不只比较100steps。
5. **验证真实消息路径**：记录注入、下一层输入、Receiver响应和梯度；修查mini的T0采样、MoE eval的no_grad、block-causal mask及EOS返回差异。CPU fake不代替真实验证。
6. **收益与归因分开**：先跑一条主线；需要主张LOO独立价值时，再用相同数据/模块对比LOO-KL与参考token CE，并做消息干预。主干冻结不排除正确路径学习。

短版交接见[mini handoff](llada2-mini_handoff.md)，实施细则和官方链接见[mini engineering plan](llada2-mini_engineering_plan.md)，跨仓库参数差异见[decoding audit](decoding_protocol_audit_zh.md)。本概览不授权新作业，也不恢复Partial Loop、全位置hard feedback或Receiver新损失为默认实验。

## 9. 相关工作与创新边界

保留影响设计的历史检索入口；正式比较前仍须核查具体版本与协议。不能把增加head、soft embedding或几层循环本身当作新贡献。

| 方向 | 来源 | 对本项目的约束 |
| --- | --- | --- |
| 中间预测反馈 | [DSLP](https://arxiv.org/abs/2110.07515)、[DiscoLoop](https://arxiv.org/abs/2607.00341)、[Tuned Lens](https://github.com/AlignmentResearch/tuned-lens) | readout与embedding feedback有先例，需证明具体监督/计算收益 |
| 联合提交 / 条件修正 | [CoCommit](https://arxiv.org/abs/2607.04469)、[ADJUST](https://arxiv.org/abs/2509.22738)、[MRP](https://arxiv.org/abs/2605.18817) | 重算尾层或外挂条件预测器须比较重合与成本 |
| 揭示调度 | [DAPD](https://proceedings.mlr.press/v306/kim26t.html)、[AXON](https://arxiv.org/abs/2606.04236)、[Learning Unmasking Policies](https://proceedings.mlr.press/v306/jazbec26a.html) | 高置信不等于高条件价值；选位置已有相关工作 |
| 循环深度 | [LoopMDM](https://arxiv.org/abs/2605.26106)、[Recurrent-Depth Samplers](https://proceedings.mlr.press/v306/geiping26a.html) | 若恢复loop，需同等计算的无Anchor对照与实测时延 |
| PI / 成功路径蒸馏 | [THU self-future](https://arxiv.org/abs/2606.18195)、[另一篇dOPSD](https://arxiv.org/abs/2607.04428)、[GKD](https://arxiv.org/abs/2306.13649)、[STaR](https://arxiv.org/abs/2203.14465) | 区分PI来源、成功筛选、可见条件与普通参考学习 |
| 共同终点 / 部分揭示 | [CLLMs（ICML 2024）](https://proceedings.mlr.press/v235/kou24a.html)、[Glancing Transformer（ACL 2021）](https://aclanthology.org/2021.acl-long.155/) | AR Jacobi收敛论证不能直接移植双向DLM；Student直接见参考与Teacher-only PI不同 |
| 监督可靠性 / 可迁移性 | [SelFusion（ACL 2026）](https://aclanthology.org/2026.acl-long.1008/)、[PI Distillation（ICML 2026）](https://openreview.net/forum?id=ebZcMQImhG) | 更多已知token未必更正确，Teacher受益也未必可由Student复现；旧Receiver门控建议仅保留为详细计划§11.4的可选后续 |
| 关键位置 / 推理训练 | [Critical Tokens Matter（ICML 2025）](https://proceedings.mlr.press/v267/lin25j.html)、[d1（NeurIPS 2025）](https://openreview.net/forum?id=7ZVRlBFuEv)、[Fast-dLLM（ICLR 2026）](https://openreview.net/forum?id=3Z3Is6hnOT) | 关键位置与原生模型训练有先例；不能直接搬AR概率、跨协议成绩或把阈值加速等同于纠错 |
| 机制 / 外部draft | [Steering Without Breaking](https://arxiv.org/abs/2605.10971)、[DFlash](https://proceedings.mlr.press/v306/chen26bv.html)、[DSpark](https://arxiv.org/abs/2607.05147) | 机制诊断和外部speculative pipeline不冒充同协议对照 |
| 强模型 / 置信度训练 | [mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)、[mini-CAP](https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP) | 分别复现原生解码；置信度提升不等于协同推理 |
