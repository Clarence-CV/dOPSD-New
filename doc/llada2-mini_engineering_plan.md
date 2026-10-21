# LLaDA2-mini engineering plan：Joint-reference OPSD 的第二模型验证

日期：2026-10-18。交付对象：Claude / 后续实现者。状态：**源码与资料审阅后的实施规格；尚未实现 mini backend，尚未加载 mini 权重，尚未提交训练或评测。**

本轮修订（2026-10-18用户已接受）：**主训练改为 `root_success_v2`：每题从prompt最多尝试8条完整Student轨迹，首次答案验证通过即停；从该成功轨迹的合格状态中均匀抽1个，用自身最终tuple做LOO，通过才训练。取消主线同状态K8续写、候选距离排序和多tuple搜索。** 节省的预算优先覆盖原6000题池中尚未访问的题，而非重复已有少数题；按独特题数、有效监督、GPU秒和固定dev曲线选择checkpoint，不把100steps当收敛标准。历史 `same_state_v1` 仅保留为证据和可选消融。

其余前置条件保持：先复现官方mini MATH，再训练本方法；G绑定官方confidence .95且无人数预算；层位通过全层诊断产生；下次真实mini运行记录消息传播。官方完整MATH recipe尚有缺项，不能把计划写成已复现。新增采样器也尚未实现，本次仅修订文档和资料清单。

本仓库基线：`joint` 分支，审阅时HEAD `0cd4a05`；Joint实现提交 `16c07e4`。开工先读[交接简版](llada2-mini_handoff.md)，本文件作为详细规格查阅；研究历史已合并到[研究概览](Selective_Draft_OPSD.md)。官方来源及固定版本直接见§2、§17–18；实际运行时将代码、数据和环境指纹写入run/meta，不再另存文档目录的来源JSON。

## 0. 执行结论、证据等级和禁止混淆的事情

### 0.1 当前唯一主路线

1. **第一交付物是官方MATH复现报告与冻结的baseline manifest。** 完成题集、prompt、scorer、采样/EOS/长度、模型revision、accuracy、实际NFE与时延的对齐之前，不启动readout或Joint正式训练。[M1][M3][M7] 当前官方完整MATH recipe仍有缺项，见§0.4；不能用自选MATH-500/boxed prompt/Math-Verify/2048去宣称复现93.22%。
2. 固定`inclusionAI/LLaDA2.0-mini`正式版。模型论文§6.1与模型卡规定block32、T0，论文规定threshold .95；HF代码`steps=32`是每block计划，实际NFE自适应。[M1][M3][M7] 原LLaDA-Instruct官方MATH分块配置为block64，属于另一个模型/recipe。[L1]
3. 组G按早层draft候选的confidence **>同一个0.95**决定，全部纳入，不设人数预算/上限；无过阈位置时，沿用官方top1 fallback的决定，但单人组没有LOO同伴，因此跳过通信。最终commit仍由最终层以同一规则决定，早层G与最终commit集合不保证相同。先检验这一版；本轮不跑较低组选取阈值。[M3][M7][F2，迁移到早层是A]
4. 注入深度来自一次全层confidence/entropy/KL/identity稳定性诊断，按§7.1冻结；**不存在预设的L16/L17，也不从旧L28按比例换算**。[E1][E2][E3] 层位尚未产生时配置必须报未解析，不能落到默认值。
5. Joint使用§6.4的 `root_success_v2`：训练Student从prompt采样，所有attempt先固定T=.9、最多8条、首次答对即停；成功轨迹抽1个有同伴组的state，自身最终tuple做全员LOO与full-vocab forward KL。部署/官方复现仍T0。T=.9和root重试预算借鉴THU作者代码[T3][T6]，是跨模型训练适配，不是mini官方MATH配置；失败轨迹只进诊断账本，不做额外分支续写。
6. readout保留独立64题验证+64题一次确认，按checkpoint选取；Joint只更新injector，每25个有效updates保存。题池先不放回遍历；首次规划50→100→200→400只是逐段上限，以pilot测得的贡献题率和成本决定能覆盖多少题，完整6000题不是首轮承诺。每个checkpoint同时保存覆盖与SU曲线。
7. 下次mini实现必须记录注入前后、下一层输入、后续接收位置变化和消息梯度。实现日志与正式计时分开。旧LLaDA的checkpoint审计保留为历史补证工作，不再抢在mini官方baseline复现之前。

### 0.2 每项设置如何使用引用

本计划使用三种标记：

- **[O] 官方直接支持**：公开模型配置、官方实现、论文明确使用的数值或规则。
- **[A] 有来源的适配**：来源支持原理或做法，但跨模型、跨目标的适用性仍需验证；必须说明改了什么。
- **[E] 本项目工程决策**：具体样本数、预算门槛、层号、模块学习率等没有官方最优值；列出依托的来源、选择理由和推翻条件，不能写成“论文证明此数值最优”。

用户要求每项配置有官方/顶会依据，不能通过给任意超参数贴一篇论文来满足。**旧group=4与L16/L17没有官方依据，已从mini主配置撤回；rank=64、dev128、最大400 updates等资源配置仍没有公开论文证明最优。** 对这些项，本计划明确给出已有证据和预注册验证流程。官方与论文也会互相冲突，以下实际代码审计优先解释冲突，不掩盖冲突。

### 0.3 开工前必须解决的 P0 问题

| 问题 | 本次找到的证据 | 必须完成的动作 |
| --- | --- | --- |
| HF mini 的 T=0 并非 greedy | 固定 revision 的 `_sample_with_temperature_topk_topp` 无零温 argmax 分支，始终 multinomial；官方 issue #6 承认并指向 2.1 的修复 [M3][M8][M9] | 仅移植正确的零温采样分支，保留 2.0 权重与架构；做确定性测试；原始 HF 行为单独标识 |
| eval 模式的 MoE 专家梯度被截断 | `SparseMoeBlock.forward` 在 eval 调用带 `@torch.no_grad()` 的 `moe_infer` [M3] | 为 injector 下游建立可微专家路径并验证输入 Jacobian；不能只检查 injector.grad 非零 |
| 模型加载类不兼容 | 2.0 `AutoModel` 映射到不带 LM head 的 backbone，现有 loading.py 恰好用 AutoModel [M2] | mini 使用 AutoModelForCausalLM；新 backend，禁止把模型名直接塞进 llada backend |
| attention、block 边界不兼容 | mini 要 4D block-causal mask；绝对位置从 0 分 block，prompt 可能占当前 block 一部分 [M3] | 新建 block grid/state contract，不能只设置 `window_mode=growing` |
| 原 readout 不能复用 | LLaDA：32层/4096维；mini：20层/2048维/157184词表 [M2] | 重新训练 readout；禁止 L28 和旧模块权重迁移 |
| EOS 返回范围不一致 | HF early-return 分支含 prompt，普通返回只含 completion；内步 EOS 检查存在触发条件限制 [M3] | 统一返回结构、停止语义、prompt 排除和未解 MASK 检查 |
| 不能按 1.4B 参数估显存 | MoE 活跃参数约1.4B，但权重索引 total_size=32,511,286,784 bytes [M1][M4] | 按约30.28 GiB 权重 + 激活/缓存/logits规划，先测 A100-40GB 单卡峰值 |

### 0.4 官方MATH复现是前置条件：已知值、缺项、验收

2026-10-18重新在线核查了模型卡/模型API、LLaDA2.X完整公开目录、dInfer完整目录、官方论文、HF discussions与GitHub issues。HF模型revision仍为`dad945...`，dInfer仍为`1ffeb961...`。目前找到的公开证据如下，**没有找到可直接对应93.22%的完整官方MATH评测脚本/逐题输出/实际NFE表**：

| 项目 | 官方可核实内容 | 本项目处理 |
| --- | --- | --- |
| 模型与报告成绩 | LLaDA2.0-mini正式版，MATH 93.22% [M1][M7] | 作为待复现的原始报告，不把它改称MATH-500成绩 |
| B/T/tau | B=32、T=0、tau=.95 [M7 §6.1] | 三项固定，不为了追分调参 |
| steps参数 | HF示例steps=32；循环在每个block内部 [M1][M3] | 配置命名steps_per_block，禁止解释为整题32NFE |
| block位置原点 | 绝对序列0开始，每32位置一个block [M3] | 保留prompt尾部和首生成block共享的情况 |
| 最低提交 | B=steps=32时schedule每步1；先接受所有>.95候选，数量不足才top1 [M3] | 这是阈值解码+进度fallback，非固定预算4或32 |
| prompt | 官方chat template已公开 [M5] | MATH具体user文本、system、few-shot/CoT样例未在上述材料中找到，不用旧LLaDA boxed文本代替官方事实 |
| MATH题集 | 论文称MATH并引用原数据集 [M7] | 实际split、样本ID、重复次数/汇总方式待核对；93.22不是500题单次二元评分的可能精确增量，但不能仅据小数倒推出官方题数 |
| 长度 | 模型卡示例512、Best Practices建议32768；dInfer示例2048 [M1][I1] | 均不等于已查明MATH表格使用的cap；缺项保留null，不默认2048 |
| scorer | MATH表中报告accuracy [M7] | 原抽取/归一化/等价验证代码与版本待核实，Math-Verify只作交叉审计 |
| T0行为 | HF固定版本仍multinomial；作者确认2.1修复 [M3][M8][M9] | 原代码行为与论文greedy意图分开记录；修复必须有差异清单，不能静默宣称bitwise复刻 |
| EOS/返回范围 | HF分支有含prompt/只completion的差异 [M3] | 原始raw结果与completion视图都保留；停止语义修补会改变NFE，要有逐步对照 |
| 公开推理框架 | dInfer脚本仅循环GSM8K/MBPP、tau=.80、B32、长度2048 [I1][I4] | 不能把该脚本参数当成mini论文MATH recipe |
| 官方MATH平均steps | 当前上述公开材料没有逐题NFE/平均NFE基准 | 本地完整记录，不能捏造官方目标，也不能把32当平均steps |

缺项有公开旁证：HF discussion #3仍在请求复现脚本，没有提供完整官方答复；LLaDA2.X issue #4报告无法复现，公开回复只有其他用户附和；dInfer issue #29也询问更多benchmark YAML，后续维护者回复处理的是MBPP故障，未提供MATH recipe。[M10][M11][I5] 这些issue是信息缺口的佐证，不是模型效果无效的证明。

**执行顺序与停止条件**：

1. 建`official_math_manifest.json`，逐项填入权重/代码/tokenizer/scorer revisions、题集ID与题文hash、全部prompt token IDs、few-shot顺序、cap、EOS、sampler、threshold比较符、fallback、dtype、attention/cache与环境。未知字段保留`unresolved`，不得用自定义值填上后标`official=true`。
2. 原HF代码设为`reference_raw`；论文greedy意图+官方已确认修复设为`reference_corrected`。先在少量完全相同题目/输入上记录差异原因。HF修复返回prompt切片通常不改变模型预测，但EOS触发检查、禁止MASK、tie-break等改动可能改变实际轨迹，必须单独开关/标记。
3. 先做基础加载与单步mask/logits一致性，再做逐步trace：每题保存prompt_len、物理block边界、每步candidate/confidence/commit集合、是否fallback、有效长度、EOS位置、stop reason。记录`denoise_calls`、prefill/cache-seal calls、每block NFE、每步提交数分布、fallback比例、总NFE和GPU latency。进度条的step、optimizer step和denoise NFE必须分字段。
4. 用确认的官方数据/recipe跑完整baseline；报告`correct/n/accuracy`、解析失败数、截断率、每题NFE与汇总均值/中位数/p90、有效输出长度、耗时。若官方有逐题输出/trace，逐题核对而非只对齐一个百分数。相同recipe仍有差异时检查精度、kernel、seed和批次；不通过调tau/block/cap/scorer把分数“凑到93.22”。
5. 只有官方recipe缺项已解决、accuracy复现结果及差异说明可审阅、我们适配后的guidance-off与冻结参考decoder逐步一致，才设`official_math_reproduction.status=passed`。官方没有公开的NFE数值只能标为“已测量、无法与官方数值核对”，不能写已精确复现其NFE。
6. 当前信息状态为`awaiting_official_math_recipe`。可继续实现只读审计、基础backend与trace、准备问题清单；**不启动readout/Joint优化，也不以自定义baseline替代这个前置条件。** 若以后取得的只是可复现替代recipe，要明确这是范围变更，不能自动降低本次用户要求。

后续需向官方核实的内容已由本表列明；本次不代用户发issue/邮件。工程计划可以完整交付，但不能承诺尚未公开的参数或实际分数已经被复现。

### 0.5 为什么原LLaDA的64与mini的32同时正确

| 来源与任务 | block | generation/step语义 |
| --- | ---: | --- |
| 原LLaDA-Instruct官方EVAL的MATH分块评测 [L1] | 64 | gen_length512；同页另有pure diffusion B512，不能混成唯一配置 |
| KLASS的LLaDA MATH脚本 [K2] | 64 | KLASS自己的阈值/历史KL协议 |
| 本仓库已经运行的Joint-v1、THU recipe | 32 | 旧实验事实；不代表原LLaDA官方MATH统一值 |
| LLaDA2.0-mini官方卡与论文 [M1][M7] | 32 | steps32是每block最低提交schedule/上限；高confidence可一次提交多个 |

因此mini的32有模型官方直接依据；本次需要纠正的是把不同模型/评测的block混作一套惯例，以及把官方解码参数已知误写成完整MATH复现已经具备。

## 1. 当前仓库事实：迁移的是哪一个方法

### 1.1 代码分层与改动边界

| 路径 | 当前角色 | mini 计划 |
| --- | --- | --- |
| `d-opsd/d_opsd_train.py`, `d_opsd_trainer.py`, `d_opsd_config.py` | THU 主线：on-policy rollout、self-future PI、LoRA/训练配置 | 作为官方协议与历史 baseline 来源；不把其 trainer 当 Joint 入口 |
| `eval/generate.py`, `eval/math500.py`, `eval/parser_helper.py` | THU 风格生成、prompt、评分 | 保留历史对照；mini 新评分和 block 规则需显式配置 |
| `selective_draft/model.py`, `modules.py` | 冻结主干、readout、soft-message injector、scoped hook | 保留模块数学定义；扩展 mini 的可微尾层和投影接口 |
| `selective_draft/joint.py` | 历史same_state_v1：同state续写、tuple筛选、LOO、组内KL | 新增root_success_v2采样分支；复用LOO/loss，不沿用continuation银行/距离排序；旧模式显式隔离 |
| `selective_draft/generation.py`, `trace.py` | LLaDA 固定/增长窗口、阈值/固定预算、轨迹 | 不能以“可导入”为通过；mini 需要新的绝对 block 划分 |
| `selective_draft/loading.py`, `backends/` | 当前 llada / illada / fake_llada | 增加 llada2_mini，分离 AutoModelForCausalLM 加载 |
| `training.py`, `checkpoint.py` | 手工 trainable-only 梯度求和、有效 state 归一化、恢复 RNG | 复用并加多卡空 rank、采样版本、scheduler、环境指纹验证 |
| `data.py`, `evaluation.py`, `config.py` | split、prompt、答案判分、计时与配置验证 | 增加新 recipe、独立 test revision、verifier、严格未生效字段检查 |
| `tacc/joint.slurm` | 默认3卡训练，之后每卡并行测一种模式，默认 test | 新脚本默认 dev；禁止同卡多进程互相污染 latency；运行 manifest 校验后才可复用结果 |
| `analysis/`, `tests/test_selective_draft_*` | 轨迹诊断与 CPU fake 回归 | 复用统计框架；现有 fake 是旧 LLaDA 结构，不足以证明 mini 正确 |

仓库相对 `thu/main` 的主要新增工作在 `selective_draft/`、轨迹分析、TACC流水线和结果文档；原 d-OPSD 不是被全面替换成 Joint。旧 `CLAUDE.md` 中不存在的入口、Dream限制、`steps==tokens` 不作为本计划的不变量。

### 1.2 最新结果要如何继承

依据 [原实验报告](../results/joint/math_v1.md)、[深入分析](../results/joint/math_v1_analysis.md)、[实际运行配置](../outputs/joint_math_v1/train/resolved_config.json)、[评分审计](../results/joint/klass_comparison.md)：

| 指标 | base | Joint-100 | 可作出的判断 |
| --- | ---: | ---: | --- |
| MATH-500，原评分 | 172/500，34.4% | 175/500，35.0% | 净+3题；32修复/29损伤，p约0.80，不能称准确率显著提升 |
| 同输出改用 KLASS 评分 | 176/500，35.2% | 183/500，36.6% | 判分敏感；不能拿新分数与旧分数混算 |
| 平均去噪步 | 100.422 | 82.232 | -18.1% |
| 平均延迟 | 5.533s | 4.927s | -11.0%，现有协议中的真实正向信号 |
| readout-only | 与 base 500/500 输出一致 | 延迟约5.934s | parity 成立；readout 确有成本 |

训练100次有效更新，另有9次全局空更新；三卡共访问327道不同题，178道贡献目标；1308个候选 state 中525个通过，监督2019个组内位置。不是“用6000道题训练了一个epoch”。训练损失随新 state 波动，既不能由其未下降判定没学到，也不能由 delta norm 继续变大认定还会持续变好。

2026-10-18已回TACC只读核查，1640个文件SHA256与本地全部一致，逐状态关联与训练日志校验通过；见[续写救回审计](../results/joint/rescue_audit.md)及[机器可读统计](../results/joint/rescue_audit.json)。226条自然错轨迹中，88条至少有一次正确分支，77条至少有一次LOO通过；它们贡献154个有效state、593个监督位置（全部的29.4%），消耗371419次续写前向（全部的73.8%）。该77/226=34.1%使用的是每题4×8=32条额外续写，不是单状态K8。

全部额外续写503489 NFE，自然rollout31934 NFE，前者为后者15.77倍；这是forward调用计数，不是实测GPU时间比例，且未计group readout/反向等开销。成功轨迹来源的404个抽样state中，至少287个自身最终tuple已在保存的认证里通过，40个已失败，77个未检测。说明可以直接使用自身参考，也说明替代参考并非毫无贡献；没有证明额外搜索是必要机制。

本次用户据此接受root_success_v2。新旧采样协议改变了成功筛选、训练state与参考分布，不能将LLaDA-v1与mini-v2的差异全部归于模型结构；也不能直接将“救回77条训练监督”写成“模型推理救回77题”。

### 1.3 已运行的 Joint-v1 定义（组大小4是历史配置）

```text
Teacher = 冻结的原始模型，无 readout/injector 干预
Student = 同一主干 + 冻结 readout + 可训练 injector
G = 当前真实去噪 state 中，当前 block 的 MASK 位置里 draft confidence 前4
    (不足4取全部，少于2不通信；不是最终 commit set；mini改为§7.0选定的规则)
m_i = sum_v r_i(v) * input_embedding[v]
delta_i = injector(concat(h_i, m_i))
同时在 G 上注入 delta，剩余原生 attention 层传播，再按末层概率提交

候选 = 从完全相同 state 出发的 K=8 条完整 Student continuation
有效候选 = 答案验证通过 + 保持已知 token + 有效停止/长度语义
参考 pi = 在 G 上取整条 continuation 的完整 tuple，不跨候选拼接
q_i = Teacher(state，仅把 G\{i} 揭开为 pi；i仍是MASK)
认证 = 对所有 i，argmax(q_i)==pi_i
loss(state) = mean_i_in_G KL(q_i || p_student_i)
loss(update) = 全部进程实际接受 state 的平均
```

双向 attention 原本已存在；新增的是 draft identity 条件及其训练。G 成员不提前锁定，也不额外占 commit 额度；仍可等后续步提交。不能把本实现称为已实现的 early commitment。Joint 自有定义来自本仓库；[T1][T2][G1]支持 on-policy/PI 蒸馏的框架，不证明这个具体目标的正确性或有效性。

### 1.4 THU self-future PI 不能直接跨到mini

这不只是换tokenizer或学习率：原LLaDA全局双向attention允许当前block看到后续block的reveal；mini原生block-causal attention禁止当前block读取未来block。**如果THU式PI只修改更晚block，保持mini原生mask，那么当前block的Teacher logits在数学上应不变。** 数值差异只能来自实现/随机性，不能当PI收益。[M3][T1]

Joint sibling PI只揭示同一当前block内同伴，因此仍处于合法可见范围。这是迁移Joint有价值的结构理由，但不保证性能。若另做THU self-future baseline，必须明确选择：当前block内PI、修改Teacher attention允许看未来、或重新训练其他建模形式；这些都是适配版，不能称为不变的原THU方法。`teacher_retain_ratio=.25`也不能脱离可见性直接沿用。

P1增加一个专门测试：修改未来block token，检查当前Teacher logits不变；修改同block同伴，检查其是否可影响当前目标。它同时验attention和PI路径，避免训练一个实际没有额外条件的Teacher。

### 1.5 LOO 到底教谁：以 A/B/C/D 为例

**位置集合G不是不动点；G上的完整token赋值才可能满足某个Teacher的坐标不动点条件。** 用A/B/C/D表示四个位置，a/b/c/d表示成功轨迹在这些位置上的最终token。这里的4只为说明，任意大于等于2的组都可定义同样的检验。

以下是本次批准的root_success_v2目标流程；当前生产joint.py仍是§1.3的旧实现，实现时须显式新增模式：

1. 当前Student从prompt采样完整轨迹，最多8次，首次答案通过verifier即停。全部失败则换下一题。失败轨迹保存用于诊断，不进入LOO分支搜索。
2. 从该成功轨迹中、依据S自身draft满足|G|>=2的状态均匀抽1个。保留真实S、已提交token/MASK、绝对位置、block边界、policy版本与RNG。抽样不依赖该state能否LOO通过。
3. 用冻结readout得到同一S上的G和自然soft drafts；令pi为这条成功轨迹最终输出在G上的完整tuple。一次性检查已提交token、EOS与长度相容性；失败则跳过，不重抽另一个state、不换另一条成功轨迹。
4. Teacher逐坐标检查，只揭开pi中的同伴；目标自己仍MASK，组外位置保持S。四位置仅为例子：

```text
检验A：Teacher(S，A=MASK，B=b，C=c，D=d) -> q_A
检验B：Teacher(S，A=a，B=MASK，C=c，D=d) -> q_B
检验C：Teacher(S，A=a，B=b，C=MASK，D=d) -> q_C
检验D：Teacher(S，A=a，B=b，C=c，D=MASK) -> q_D

接受条件：argmax(q_A)=a AND argmax(q_B)=b
          AND argmax(q_C)=c AND argmax(q_D)=d
不额外要求q_i(pi_i)>.95；不迭代修tuple，不另采候选。
```

5. 全员通过才保留全部q分布作为目标；否则跳过。Teacher仍是认证器和分布目标提供者。Student只看原S与自己的soft drafts，不接收成功轨迹的hard reference：

```text
阶段一：r_i = Readout(h_early_i(S))
        loss_readout = mean_i KL(q0_i(S) || r_i)

阶段二：m_i = sum_v r_i(v) * E[v]
        h'_i = h_i + Injector_phi(h_i, m_i)，i属于G
        p_phi = 冻结尾层(h')
        loss_joint = mean_i_in_G KL(q_i(S, reference_peers) || p_phi_i(S, natural_messages))
```

**所以，“学会ABC对D条件概率的影响”是第二阶段整个Student的研究目标；第一阶段head当前只拟合无reveal的q0。** `training.py:ReadoutTrainer.state_loss`没有LOO；`model.py:set_stage('joint')`冻结readout，只训练injector。要让第一阶段head本身接受ABC消息并拟合q_D，需要修改head输入、训练样本和推理顺序，属于另一个方法，不能把当前实现描述成已经做到。

还要限定“条件”的含义：Teacher真的看到hard a/b/c；Student只看到自然r_A/r_B/r_C的embedding期望。它没有获知实际选中的reference，因此不能保证复现每一个hard tuple对应的q_D。同一S存在多个成功reference时，前向KL对随机目标的最优解倾向于其条件平均；自然消息不足以区分它们时，增加训练步数无法补回这部分信息。记录reference与自然argmax的差异；同state多reference只在另行登记的机制诊断中检查，不为此恢复主训练候选银行。

当前loss只落在G上。若D是G外的Receiver，它会受attention传播影响，但没有直接的q_D监督；不能将G外变化自动称为学到了正确协调。LOO通过也只表示Teacher的局部坐标自洽，不保证数学推理正确、不保证全局联合MAP、不保证Student实际提交的tuple同样通过、不保证后续rollout收敛。论文[G1]支持有特权信息的on-policy蒸馏框架；[S1]提醒Teacher更少MASK仍可能出错；上述特定认证与目标是本项目设计。

本次推荐保持这个两阶段分工，先做§11.3的**同状态、同G、固定D自身消息、只替换ABC消息**检验。若只有D自己的消息影响D，或injector主要依赖h而忽略m，就不应宣称学到了同伴条件效应；再考虑消息监督/结构改动，而不是先把两阶段混为一个难以归因的新目标。

### 1.6 收益归因：冻结主干不排除“找到正确参考再学会”

“没有训练模型参数”不准确：旧Joint训练了786432个injector参数，主干/readout冻结；mini也训练新injector。只要尾层对注入有非零输入梯度，`p_phi = frozen_tail(h + Injector_phi(h,m))`就依赖phi，正确参考上的梯度可以改变后续题目的输出。因此，更强的成功轨迹筛选、提高正确token置信度、局部identity修正、同伴消息利用，都可能贡献观察到的变化。

外部依据：STaR（NeurIPS2022）明确使用模型自产且验证成功的推理来再训练，支持“筛出的正确路径本身有训练价值”这一可能性；LoRA（ICLR2022）冻结原始权重仍可通过可训练附加参数适配模型。本项目是residual injector而非LoRA，也不是STaR完整算法，引用不能证明我们的具体收益来源。[A1][A2][A3]

旧Joint Teacher目标很尖锐，已有审计中通过认证位置的目标概率中位数约.9984。当q接近参考pi的one-hot时，`KL(q||p)`的参数梯度接近`-log p(pi)`的梯度。所以成功路径模仿是具体、可检验的替代解释；冻结backbone不能排除它。且当前accuracy原评分仅+3/500、统计不确定，不能先假定已有显著能力提升再追溯其原因。

首轮仍只跑一条简化主线，保存可重用的(S,G,pi,q)用于归因。若要进一步主张“LOO软分布/同伴条件有额外价值”，优先做§11.1.1的**相同数据与injector、LOO-KL对比reference-CE**，并结合固定state消息干预；它是明确的后续受控诊断，不自动增加首轮训练网格。不通过该对照，也可报告整体方法的准确率—时延变化，但不能把全部收益命名为已证明的协同推理。

## 2. 来源索引：以哪份官方证据约束配置

以下为实施时常用入口；完整URL及固定commit/revision直接保留在§18，运行时另外保存实际依赖版本和内容hash。

| ID | 官方/论文来源 | 本计划使用的证据 |
| --- | --- | --- |
| [M1] | inclusionAI/LLaDA2.0-mini，固定HF revision 模型卡 | 正式版身份、16B/1.4B、20层、默认block/温度、长输出建议 |
| [M2] | 同 revision config.json | 2048 hidden、157184 vocab、20层、256专家/top8、BF16、上下文、AutoModel映射 |
| [M3] | 同 revision modeling_llada2_moe.py | attention、generate、EOS、temperature bug、no-grad专家分支、同位置logits |
| [M4] | 同 revision model.safetensors.index.json | 精确权重bytes；不下载权重也可估算静态下限 |
| [M5] | 同 revision tokenizer_config.json / [M6] tokenizer.json | 官方chat template、thinking off、特殊token ID |
| [M7] | LLaDA2.0原论文，arXiv:2512.15745v1 | §5.2 CAP、§6.1 tau=.95/B32/T0、block/threshold消融 |
| [M8] | inclusionAI/LLaDA2.X issue #6 及回复 | 官方确认2.0零温采样问题，指向2.1修复 |
| [M9] | LLaDA2.1-mini 固定版本代码 | 只借用零温argmax修复，不更换2.0模型 |
| [D1] | dFactory 官方 mini SFT YAML | block32、same_token_labels、FSDP2、AdamW与显存技术；路径仍写preview，不能视为正式版复现清单 |
| [D2] | dFactory训练入口；[D3]模型实现 | 干净/加噪拼接及mask、同位置loss、fused MoE；仅参考训练结构 |
| [I1] | dInfer官方 mini评测脚本；[I2]策略实现；[I3]缓存实现 | 优化推理/原始概率confidence、prefix cache；脚本tau=.80与论文.95不同 |
| [C1] | mini-CAP固定模型卡；[C2]其代码 | 正确预测上的熵最小化、confidence training强相关baseline |
| [T1] | THU d-OPSD论文，当前返回v2 | 每25步评估、前501步选best、pass8、256/512、block32、half-length steps、失败模式 |
| [T2] | THU固定版本README；[T3]YAML；[T4]MATH启动；[T5]评测启动 | 作者报告NeurIPS2026接收；实际覆盖参数、TRL补丁与A100注意事项 |
| [T6] | THU固定版本trainer源码 | root重采样传入原prompt，首次成功即停；不要求同state八次分支 |
| [M10][M11][I5] | HF讨论/GitHub公开issue | 官方MATH完整复现recipe仍缺失的旁证；不是模型效果无效的证明 |
| [E1][E5] | CALM论文与作者T5X实现 | NeurIPS2022，中间层confidence与自适应计算；不指定mini层号 |
| [E2][E6] | LayerSkip论文与作者实现 | ACL2024，共享head与early-exit训练；本方法仍使用尾层 |
| [E3][E4] | Tuned Lens作者论文与代码 | 逐层词表分布与KL读出；不将其冒称mini官方配方 |
| [A1][A2] | STaR论文与作者代码，NeurIPS2022 | 成功的自产推理可用于学习；完整算法含额外rationalization，本计划不照搬 |
| [A3] | LoRA作者代码与论文，ICLR2022 | 冻结原始权重仍可训练附加参数；本项目injector不是LoRA |
| [K1] | KLASS官方README；[K2]MATH脚本；[K3]采样实现 | NeurIPS2025 Spotlight；B64、tau .6、KL .01、history2，及fallback |
| [L1] | 原始LLaDA官方EVAL；[L2]生成；[L3]MATH prompt | 不同Base/Instruct、512/B64与纯扩散设置；官方boxed prompt |
| [L4] | iLLaDA官方评测脚本；[L5]模型卡 | var=True、MATH4096、每block32、MASK/思考结束处理不可跨模型照搬 |
| [F1] | NVIDIA Fast-dLLM v1 README；[F2]生成代码 | confidence并行和cache；必须分别度量其收益与语义 |
| [G1] | GKD，ICLR2024论文 | Student分布上的训练、散度依任务而变，不支持盲改reverse KL |
| [S1] | SelFusion，ACL2026论文入口及[正文][S2] | §3.2更少MASK仍可能对错误token过度自信；Teacher目标需检查可靠性 |
| [H1] | HF Chat Templates官方文档 | 模板、generation prompt、避免重复special tokens |
| [H2] | HF Math-Verify官方仓库 | 解析/等价判分、gold-first顺序、异常与超时处理 |
| [H3] | Transformers 4.57.1 attention mask工具 | HF mini传入4D additive mask的语义 |
| [H4] | HF Trainer文档 | checkpoint/训练状态/选择机制参考，当前Joint不是HF Trainer |
| [R1] | ASHA论文，MLSys2020，arXiv:1810.05934 | 分阶段增加资源、早期排序有噪声；本计划不声称实现正式ASHA |
| [R2] | rliable官方仓库，NeurIPS2021论文 | bootstrap区间、seed波动、避免仅报单点；本任务采用按题配对统计 |
| [Q1] | TACC Lonestar6官方文档 | 本次复核的队列费率、15分钟最低计费、先小规模验证 |

公开代码不存在“所有dLLM统一的最佳配置”。例如论文 mini tau=.95，而 dInfer脚本=.80；THU全局128步，而mini每block32步；原始LLaDA全局双向，而mini分块因果。本计划的主协议选择模型论文/权重架构，并将推理优化作为独立可核验实现。

## 3. 统一协议表：哪些可以沿用，哪些不能

| 项目 | THU LLaDA MATH | 本仓库Joint-v1 | KLASS LLaDA MATH | iLLaDA官方MATH | mini主计划 |
| --- | --- | --- | --- | --- | --- |
| 权重 | LLaDA-8B-Instruct | 同左，冻结主干 | 同左 | iLLaDA-8B-Instruct | LLaDA2.0-mini正式版 |
| attention结构 | 原生全局双向 | 原生全局双向 | 原生全局双向 | 以其官方代码为准 | **block-causal，块内双向** |
| 新增长度 | 256 | 256 | 256 | 4096 | 官方MATH长度待核实；不预填2048 |
| block | 32 | 32 | 64 | 32 | **32 [O]** |
| steps含义 | 全局128 | 阈值模式不使用128 | 256用于上限/fallback计划 | 每block32 | 每block上限32，实际自适应 |
| 提交 | 固定平均2/step | p>.9，否则1 | p>.6且历史KL<.01，否则fallback | 官方变长/固定预算脚本 | p>.95，否则1 |
| 未来窗口 | 固定完整MASK | 固定完整MASK | 固定完整MASK | 逐block增长 | 当前绝对block末尾为止 |
| EOS | 原实现完整解码 | block结束检查 | 完整长度 | 任务停止词/EOS | 生成区域中首EOS左侧已全部解码后停止 |
| prompt | THU reasoning/answer | 同THU | system + final answer | 模型模板+task配置 | 官方mini模板已知；MATH题目格式/few-shot待核实 |
| scorer | THU normalize matcher | 同左，已发现局限 | KLASS matcher | lm-eval任务配置 | 官方MATH主scorer待核实；Math-Verify仅交叉审计 |
| cache | 无本计划优化 | 无 | 无额外cache | 另核对 | 正确性先no-cache，随后prefix cache双边对齐 |

官方MATH复现与研究期MATH-500选择必须分开建manifest。原始LLaDA官方31.4%是Base/其他协议；mini的93.22%是需要核对原始评测条件的公开目标，不可换题集后硬比。缺完整recipe时状态写“尚未精确复现”，禁止直接进入方法优化。

## 4. mini 模型与执行环境的精确规格

### 4.1 固定模型身份

```yaml
model_id: inclusionAI/LLaDA2.0-mini
revision: dad945cac317da394b390f82c7b40691d8a881ed
loader: AutoModelForCausalLM
trust_remote_code: true
dtype: bfloat16
quantization: none
num_hidden_layers: 20
hidden_size: 2048
vocab_size: 157184
tie_word_embeddings: false
num_experts: 256
num_experts_per_tok: 8
num_shared_experts: 1
num_attention_heads: 16
num_key_value_heads: 4
head_dim: 128
max_position_embeddings: 32768
mask_id: 156895                 # <|mask|>
eos_id: 156892                  # <|endoftext|>
pad_id: 156892
role_end_id: 156900             # <|role_end|>，不自动等同EOS
```

来源 [M2][M5][M6]。HF config并没有完整列出所有generation特殊ID，必须从tokenizer和generate联合断言；不能回退到LLaDA的126336/126081，亦不能借iLLaDA的MASK=5。PAD与EOS相同不意味着prompt padding应该触发停止。

mini backend映射：[M3]

```text
base.model.layers                    -> transformer blocks
base.model.norm                      -> final RMSNorm
base.model.word_embeddings            -> input embedding
base.lm_head                         -> output head
project(h) = lm_head(norm(h)).float()  -> norm恰好一次
block_hidden(output) = output[0]      -> 保留其它tuple成员
```

末层预测与输入MASK在**同一位置**，不做AR式右移。`AutoModelForCausalLM`只是加载类，不代表此模型是causal token-level解码。[M3][D1][D2]

### 4.2 环境

- [O] 固定HF config标注 Transformers 4.57.1；代码使用 `ALL_ATTENTION_FUNCTIONS`、`TransformersKwargs`等新接口。历史TACC 4.49.0不能被默认为兼容。[M2][M3]
- [E] 新建独立 `llada2_joint` 环境，保留现有Torch/CUDA/TACC适配经验；先以 Transformers 4.57.1 为兼容候选，安装后记录实际版本。config里的版本声明不是全依赖lock，不能保证一装就通。
- [O/A] 优先BF16 + SDPA +显式block mask；eager只作小规模数值参考。不要照抄THU YAML中的 `flash_attention_2`：普通causal FlashAttention不自动支持所需块内双向/块间因果掩码。[D1][M3][H3]
- [E] 不在旧dOPSD环境升级TRL，不引入与Joint无关的vLLM依赖；当前Joint是自有trainer。dFactory/VeOmni可作后续fused backend，不能未经转换parity就替换HF模型。
- 保留 `tacc/env.sh` 的缓存与离线约定；WORK放环境/模型缓存，SCRATCH放run。固定 model code 与 tokenizer revisions，审阅remote code后离线加载。
- 保存 `pip freeze`、Python/Torch/Transformers/FlashAttention/Triton版本、CUDA driver、GPU型号、attention backend、MoE kernel、git diff内容hash、模型文件hash；`16c07e4+dirty`式短标签不足以重建运行。

### 4.3 显存与三卡拓扑

权重索引总大小为32,511,286,784 bytes，约30.28 GiB。[M4] 1.4B active只影响部分计算，不让所有专家权重消失。

额外显存示例（只估单份tensor，不是总峰值）：

```text
full_logits_FP32(N) = N * 157184 * 4 bytes
N=2048 -> 1.199 GiB
N=4096 -> 2.398 GiB
prefix_KV_BF16(N) ≈ 20 layers * 2(K,V) * N * 4 kv_heads * 128 * 2 bytes
N=4096 -> 160 MiB
```

真实峰值还包括softmax/logsoftmax副本、autograd保存的尾层激活、MoE派发、CUDA workspace、temporary logits、readout投影。3张40GB不等于单进程有120GB。

决策顺序：

1. [E] 首选每卡一个完整冻结BF16模型，三个rank只同步小模块梯度；Teacher与Student共用冻结主干，Teacher forward关掉guidance，**不复制第二份主干**。
2. 先测单卡全流程：短prompt/长prompt、实测选定层注入后反向、最坏长度、最多8次串行root尝试及首次成功停止。保留至少约10%可用显存余量作为工程门槛；不是声称官方保证这个余量足够。
3. 若OOM，先做按当前block/目标位置投影，及时释放teacher logits、逐state backward、只保留尾层图。不能默认4bit就等价BF16。
4. 仍OOM再评估dFactory fused MoE/activation checkpoint或单节点FSDP2。FSDP使可变次数rollout成为全rank参与的collective计算，**不可把独立rank循环原封不动套FSDP**。[D1][D2][D3]
5. 量化视为另一个协议，要同步量化Teacher与base对照并重新量化parity/准确率；非默认。H100、多节点按项目约定先给预算再审批；本计划不自动申请。

## 5. 解码工程规范：必须逐项断言

### 5.1 Block grid 与 attention

令P为无padding的真实prompt token数，B=32，G为请求的最大新增长度。

```text
block_id(position) = floor(position / B)           # 全序列绝对位置
first_decode_block = floor(P / B)
physical_end = ceil((P + G) / B) * B
active_window_end = (current_block + 1) * B
allow_attention(query, key) = block_id(key) <= block_id(query)
```

[O] 这是HF mini的分块方式。[M3] 例如P=45，首个待解block是[32,64)，其中32..44是已知prompt，只能提交45..63。不是从P开始人为建立[45,77)的全新block。

4D additive mask：允许位置为0，禁止为负无穷或被选backend要求的最小值，shape `[batch,1,N,N]`。HF当前forward要求方形mask，直接传 `[B,N]` padding mask会失败；传None还可能在 `.size()`处失败。调用 helper 前后都验证语义。[M3][H3]

必须做的结构测试：同block右侧token可以影响左侧预测；未来block变化不能影响过去block；当前block能看所有过去block；MASK位置可被attention读取；PAD才屏蔽。不要用“is_causal=True”代替block-causal mask。

初期batch=1，无padding，避免把padding长度加入block编号。后续批处理要按每条样本原始位置建mask/position_ids，或按P%B和长度分桶；同一题单独/混batch输出与commit轨迹需一致。

### 5.2 Greedy、温度、confidence、MASK

主评测：

```text
raw_probs = softmax(raw_logits.float(), dim=vocab)
candidate_logits = raw_logits.float().clone()
candidate_logits[MASK] = -inf
if temperature == 0:
    token = argmax(candidate_logits)
else:
    token ~ softmax(candidate_logits / temperature)
confidence = raw_probs[token]     # 不因屏蔽MASK而重新归一化
```

- T=0必须不消耗sampling RNG，重复seed不影响结果；不同seed也应同输出（允许硬件非确定性另行排查）。[M1][M7][M9]
- 当前2.0 HF不是上述行为。此前只在CPU从官方源码AST抽出sampler，以1000份logits `[2,1,0]`、seed0、T0、top_k0/top_p1测试，产生324个非argmax样本；未加载任何模型。固定源码见[M3]，该合成检查不代替真实权重parity。
- 官方issue回复明确修在2.1；本计划只借正确的T0分支。不能用“最新版已经修了”假定2.0固定revision也修好。[M8][M9]
- [A] 训练非零温采用上述**原始模型概率**作confidence，延续现有Joint与dInfer；HF mini原生nonzero-temperature sampler用温度/截断后的概率，两者不同。manifest必须写 `confidence_space=raw_model`。[I2]
- [E] top_k=0、top_p=1、不加repetition penalty、不做CFG、不加EOS/思考结束logit boost；这些变换会改变概率阈值含义。若需要它们，另起protocol。
- [A] 禁止把MASK当新提交token，保证每步实际去掉至少一个MASK。dInfer也显式处理mask候选，但它的过滤/fallback细节与本计划不完全相同。[I2]
- softmax/KL使用FP32，主干BF16，模块参数FP32。与KLASS内部FP64的差异必须登记；tau附近的tie需统计，不能把BF16边界波动解释成机制提升。

### 5.3 阈值主协议与固定预算辅助协议

主协议标识建议 `mini20_thr095_b32_greedy_v1`：[M3][M7]

```text
eligible = 当前block内尚未提交的MASK，排除prompt与已停止row
commit = {i in eligible: confidence_i > 0.95}
若commit为空：选confidence最大的1个位置
平局：较小绝对位置优先（明确实现，不依赖不稳定topk）
一次同步写入所有commit
最多B个有效迭代；仍有MASK则报错，不静默补答案
```

在`steps_per_block=32, block=32`下，HF每步最低计划为1；因此与上面阈值+fallback结构一致，前提是已修正sampler和特殊token。**把steps改16会变成每步最低2个的混合策略，不是纯阈值实验。** [M3]

HF另有`steps=min(steps, gen_length//minimal_topk)`，所以`gen_length<32`的小smoke会改变实际schedule。主协议固定`minimal_topk=1`且cap至少32；短smoke若要验证主语义，至少给32个物理生成位置，不能以8-token smoke的NFE推算正式作业。[M3]

两种廉价辅助对照：

- `top1`: 每步只提交一个最高confidence位置，不使用阈值；质量参考，较慢。
- `fixed_k2`: 每步最多提交两个，最后不足两个全提交；控制固定并行度；边界半block需单独调度。

[A/E] fixed_k2依据THU平均2/token-step与KLASS top1，但不是把旧全局 `steps=G/2`原样映射到mini。当前Joint trainer强制threshold，第一轮仅在评测做辅助固定预算，不声称已训练适配固定预算。

首轮不做tau搜索。官方baseline和.95主方法完成后，才可另立 `{0.85,0.90,0.95}` 的后续曲线；[M7]对0.85/.95有直接消融，[F2]和现有实验支持0.90作为中间参照。只在最终少数checkpoint画曲线，不给每个checkpoint跑大网格。阈值越低base可能同样加速，因此“同tau下Joint更快”不是完整结果。

### 5.4 EOS、长度上限与返回值

推荐主语义 `prefix_complete_eos` [A]：沿用官方“EOS左侧完整才结束”的意图，修正检查时机与输出格式。[M3]

1. 仅检查生成区域 `[P, P+G)` 内**实际已提交**的EOS，忽略prompt里的EOS/PAD；候选EOS未被commit不生效。
2. 每次commit后检查最左EOS，**即使这一轮没有新提交EOS也检查**。
3. 只有EOS左侧所有生成位置已非MASK才能停止；先出现在靠右处的EOS不能让左侧未完成答案消失。
4. raw state保留完整可见窗口，completion统一是`[P, first_eos)`；另存包含EOS的token范围/结束ID。两条返回分支都不包含prompt。
5. EOS右侧允许尚有MASK；这不等于整条成功轨迹无效。有效性检查应针对到首EOS的生成前缀及G所需的参考位置，不要求其后窗口全部解码。
6. 无EOS则在cap结束，记`length_cap`；记录是否已有可评分boxed答案。`length_cap`、`unparsed_answer`、`unfinished_reasoning`是不同统计，不能混为“截断率”。
7. `<|role_end|>`不是默认EOS。记录其出现情况；若需新增停止ID，只能在analysis集验证后改变所有方法的统一协议。[M5][M6]

当前HF代码缺陷/差异：early return为`x[:,:eos_pos+1]`（含prompt），常规return为completion；内步检查仅在新commit含EOS时触发；外层即使`eos_early_stop=False`仍在block末看到EOS后break；EOS搜索的内步分支包含prompt范围。因此**不能用该boolean判断“完整长度无停止”**。[M3]

另保留`block_end_eos`对照：把当前block解完再停止。用于隔离Joint加速究竟来自更早合法EOS还是一般并行提交。最终至少在一个固定128题子集报告两种停止策略的敏感性；不是为每个checkpoint倍增评测。[E，来源M3与本仓库历史协议]

**最后一个block的长度舍入必须显式决定：**第一版采用HF原生`physical_end`，允许最多31个超出语义cap的物理位置参加最后block计算；返回答案仍最多G个token，计时/NFE包括这些计算，EOS检测不接受cap外EOS。manifest写 `last_block_policy=native_round_up`。所有base/Joint相同。若改为严格禁止cap外commit，它是另一种协议，需配对检查，不能悄悄“修正”。上下文约束使用physical_end，不只检查G。[O/A，M3]

为避免cap外填充位置贡献训练目标，通信/监督集合G另限制在语义生成区间`[P,P+max_new_tokens)`；这是末block的[A]适配（此处G指位置集合，不指前述生成长度）。区分`decode_eligible`与`guidance_eligible`，不要用同一个mask同时表示两者。若已有EOS，在其右侧不再选择新的训练参考位置；保存两种集合供审计。

### 5.5 长度先复现官方，再谈成本裁剪

不再把2048定为主MATH默认。模型卡代码示例512、Best Practices的32768与dInfer示例2048各有上下文，目前没有公开证据将其中某一个精确绑定到MATH93.22%。[M1][I1][M7] `official_math_manifest.max_new_tokens`保持未解析，完整recipe未核实前不开展“选哪档长度得分高”的实验。

baseline复现采用官方原始cap、停止与计分规则；另外记录实际输出长度/截断率、EOS/role_end、prompt+输出上下文边界。若需要为工程smoke使用短cap，另写`smoke_only=true`，不能进入主结果。模型卡长输出建议不授权超出模型有效上下文，也不意味着每题必须实际填满32768个位置。

官方baseline复现成功后，若后续训练预算需要短cap，再在独立protocol题上做长度影响诊断，明确这是研究协议的新版本。只对截断题续长也要先证明公共前缀不因末block舍入/停止逻辑改变。主比较中base、Joint、每次root rollout必须使用同一已冻结长度；不能把短cap新结果继续叫官方原始复现。当前任务没有授权以成本理由自动降低复现标准。

### 5.6 Cache：先语义正确，再验证可部署速度

[O] 原HF mini缺省use_cache=false；dInfer有prefix cache；Fast-dLLM强调cache和并行提交是两个加速来源。[M2][I1][I3][F1]

建议的两阶段交付：

- **正确性基准**：HF-style完整前缀重算，no-cache，eager/SDPA核对，base/zero-injector/group-readout parity。
- **性能实现**：缓存已经完整结束的绝对block，仅重算当前block；base与Joint使用同一实现。当前block内的MASK改变会影响该block所有位置，不能复用其旧K/V；prompt尾部若与生成共享block也不能提前缓存。

特别重要：最后一次denoise forward的输入仍有MASK且可能带guidance，**不是下一block应看到的完整clean block**。缓存seal应对最终已写入tokens重新前向，past block无eligible位置、guidance off，得到可复用K/V；不要直接存最后masked pass的cache。Student、Teacher与不同counterfactual只共享完全不受当前block影响的prefix缓存；当前blockcache必须隔离。[A：由M3的attention结构及I3机制推导]

HF mini当前mask形状检查不直接支持`q_len!=kv_len`的缓存调用；需要单独的backend/cache接口，不能仅传use_cache=true。验收包括logits、逐步commit、最终输出、cache seal NFE和时间。若不同kernel带来临界token变化，报告差异率并锁定生产backend重新建立base。

不能用慢HF base与fused/cache Joint比较，也不能以训练用no-cache而推理采用cache为由跳过on-policy轨迹一致性。训练可复用安全prefix cache，但Student/Teacher当前state的分布必须与部署规则一致。

## 6. 数据、prompt、评分：先冻结，再选择checkpoint

### 6.1 数据划分

继续使用MATH为主任务，GSM8K作为锁定模型后的第二任务；目前Joint正向信号来自MATH，不能让旧GSM历史覆盖它。

| 集合 | 建议用途 | 规模与来源 |
| --- | --- | --- |
| train_pool | readout与Joint训练题池 | `ankner/math-500` train中的原6000题manifest；当前官方数据revision见下 |
| dev | checkpoint/阈值选择 | 原500题；抽固定分层128题作`dev_fast`，其余与全500一起作最终dev |
| analysis | 协议、长度、readout机制诊断 | 原500题；协议128、readout选点64+确认64、root采样成本检查32+确认32、机制128、预留52；均不用于梯度优化 |
| test | 最终MATH结果 | `HuggingFaceH4/MATH-500`全部500题；从train来源独立加载 |
| train剩余500 | 预留，不自动当新test | 若后续需要，先声明用途再使用 |

本次查得dataset revisions：

```text
ankner/math-500:             412eb03a7193aed402c5e4009fc958c66e61b937
HuggingFaceH4/MATH-500:       6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
```

[A] THU用上述训练源与测试源，本仓库已经划分且进行过题文去重；本计划还要求固定源revision、题文hash、原始题号、题型/难度字段。现有`data.py`给test单独load_dataset时未传独立revision，需要新增`test_dataset_revision`，而非只记录训练源revision。

原test已用于方法探索，不能回头称它对“研究想法”完全未见。mini训练和参数选择必须不再读取其分数；最终明确报告“模型迁移的固定benchmark”，严谨独立结论可扩展至未用于选择的更大MATH test子集。

### 6.2 Prompt：官方模板已知，MATH完整recipe仍未查到

`tokenizer.apply_chat_template(..., add_generation_prompt=True)`及固定revision中的角色/special tokens/thinking字段以mini官方为准。[M5][H1] 但MATH题目外层的system/user指令、few-shot数量及样例、是否要求boxed，不由通用chat template自动确定。

撤回上一版将原LLaDA的`Please reason step by step...boxed`文字与mini模板拼接后设为主recipe的决定。这个组合可用于独立smoke，却不能复现另一模型官方成绩。正式prompt字段必须来自对应MATH官方recipe，缺项保持unresolved。[M10][M11]

记录每题messages、few-shot来源/顺序、rendered_text、input_ids及hash。优先模板直接tokenize；若先render再tokenize，避免重复添加BOS/EOS。[H1] 不自行加THU `<reasoning>`后缀、thinking开关、iLLaDA end_think boost。官方已经复现后，研究期若统一新prompt，应另存protocol版本和重新跑base，不能把改变后的效果全算作方法增益。

### 6.3 Scorer：官方主评分与交叉审计分开

官方复现使用原始MATH scorer的准确代码/版本、gold字段、答案抽取与聚合方式；目前这些字段没有在已核实官方材料里完整公开，不能默认把HF Math-Verify当作它的替身。[M7][M10][I5]

HF Math-Verify固定commit `ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b`作为并行交叉审计工具，gold-first调用，固定ANTLR与解析配置；另算当前THU/KLASS matcher，用逐题差异解释分数变化，不以哪个高选主成绩。[H2] 每题输出原始completion、官方parsed答案、官方判定、交叉判定和错误类型。scorer只能看completion，不能从prompt/few-shot中抽答案；需审核多个boxed、分数/小数、集合/区间、矩阵、单位、解析失败/超时等fixtures。

root成功筛选用的verifier待官方baseline冻结后与主任务评分对齐。若原始scorer不适合安全批处理，可单独规定CPU超时（原计划5秒是[E]预算，非数学定义），但新增超时/解析规则导致的判分差异必须单列。gold解析失败先修数据，不当模型错；answer-only通过不能证明整条推理或LOO tuple语义正确。

### 6.4 root_success_v2：从prompt找到成功轨迹，只抽1个state，用自身参考

本次用户已接受该主线。K=8现在只表示**每题完整root rollout的总尝试上限，含第一次**；`states_per_successful_rollout=1`，`continuations_per_state=0`，`reference_source=own_successful_trajectory`。不再存在独立candidate温度、候选排序或同state八条银行。[T3][T6支持root重试预算；其余为本项目明确的采样适配]

```text
q = 下一个尚未访问的训练题（固定shuffle，跨rank无重复）
冻结这一轮Student policy版本，全部root尝试都用该版本
for attempt in 1..8:
    Y, trace = 当前Student从原始prompt完整生成(T_train=.9)
    保存attempt、RNG、输出、verifier结果、NFE、耗时
    若答案通过verifier：break
若8条均不通过：记录root_budget_exhausted，进入下一题

eligible_states = 成功trace中，基于该state自然draft的|G|>=2的状态
若为空：记录no_eligible_state，进入下一题
t = Uniform(eligible_states)，只抽一次
S = 该去噪forward之前的真实状态；G保持当时定义
pi = Y.final_tokens[G]             # 单一成功轨迹的完整tuple
若known-token / prompt / cap / EOS相容性检查失败：跳过，不重抽
若pi任一位置LOO未通过：跳过，不重采轨迹或参考
否则：在S上自然消息注入，用Teacher full-vocab q训练injector
```

从prompt的每次尝试都独立初始化生成MASK和缓存，不继承失败轨迹已提交token；RNG按run seed、题ID、epoch、attempt确定。首次**答案**成功便停止，即使后来无组或LOO失败也不补足8次。训练T=.9是THU探索设置的迁移，全部attempt同一温度，规则最简；它与官方T0评测显式分开，见§5.2的raw confidence差异。pilot发现不可用时停止并修订版本，不私自调温度。[M1][T3][T6]

#### 6.4.1 为什么仍完成轨迹再抽state，以及具体优化的分布

这里完整rollout用于验证答案和获得同一路径的reference，不是先生成一个无关结局后又支付8条分支。回看trace只读保存状态，不重算前缀。原生单调提交decoder下成功轨迹与自身较早S相容；实现仍须逐位置断言，不依赖口头假设。若要在中间直接停止，就失去终局成功标签和这条路径的完整reference，属于另一种监督策略。

抽样明确为“成功轨迹 + 有同伴组的state + LOO通过”条件分布，不再声称所有greedy状态的均匀无偏估计。选择eligible_states只看S当时的draft/G，不预先试LOO再挑通过者。记录全部轨迹的总states/eligible states，另存`eligible_states/trace_states`；训练接受率的分母从访问题数算起，不能仅用已经抽到有效组的states来美化覆盖。readout固定bank仍保留正确/错误的greedy轨迹，与在线Joint分布分开。

单state的理由是以每条成功题轨迹作为一个监督机会，减少同题状态重复及teacher开销，把预算给尚未访问的题；没有论文证明1普遍最优，本轮不再搜索1/2/4。G人数由confidence决定，不能将“抽一个state”误作“只选一个token”。

#### 6.4.2 题目覆盖、遍历顺序与停止计数

- 数据源仍是原7500题中的既有6000题train_pool；dev500、analysis500、预留500不移入训练。增加的是6000池内实际访问与贡献监督的题数，不能称为扩大数据源。
- 初始题序由seed42固定shuffle，按全局cursor切分到rank，第一遍结束前不重访任何题。每rank每round取1题；root重试不是新题。readout用过的训练题仍可用于Joint，held-out题不能回流。
- 每题仅一个成功轨迹、至多一个进入loss的state。LOO失败不反复抽到成功，不以已知可解率重新排序题目。第二遍只在第一遍完成、dev仍支持续训且重新给出预算后开启；policy变化后需重采，不能回放旧bank冒充on-policy。
- `question_visits`、`distinct_questions_attempted`、`root_attempts`、`questions_with_correct_rollout`、`questions_with_eligible_state`、`states_drawn`、`invalid_reference_states`、`loo_rejected_states`、`accepted_states`、`distinct_questions_contributing`、`supervised_positions`、实际GPU秒/SU全部累计并随checkpoint保存。
- 第一遍每题至多一个loss，因此`accepted_states == distinct_questions_contributing`（在本遍内计；数据重复或重用须报错）。三rank每有效update接受1–3states，100updates为100–300state，不能声称等同旧525state。`question_visits = 3 * attempted_rounds`在三rank均有题的完整round成立；空轮不更新参数，但照常消耗题和时间。
- 首轮分段50→100→200→400有效updates，上限同时受已批准SU、6000题第一遍和数值/数据质量闸门限制，任一先到即停并评审。400updates即使没有空轮也需至少约1200题次；有空轮更多，贡献监督题数仍只在400–1200之间。若先耗尽6000题仍未到400updates，报告低yield，不能偷偷循环容易题补足步数。
- 闸门通过后，省下的预算优先继续这条链覆盖未访问题，再考虑增加seed或重访。若相同SU下贡献题数不升，或dev退化，不能因“rollout条数更少”就宣称性价比改善。

#### 6.4.3 如何估计能多训多少题

旧运行每题成本包含自然rollout和4×8条剩余长度不同的续写；新运行是1–8条完整root rollout，不能直接承诺4倍、8倍或15.77倍加速。旧轨迹的短续写可能比完整root便宜，成功率/长度也随policy改变。

若某题每次root成功概率固定为p且近似独立，最多8次时：

```text
P(root成功) = 1 - (1-p)^8
E(root attempts) = sum(a=0..7, (1-p)^a)
                = [1-(1-p)^8]/p       # p=0时取8
```

真实题间p不同、计算长度与成败相关，预算以pilot实测的题级总成本为准，不能拿官方MATH accuracy代入这里的训练p。保存每题完整8次上限下的实际停止位置；未生成的后续attempt不是失败样本，也不应为了画曲线补采。更大的root预算给更多题成功机会，也可能使难题消耗更多预算；不保证单位GPU时间的监督增多。[T6支持重试机制；上述概率式仅为有条件的预算说明]

使用§7.3测得的`question_seconds`和`question_yield=accepted_states/question_visits`，在§13给出相同SU下预计访问/贡献题数及保守区间；每50updates用实际账本更新。对旧LLaDA与新mini的吞吐差只作工程参考，不能当受控采样消融。

### 6.5 第一阶段专用数据交付：题级隔离、固定state、一次确认

ID按下面规则在开工时生成，不另维护文档目录的静态JSON。父划分来自`outputs/joint_math_v1/meta/split_manifest.json`（本次核对SHA256：`fee6734c24b92be7a1e990583ade93a6777afc812085a97cfbb9c61c6d86c164`），保持原6000/500/500题级隔离，不从MATH-500 test拿题。

在train、analysis各自内部，按`SHA256(UTF8('mini-readout-v1|42|' + question_id))`升序排列，然后按下表中属于该父集的顺序连续切片：train先initial256、再extension256；analysis依次protocol128、readout_val64、readout_confirm64、sampling_pilot32、sampling_confirm32、mechanism128、reserve52。本次已验证该规则能逐ID精确重建原计划的1012个选中ID。生成后的manifest保存到实际运行的`run/meta/readout_split.json`，并断言无交叉、数量正确。

原父manifest的dataset revision为null，ID来自数据行号；正式运行前仍须在固定revision核对题文hash和行号映射，不能仅凭ID相同宣布数据已完全冻结。

| 子集 | 题数 | 使用次数及用途 |
| --- | ---: | --- |
| readout_train_initial | 256 | 原train内，训练readout；每题1state |
| readout_train_extension | 256 | 仅触发扩数据规则时加入，不提前采样 |
| protocol | 128 | 原analysis内；已取得官方recipe后的parity/停止等诊断，64题检查+64题确认；不用于挑自定义recipe凑分 |
| readout_val | 64 | 每个readout checkpoint重复测固定state；可用来选层/选点 |
| readout_confirm | 64 | 层位与head checkpoint选定后只打开一次；不得回扫所有点 |
| sampling_pilot | 32 | 冻结head后运行root_success_v2，估计成功/LOO贡献率、实际成本与题覆盖 |
| sampling_confirm | 32 | 确认相同root重试预算、.95 G、唯一自身reference的yield/成本，不调策略 |
| mechanism | 128 | 后续消息干预、固定probe；与readout选点隔离 |
| analysis_reserve | 52 | 预留；用途先登记再读取结果 |

这份ID分配是可复现的hash随机划分，**尚未根据题型/难度分层**。加载固定revision数据后补齐`type/level/problem_hash`并报告分布；不得把它写成已经分层。如果有缺失/重复题必须修复，需在任何模型分数产生前冻结manifest v2并说明重分配原因，不能按模型答对与否换题。已有analysis在旧LLaDA上用于研究，不宣称是未被研究者见过的全新test。

采state方案 [E，原则依据G1的Student状态分布、H4的验证选点]：

- readout阶段injector为零，Student与base相同。按已复现的greedy部署策略采自然轨迹，不用任务正确性过滤；readout/layer诊断的固定state bank与在线Joint sampler分开标记，不假定两种采样完全同分布。
- readout/profile固定bank每题从实际去噪state均匀抽1个，seed由题ID派生；这种一次性诊断采集会完成轨迹，以知道总步数。记录t/T和block内已提交比例；不继承旧readout仅保留成功轨迹的筛选。在线Joint按§6.4完整生成以验证答案并取得自身reference；训练温度与筛选不同，不复用greedy诊断bank冒充在线数据。
- val/confirm各最多64states。所有checkpoint复用完全相同的token状态、prompt、attention mask、位置和Teacher q0；不能每个checkpoint重采“更好拟合”的state。
- 评测每个state的全部当前block MASK位置（最多32）；逐state均值再逐题均值，最后对题平均。统计bootstrap以题为单位，保留同题state相关性；不能把2048个位置当2048道独立题。
- 训练池先缓存state IDs；同一题的任何状态都只能在一个题级split中。readout_val/readout_confirm/sampling_pilot/sampling_confirm均不进入Joint优化。
- 保存Teacher完整分布有成本：每套64states×32positions×157184vocab×2bytes约0.60GiB；确认/验证合计最多128states约1.20GiB。BF16只存logits时在评测中转FP32归一化，并抽样核对量化误差；严格目标可保存FP32约1.20GiB/64states或重算。不能缓存top-k后冒充full-vocab KL。早层h小得多，缓存后readout checkpoint sweep只需adapter+norm/head，避免反复跑完整20层。
- 题数/状态数/目标分布缓存是预算决策[E]；官方来源没有证明“64题足够保证90%”。交付中必须附按题CI与未覆盖的难度桶。

## 7. Readout 与 Joint 训练参数

### 7.0 通信组使用官方confidence规则：没有组长超参数

当前生产`_select_group`还是top-k，必须修改；只把config中的4换成其他数或修改旧`message_threshold`均不能满足本计划。

```text
唯一主阈值 tau = generation.commit_threshold = 0.95
eligible = 同一个真实S中的当前block未提交MASK，排除prompt/已停止row
r_i = 早层readout在同位置的完整词表分布
v_i, c_i = 已冻结官方候选选择/概率口径应用到r_i（主推理T=0）
G_high = {i in eligible : c_i > tau}
G_decision = G_high，若为空则为最高confidence的单个fallback位置

若 |G_decision| < 2：不注入、不做LOO；最终decoder照常推进
否则 G = G_high：全组同时注入，人数完全由状态决定
不截断为top4/top8；不为了凑两个而降低阈值或额外补低confidence同伴

最终p由后续层计算；最终commit同样应用已冻结官方规则和tau
```

只有当前block长度形成天然上限（mini最多32位置，首block可更少）。Teacher的tuple人数随G变化；单人组没有其他同伴，留一只退化成q0，因此不为单人组做LOO或构造没有同伴PI的训练样本。G为0/1并不阻止模型生成，官方decoder仍有top1 fallback。记录过阈集合人数和fallback决定人数两个量，不能把fallback的低置信token称为“>.95组员”。

**对齐的含义是同一阈值、比较符、候选token合法性、概率计算口径、block和fallback规则。** 早层r与末层p不同，所以G与最终commit可以不同；不得为了让集合相等而提前调用完整模型取最终commit作为推理期G，那会额外花一遍前向并改变方法。stage2状态来自root采样T=.9的成功轨迹；G始终使用readout argmax token的raw confidence和同一个.95规则，部署G与训练G同定义。最终训练decoder允许采样token并读其raw概率，评测decoder为T0；两种轨迹分布不可混称完全一致。

原HF在T0的实现bug、特殊token处理与停止契约先按§0.4冻结。若官方复现最终采用另一已验证实现，G调用同一`candidate_and_confidence`辅助函数；不能在G处softmax重新归一化、在最终commit处却用raw probability。若在T>0版本中选取sampled token，则confidence必须读取该token对应概率而非偷偷换max；该版本另标，不混入本轮greedy主协议。

LOO仍为完整tuple全员argmax认证。**不把Teacher认证也加上>.95**：用户要求的是组选取与decoder一致，Teacher在revealed状态做局部自洽检验是另外一个定义。反过来，也不能用“Teacher可以认证”来绕过推理期的.95组门槛。[本仓库joint.py；M3/M7支持最终阈值，应用到早层是本研究的适配]

首轮只记录下列指标，不用它们搜索group_size或tau_select：

- 每题/每state的`high_conf_group_size`分布0/1/2/.../B，通信覆盖、fallback比例，分母为全部记录的轨迹states；另报成功轨迹内的eligible比例和题级无组率。
- 与同state最终q0的token agreement、过阈集合交并比、confidence差异；最终commit实际落在G内/外的数量。
- 自然draft tuple与成功轨迹自身reference是两种tuple；主训练只认证后者，前者仅在机制probe测量，不为每个训练state额外加一次认证。
- `accepted_states / question_visits`、`accepted_states / states_drawn`、`accepted_positions / GPU_second`、root首次成功attempt、Teacher输入/forward次数、全员认证失败原因。
- G长度增大会提高Teacher认证成本。先使用batch1/位置chunk降低峰值显存，不能OOM后偷偷截成top4；超出预算则如实停止/优化计算。

只有官方baseline、.95组策略和主方法完成首轮后，才另立降低confidence的后续实验。若之后只降低G阈值而不降低最终commit阈值，应明确标记二者已不完全对齐；不能在首轮为了提高coverage悄悄执行。配置主版拒绝独立`group_size/group_max_size/group_select_threshold`，`group_threshold_source=generation.commit_threshold`是绑定关系，不是第二个可调数值。

部署不使用Teacher/正确答案，不根据哪个组能认证而动态选组。新增`group_protocol.json`仅记录这个固定算法及主decoder hash，不承担选组调参功能。旧top4仅出现在历史结果说明中。

### 7.1 先做全层分布诊断，层号是诊断产物而非手调超参数

上一版L16/L17来自“留4层”或“28/32按比例映射”的启发，并非论文建议；现撤回。mini有20层只限定可检查的层号，不能据此指定注入层。[M2]

文献核查与证据边界：

| 文献/作者代码 | 支持什么 | 不能推出什么 |
| --- | --- | --- |
| CALM，NeurIPS2022 [E1][E5] | 根据中间层预测confidence做经校准的自适应计算；难度/位置影响所需深度 | 它是AR模型的早退，不给mini固定L16/L17，不证明高confidence等于语义正确 |
| LayerSkip，ACL2024与官方代码 [E2][E6] | 共享词表head、早层训练与后层验证；作者明确需相应训练recipe才能可靠早退 | 不能直接复制示例exit_layer或认为普通checkpoint早层已经可用；本项目不跳过尾层 |
| Tuned Lens，作者论文与代码 [E3][E4] | 逐层词表分布轨迹、以末层分布KL训练读出；raw logit lens会受表征坐标偏移影响 | 原论文主要为AR；raw概率不尖不能证明信息不存在，也没有为mini给出层号 |

检索到的C4“early exit”主要是提前停止**去噪步**，不是在Transformer层内找readout，不能拿它替层号背书；本次没有找到可直接给出LLaDA2.0-mini“第几层分布变尖且适合消息注入”的已验证官方/顶会结论。下面采用可复用的实测流程[E/A]，不声称存在所有dLLM通用的理论最优层。

**一次全层profile**（官方baseline通过以后）：在readout_val固定64题的自然greedy轨迹各均匀抽1state，错误轨迹也保留。同一state一次前向抓取每层输出，只保留当前block MASK位置；按模型原生final norm+LM head读出r_l，L层得到q0。不得重复加norm、错位投影或采用AR的右移标签。所有层测同一批state，不给候选层分别采题。按层/词表chunk处理，避免同时驻留`[layers,positions,vocab]`。

逐层报告：归一化熵`H(r_l)/log(V)`、max probability的分位数、超过官方tau的位置比例、与q0的KL/JS、top1 agreement、threshold组人数/覆盖、与q0高置信集合的precision/recall、MASK/EOS/role_end比例。按题难度与block进度分桶。只看到熵下降或max probability变大不算可用，必须同时核对token identity和概率保真。[E1][E2][E3]

为避免“看曲线喜欢哪层就选哪层”，预先固定如下**最早稳定达到官方confidence的选择规则**，不在dev/test accuracy上扫层：

```text
F(S) = 最终原模型q0在当前block中confidence > tau的位置
对每个 i in F(S)，记录 stable_crossing(S,i)：
  第一次满足以下条件的层 l：
    对所有 k=l,...,L，r_k在i处的argmax均等于q0 argmax，
    且该token的r_k概率均 > tau
  若只有最后一层满足，crossing=L（不能当早层成功）

C_l = 在每题F(S)中，stable_crossing <= l的位置占比，再按题平均
      F为空的题单列，不以1分填入；同时另报能形成>=2人组的覆盖
只考虑 l < L，且至少有实际>=2高置信成员的通信state
l_best = C_l最大者
l_star = 在 C_l >= C_l_best - SE(C_l_best) 的层中取最早一层
SE按题bootstrap；沿用readout checkpoint的one-standard-error选择思想
```

这条规则的目的：用官方同一个tau定义“已形成可用的尖锐预测”；选择稳定高置信覆盖已接近可用早层最大值的最早层，为同伴影响保留尽可能多的后续attention层。没有新增手工entropy cutoff、固定尾层数、按模型深度比例或2/4/8组预算。统计一标准误规则是本项目预注册的选择约定[E]，不是模型官方定理；同时发布全部曲线/逐题值，让选择可复算。bootstrap样本数等属于估计精度设置，不作为提高MATH分数的调参入口。

若`C_best=0`、只在L出现稳定过阈、>=2组覆盖不足、独立确认曲线不支持这个层，状态记`no_usable_early_layer`，不能默认回落L16/L17。raw lens偏移可能是原因：可进一步用训练题的缓存hidden拟合各层轻量KL-aligned probe再重复相同诊断，这是显式额外诊断预算，不是先指定一个喜欢的层。不能据raw失败宣称模型早层没有identity，也不能把有信息等同于适合当前残差注入。

先用readout_val锁定层规则和暂定层号，不提前打开readout_confirm调层；随后head也按val选定，再一次性在confirm核查同一层和同一个head，不回扫候选。输出`layer_profile.json/csv`、confidence/entropy/KL曲线、stable-crossing直方图、`layer_decision.json`（层号、规则、数据hash、尾层数、证据与失败项）。没有val决策产物时`early_layer=null`并拒绝head训练；confirm未通过时拒绝正式Joint。换另一个dLLM时重复同一诊断，使用该模型原生projection和已验证decoder阈值；如果其官方协议本来不是置信度解码，应先登记协议差异，不能宣称.95对所有模型通用。

**随后训练这个层的readout**：目标仍为`KL(q0(S) || r(S))`，不是LOO；首轮rank64，adapter residual-up初始化零、主干冻结，Joint再冻结readout。完整vocab蒸馏，不用top-k缓存替代目标。[E3][E4；模块规模来自本仓库]

readout默认优化器lr=1e-4、AdamW(.9,.99)、wd=.1、clip=.2、warmup5updates；数值来源边界仍为§0.2的[E/A]，不是官方mini最优。首段最多200updates，保存/验证0/25/50/75/100/150/200。每rank每update一个state、全部当前block MASK位置最多32；三rank最多3states/update。现在每题只采1state，256题池约256states，必须记录训练遍历次数并防止重复拟合。下面的验证表用于选checkpoint和决定能否进入Joint，不能反过来在test上调层。

#### 7.1.1 每个checkpoint必须交付的验证表

1. `KL_all`：所有eligible位置的full-vocab `KL(q0 || r)`，先位置mean、state mean，再题mean。另报train同口径值，避免看相同训练state上的拟合假象。
2. `KL_selected`、`agreement_selected`：仅对同一官方.95 threshold G计算；`agreement_all`不能代替被选中位置的可靠性。
3. `group_coverage`和`tuple_agreement`：均须显示分母；避免仅剩极少容易state后agreement虚高。逐题/难度/正确与错误自然轨迹/去噪早中晚分桶。
4. `confidence_fidelity`：令v=argmax(r)，报告`abs(r(v)-q0(v))`均值及p90，另报signed差，检查早层过度自信。与q0 argmax的一致性可以画precision–coverage，但不能把`max(r)`视为“语义正确概率”。
5. MASK/EOS/role_end预测比例，组成员数量分布，原生q0熵；含特殊token的原始指标与普通token子集指标同时显示，不通过删特殊token美化全部指标。
6. readout-only与off的logits/greedy轨迹parity、内存峰值、readout投影/soft message/额外延迟。阶段一不改变base输出，**不能拿MATH答案准确率作为head checkpoint的主排序指标**。

#### 7.1.2 首轮可交付门槛与选点算法

下列数字均为预注册的[E]工程闸门，目的在于避免把一个明显失真的head送入昂贵Joint训练；[G1][H4][R2]支持独立验证与统计方法，不支持这些精确数值为最优。它们不是“满足便保证方法成功”的充分条件，也不是低于它们就证明软消息无用。

| 项目 | 首轮目标 | 失败处理 |
| --- | --- | --- |
| 全位置分布拟合 | 相对step0 raw lens的`KL_all`下降至少20%，且按题paired bootstrap的改善95%区间下界>0 | 先检查层位、真实更新和数据覆盖；不能只看train KL |
| 选中位置一致性 | 固定.95 G上，`agreement_selected >= .90`（点估计，同时报CI；这是保真诊断，非另一个组选取阈值） | 检查head/层诊断；不改变G规则、不降低.95凑通过 |
| 非空通信覆盖 | 在有>=2 eligible位置的固定state中，`group_coverage >= .50`；实际选中位置总数>=128 | 避免仅筛出几个极容易state；覆盖不足则如实判未通过，首轮不降低.95 |
| 概率值保真 | 选中位置的`mean abs(r(v)-q0(v)) <= .05`，并报告过度自信尾部 | 查head校准；如引入温度校准，必须作为新readout参数在val拟合并在confirm固定验证 |
| 实现正确性 | readout-only/zero-injector对off的确定性输出parity；finite logits/KL；实际开销有测量 | 先修实现，不以“需要更多训练”解释 |

如果step0本身已满足一致性、覆盖、概率保真和parity，允许直接交付step0作为无训练readout对照；此时不要求相对自身降低20%KL，也不为达到更新次数而训练。若选择训练后的head，则仍报告其相对step0改善与理由。

选checkpoint流程必须固定，不能“看到哪一个好再定指标”：

```text
固定state bank与指标 -> 保存step0 -> 训练并测25/50/75/100/150/200
在val上排除门槛失败的checkpoint
对其余checkpoint，以KL_all最小者为best
若更早checkpoint的KL_all <= best的KL_all + best的按题bootstrap标准误，选最早者
若仍并列，选推理额外耗时更小者；不按train loss或MATH test排名
锁定layer、snapshot与固定的.95 G协议 -> 只在readout_confirm评估这个选定版本
```

确认集要求一致性/覆盖/概率保真门槛继续成立，KL相对raw lens保持正向改善并报告CI。确认集不要求机械复现val的每一位小数；若点估计或区间显示明显冲突，状态写`not_confirmed`，停止长Joint，不回头用confirm挑第二个checkpoint。重新设计需登记新一轮并重新保留未用于选择的数据。

停训规则 [E]：最早到100步再看是否已连续两个验证点相对当前最好`KL_all`改善不足1%，并确认训练states至少覆盖一遍。二者都满足可停；若val退化且train继续下降，直接保留较早best并诊断过拟合。200步并非强制交付：3rank×1state×200约600state次，对256state池已是约2.34轮；若val不改善，不应继续重复拟合。同样需要按实际有效state吞吐而非卡数推测覆盖。扩到400必须有验证改善和新的预算依据。

扩训练题触发：已完整遍历初始256题的state池且train/val差距仍扩大，或少数题/难度桶主导有效样本时，优先加入manifest中的256题扩展集，保持val/confirm不变。每次改变数据或层位另记run，不将多轮选择伪装成单次预注册成功。首轮20%/90%等门槛不满足时可开展低成本机制诊断，但不据此启动数百步Joint。

阶段一交付包至少包含（所有拟合门槛均为工程验收，不是额外decode超参）：`readout snapshot`、layer/model/tokenizer/protocol hash、split/state manifest、每个checkpoint的逐题指标与汇总、选点理由、一次confirm结果、固定.95 G的可靠性/覆盖表、时间/显存报告。**readout达标仅允许进入固定组协议的采样pilot与短Joint试验；是否能改善同伴条件预测仍由§11.3验证。**

### 7.2 Joint首轮参数表

| 参数 | 主设置 | 支持范围与理由 |
| --- | --- | --- |
| backbone / Teacher | 冻结正式mini，Teacher guidance off | 现有Joint定义；T1/G1支持self-teacher框架 |
| readout / layer | 冻结新mini head；层号读layer_decision.json | §7.1；无手工默认层 |
| G | 当前block所有draft raw confidence>.95的位置，少于2不通信 | [A] 绑定官方主阈值，不设人数上限/第二个阈值 |
| ranks / scale | readout64、injector64、scale1 | [E] 延续模块规模；不是官方mini最优 |
| soft message | 全vocab期望embedding，FP32跨chunk累加 | 不换hard reference或top-k近似 |
| sampling protocol | root_success_v2 | 用户接受的简化；历史same_state_v1独立保留 |
| root policy | 当前Student；全部attempt T=.9；最多8次，含首次；首次答案成功即停 | [A] T3/T6，迁移训练探索策略；不是官方T0评测 |
| state / reference | 每条首次成功轨迹均匀抽1个至少2人组的状态；唯一pi来自自身最终token | [E] §6.4；不得按LOO结果重抽 |
| extra continuations / tuple ranking | 0 / none | 删除主线候选银行与embedding-distance搜索 |
| prompt/cap/block/stop/scorer | 使用已冻结主协议，训练温度及raw-confidence适配单列 | M3/I2，基线必须已复现 |
| root questions per rank round | 1；三rank完整round为3个不同题 | [E] 简化同步；访问同题的root attempts不算新题 |
| refresh / data order | 每round新题、新trajectory，第一遍不放回，policy版本固定至更新 | G1；失败也推进cursor，成功筛选偏差显式报告 |
| Teacher target | 自身完整tuple，全员LOO argmax认证，完整q | Joint研究定义；不声称官方规定LOO |
| loss | full-vocab forward KL，G内mean、有效state间mean | G1支持蒸馏；具体目标为本项目 |
| optimizer | AdamW(.9,.99)、wd=.1、clip=.2、injector lr=1e-4 | betas/wd/clip参考T3；lr为旧Joint起点[E] |
| schedule | constant，显式warmup5有效updates | T3支持常数；5为工程值，续跑不重算warmup |
| horizon | 50→100→200→400，且受已批SU/第一遍6000题/安全闸门约束 | R1支持分阶段，不证明400最佳；§6.4.2 |
| checkpoint / eval | 每25保存；0/25/50/100/200/400测dev_fast128 | T1/T3支持密集checkpoint；具体评测预算见§13 |
| batches | root attempts串行；LOO batch1；每次state单图 | [E] 40GB显存起点，profile后优化 |

配置只保留`training.root_rollout_temperature=.9`与`generation.temperature=0`两个用途明确的温度，不再存在state/candidate温度配对。所有子任务按显式stage设置过滤：readout不过滤答案，Joint成功过滤；旧`filter_correct_only`不得静默控制两者。切换采样协议须新run，不能把旧v1的optimizer/RNG直接当v2精确resume。

### 7.3 root采样覆盖pilot：只估成本与可行性，不搜索组规则

官方baseline、层诊断、readout确认和真实梯度smoke通过后，用analysis的`sampling_pilot32`和`sampling_confirm32`；冻结policy/零injector，执行§6.4完整协议。每题1–8条root rollout，64题共64–512条完整轨迹，最多64个抽中states、64个自身tuple；额外同状态续写为0。首次成功即停，不生成剩余attempt以补齐候选库。

必须交付题级漏斗：visited → within8 correct → has eligible state → reference compatible → LOO accepted；每级计数/分母、root attempt分布、全失败题比例、no-group/EOS无效/LOO拒绝原因、组人数、NFE、模块时间、rank失衡、accepted questions/GPU-hour。每题全部失败尝试仍计成本。bootstrap按题，32+32只适合预算估计，不支持小幅任务增益结论。

T=.9来自THU训练探索；mini卡提醒高温可能混语言/降质[M1]。报告混语言、无答案、截断与原始输出；若失败不能未经说明降低温度/阈值或回到同state搜索。先定位模型/解码/答案抽取/轨迹长度；必要的策略变化另记protocol，confirm不用于反复调参。

在首次成功即停记录上，可以报告前1/2/4/8次内找到正确答案的题数：成功后未生成的后续attempt对“截至K已成功”不影响；但不能把未生成attempt标成失败，也不能计算完整8样本银行的每候选准确率/多样性。LOO只针对首次成功轨迹的一个state，不能将root success@8等同于训练参考接受率。

以pilot与独立confirm实测预测50/100/200/400updates需要的题次、root条数、有效state和SU，同时报告同SU可覆盖的新题数及不确定性。pilot冻结policy的yield只作初估，训练每50updates滚动复核。没有足够监督或单卡放不下，就停止并报告，不自动降低.95或增加同题state数。[T6/G1/R2；样本数为E预算设置]

### 7.4 真正的训练梯度：冻结不等于no_grad

现有`GuidedModel.train()`强制`base.eval()`。mini专家的eval分支带no_grad，导致injector经过routed experts的输入梯度缺失；残差/attention/shared expert仍可能传梯度，所以“loss.backward成功、有非零grad”是伪通过。[M3]

实现可选路径：[A]

1. 仅在Student的可微尾层，将需要输入梯度的MoE路由改为相同前向值、支持autograd的expert实现；Teacher和rollout仍可走推理路径。
2. 或采用dFactory的已验证可微fused expert路径，但先解决权重布局转换、gate/共享专家和数值parity。[D3]
3. 不建议简单调用整个模型`.train()`：当前wrapper会再切回eval，且混入dropout/训练特有行为。即使config dropout=0，也要显式验证全部分支。

activation checkpointing也不能直接开启：当前框架的hook只在一次forward上下文内注册，backward重计算时hook可能已移除；此外HF的checkpoint分支受`self.training`控制，而主干保持eval。若采用重计算，优先在明确的“注入后尾层函数”上用非reentrant checkpoint，固定其输入/kwargs与注入次数，验证梯度与不开checkpoint一致。否则保持关闭。[M3][D2] 官方SFT全模型checkpoint recipe不是当前scoped hook结构的直接证明。

验收：小尺寸MoE中用输入requires_grad，分别测试routed expert项、shared expert项、residual项；对injector参数方向做FP32 directional finite difference，确认loss变化与梯度相符；排除top-k gate切换点再测。真实GPU小state验证一次参数更新后目标KL变化，主干所有parameters仍无grad/未更新。不能改变router top8或expert_bias算法来“修梯度”。

### 7.5 KL、归一化、零目标和恢复

- `KL(q||p)=sum_v exp(logq_v)*(logq_v-logp_v)`，完整vocab；q.detach，logsoftmax FP32，不把top-k trace当训练目标。
- G里各位置等权，state间等权；不同rank实际有效state数不同，先sum梯度再除全局有效state数一次，不能再按world_size除第二次。现有实现先每rank除global_n再SUM，语义等价，测试保留。
- 某rank无有效state必须参加所有collectives；全局无目标则不推进optimizer/scheduler，只推进attempted round与题目cursor。记录两套计数。
- warmup用绝对update count。现有实现`optimizer_step=0`时首步lr=0但Adam矩仍更新：新计划明确记录并建议改为`min((step+1)/warmup_steps,1)`；这是新的scheduler语义，原LLaDA精确resume不能无提示改写。
- 恢复需保存模块、optimizer、scheduler/warmup计数、每rank RNG、data顺序/cursor、root-attempt与state-selection RNG、全局题cursor/epoch、state/policy_version、selected layer、verifier与generation协议hash。
- `snapshot/`仅供推理，不等于完整恢复点。跨world_size恢复重新命名为新run，不声称位级精确resume。保存成功后原子发布COMPLETE标记。
- 精确resume测试必须覆盖空rank、全局空轮、checkpoint边界、恢复后下一题/下一个root attempt/抽中state与下一次更新一致。不能只比较最终loss近似值。

### 7.6 成功轨迹、自身reference和LOO的状态契约

State保存：绝对prompt长度P、当前block、visible_end、forward前input_ids、已知位置mask、eligible/G、position_ids、attention spec、cap、EOS、policy版本与采样RNG。trace重放用于读取S和复算Teacher，不再从S分支生成。

成功trajectory保存完整raw tokens与合法completion视图。用绝对坐标取`pi=Y[G]`，逐项检查prompt一致、所有S中已提交token一致、G原本均为MASK、G在当前block且都落在有效completion内。若G任一位置在有效EOS之后、最终未生成或含MASK，则整个state标为invalid_reference并跳过；不能删掉该组员后继续认证，更不能用EOS后垃圾token。EOS后其余MASK不应导致有效前缀整体失效。

抽样前eligible只取决于S的|G|>=2；相容性/LOO失败不重抽。由于当前Student在同一policy版本生成成功trace，重新读出S的G应精确一致；若不一致须诊断RNG/cache/hook或policy lag，不能重新选择更有利的一组。

留一Teacher输入只改变G中非目标位置，目标与G外位置保持S；batch正确扩展4D mask/position_ids/prefix cache。保存input diff hash，断言改动集合恰为`G\{i}`。Student forward不得读gold、pi、成功尾部hidden或含当前block PI的Teacher cache；冻结Teacher的目标detach，损失只更新injector。

## 8. 训练花费怎么降：先扩独特题，再做等价工程优化

| 优先级 | 优化 | 是否改变方法 | 验证要求 / 来源 |
| --- | --- | --- | --- |
| 已批准主设置 | root最多8次、首次成功停、1state、自身tuple | 改变历史v1的状态/参考分布 | 新protocol_id；保留救回审计，不宣称等价 |
| P0 | 首遍不放回，省下预算给未访问题 | 冻结采样规则 | 分别记访问/成功/贡献监督的独特题，避免重复容易题 |
| P0 | 仅对当前block/G所需位置投影LM head | 数学目标相同，舍入需核查 | M3同位置head；logits/commit parity |
| P0 | 每state一张尾层图，及时释放rollout logits | 不改变目标 | 当前training.py的逐state backward |
| P1 | 干净prefix cache、当前block重算、fused MoE/SDPA | 必须验证数值/路由等价 | I3/F1/D3；base/Joint共用，计seal成本 |
| P1 | LOO目标按完整输入hash缓存 | 只允许相同Teacher输入复用 | hash含model/protocol/state/G/pi；不缓存跨policy Student输出 |
| P2 | batch独立题root或LOO | 可能改变RNG/内核，需版本标记 | 先profile；按题/attempt独立RNG，无跨题串扰 |
| 后续独立实验 | 修改root上限、按可解率采题、更多states或旧分支搜索 | 改变训练分布/方法 | 不属于本轮默认；先说明成本与假设 |
| 机制对照 | 冻结bank做LOO-KL与reference-CE | offline诊断，不是在线主训练 | §11.1.1，相同bank/初始化/可训练参数，明确边界 |

不要逐步保存全位置全vocab logits。训练主档保留trace/token/state hash、选中S/G/pi、首次成功及失败attempt元数据、verifier结果、计时和必要Teacher目标；完整q可用固定Teacher重算，top-k只作诊断。失败attempt可保存压缩轨迹/元数据，必须能审计停止位置与成本。前缀cache/序列batch是工程优化，不应与采样机制变更混写成一个加速来源。

## 9. 实验阶段和清晰的停止条件

### P0：完成官方mini MATH复现，所有训练的前置闸门

按§0.4先取得并冻结完整官方recipe，完成基础backend/原生forward/decoder trace，报告原始HF与官方意图修正的差异，再运行完整官方MATH基线。status未passed时不进入readout/Joint，recipe仍缺失就保留awaiting状态；不得用自定义500题recipe凑分替代。此阶段可做CPU fake与无更新的真实加载/推理诊断。

旧LLaDA checkpoint sweep是独立历史补证，非mini先决条件；本次救回审计已完成，不再重复生成旧候选。若以后做旧25/50/75/100的dev sweep，单独预算，保留旧prompt/cap/tau，不用test挑点。

### P1：mini训练实现、消息路径、梯度和显存smoke

P0通过后，先完成CPU契约测试，再在TACC计算节点用train中的8–16题进行少量参数更新；analysis题只做无更新诊断。验证off/readout-only/zero-injector parity、正确root重试及首次成功停止、自身tuple相容、LOO无泄漏、MoE尾层梯度、2–3updates、save/resume、多卡空rank、峰值显存和实际路径日志。正式layer尚未选定时，smoke只作显式标记的探针hook单元检查，不保存为正式head/injector起点。

P1通过：finite loss/梯度、真实消息到达尾层、参数更新与梯度方向吻合、主干checksum不变、恢复下一题/attempt/state一致。失败就修实现，不启动长训“看会不会好”。[M3][D3][H3][G1]

### P2：层诊断、独立readout交付、root采样成本确认

- 固定官方复现后的研究protocol；dev500生成一次base，固定分层dev_fast128。不能先看dev成绩再补prompt/cap。
- §7.1全层profile与数据驱动层选择；训练256题readout，固定64题选点、独立64题一次确认。layer/head joint lock后才打开confirm，保持题级隔离。
- §7.3 sampling_pilot32+sampling_confirm32使用固定.95 G、root最大8、首次成功、自身reference；评估漏斗与成本，不做group设计网格。
- 输出coverage_budget.json：首轮50/100/200/400各段预计访问与贡献题数、GPU小时/SU、误差范围、评测预留。主训练预算未可估则不长训。

本阶段只决定层/head和可行性，不在MATH-500 test挑选协议或checkpoint。

### P3：Joint首条训练曲线

仅在官方baseline、layer/readout及运行时消息路径验收通过后，首seed=42，固定主配置，只训练一条按checkpoint接续的训练链：

```text
0 -> 25 -> 50 [评审]
50 -> 100 [评审]
100 -> 200 [评审]
200 -> 400 [评审]
```

每25保存；0/25/50/100/200/400做dev_fast128。不要为每步做全test，也不要只测最终snapshot。

旧实验100updates对应525个有效state、178个贡献题、327个访问题；新方案一题至多1state，100updates贡献100–300题、至少约300个访问题（完整3rank rounds），实际由全局空轮和yield决定。400updates贡献400–1200题，能否跑到取决于SU/6000题首遍/性能闸门，不能承诺固定倍数。保存train_coverage.csv并以updates、贡献独特题、GPU小时三种横轴画dev曲线；step100、约525accepted states、旧运行同等SU三个位置都是描述性参照，不是相互等价的停止点。明显错误/NaN/协议失败立即停。

继续到下一阶段的条件 [E]：

- 当前或此前至少一个checkpoint有实际速度/准确率改进的信号，或固定probe显示可学习且收益尚在出现；
- 没有持续的high-confidence错误、G外commit翻转伴随任务退化；
- 与上一检查点相比没有被稳定支配；小样本1题波动不算淘汰依据；
- 预算按实测足够，下一作业不触发未获授权的资源边界。

停止/回滚的条件：数值或梯度错误立即停；同一固定dev128连续两点明显退化（例如下降超过5pp且paired区间支持变差），扩大到dev500确认后回滚；100之后两次扩预算仍无Pareto改善且固定probe也不改善，停止本配置。阈值5pp是排查门槛，不是容许论文精度损失。

若clip几乎常开、delta/hidden持续变大且G外漂移与损伤一起上升，最多开**一个**低lr分支5e-5，从相同合法恢复点开始，标明optimizer和scheduler处理。不要同时调group/rank/scale/温度/loss。若仅loss波动而任务指标改善，不以loss早停。

THU论文§4.5明确记录了达到峰值后发生catastrophic policy collapse；本仓库历史step448→1344也出现退化。这支持密集保存和分段验证，但不证明mini必然在某一步崩溃。[T1] 所有`high_conf_deviation`都应解释为“偏离所选参考/Teacher”，不能未经任务验证当成真实token错误率。

### P4：选点、确认、再决定是否超400

至少把dev_fast上的accuracy最好、速度最好的两个非支配点，以及latest（若不同）提升到dev500；通常2–3个。保留原base。若最佳在200附近，可评已有保存的150/175/225/250，先dev_fast再最多一个追加dev500，不新跑全训练网格。

400仍持续改善、还有未访问题且成本可接受，才提出400→800的有预算续跑；若6000题首遍先结束，先审计难度覆盖与yield，第二遍需显式新epoch和预算。两者都不是默认必须完成的实验。若停止是预算原因，结论是“预算范围内最佳”，不是“已收敛”。

mini若有稳定正向信号，固定协议后用seed43重复一条较短验证曲线至选定范围；正式稳健性结论再考虑seed44。不能把重复计时当独立训练seed。所有新增seed成本提前记账。

### P5：冻结后最终测试

首轮冻结最多两个checkpoint（accuracy-best、latency-best），tau仍为官方.95；然后MATH-500全500，GSM8K全1319作迁移检查。GSM8K也固定prompt、stop/scorer，不用它反选MATH checkpoint。

有两个候选不代表必须把全部组合都在所有任务跑一遍。优先冻结一个主checkpoint，在MATH500做base/Joint及一个dev选定的相近时延base点；GSM8K只做base与这个主checkpoint。第二checkpoint、完整test tau曲线、CAP/KLASS按§13.4的分层预算执行，避免形成无意的全网格。未做的对照必须标明缺失，不以跨协议文献数字填补。

如果MATH-500已参与太多前期研究，追加未用于选择的MATH剩余test部分作为confirmation，题目集合先固定并去重；不能只选“Joint可能赢”的题。

## 10. Checkpoint选择与样本量：能回答什么、不能回答什么

### 10.1 不用训练loss选最终模型

on-policy每步state和有效题难度都变，原日志loss不是同一验证集上的risk。增加固定probe：从analysis的mechanism子集前32题、每题1state得到最多32个base states，固定参考/Teacher目标，检查readout KL、Joint KL、q0漂移和注入幅度；与§11.3共用这批state，同时另报当前on-policy数据上的覆盖/损失。固定probe是离线诊断，不能取代任务dev。[G1][S1]

每checkpoint输出：

```text
optimizer_updates, attempted_rounds, unique_train_questions
distinct_questions_contributing, question_visits, train_pool_coverage
root_attempts, first_success_attempt_histogram, root_exhausted_questions
eligible_state_questions, invalid_reference_states, loo_rejected_states
accepted_states, supervised_positions, gpu_seconds, total_SU
dev_accuracy, base_to_joint_fixes, base_to_joint_regressions
latency_mean, latency_median, latency_p90, total_gpu_time
denoise_NFE, prefill_NFE, cache_seal_NFE, total_forward_calls
tokens_generated, useful_tokens, cap_rate, no_answer_rate, EOS_rate
outside_KL, outside_commit_flip_rate, high_conf_deviation
delta_norm / hidden_norm, grad_norm_before_clip, clip_factor
```

### 10.2 预先定义两个选择目标

- **accuracy-best**：dev500正确题数最多；差异在抽样误差范围内时优先更早、更低成本的checkpoint，同时报告并列不确定性。
- **latency-best**：dev500上实测更快、经验accuracy不低于base超过1pp的候选中选最快；若不存在，返回base。`1pp`是工程容忍线[E]，**不是已经证明的非劣界限**。

两者都必须在同一protocol下选，禁止用top1 base的accuracy与阈值Joint的速度组合出不存在的点。附完整Pareto图，包含base自己的tau曲线。若要称“在最多1pp精度损失下加速”，还需要配对区间下界高于-1pp；样本不足就只报观察值。

### 10.3 128/500题的误差量级

以单模型p=.5作最保守近似，95%区间半宽约：

| n | 单模型accuracy半宽 |
| ---: | ---: |
| 128 | 8.7pp |
| 256 | 6.1pp |
| 500 | 4.4pp |
| 1000 | 3.1pp |

比较同题必须用配对差值。定义`d_i = correct_joint_i - correct_base_i`，discordance为只有一方正确的比例r，则：

```text
delta_hat = mean(d_i)
SE(delta_hat) ≈ sqrt((r - delta_hat^2) / n)
```

原实验r=61/500=.122，delta=.006，SE约1.56pp，95%半宽约3.06pp；+0.6pp不足以证明提升，也不足以证明精度严格不降。若r约.12，希望80% power检测2pp，粗略n≈(1.96+.84)^2*.12/.02^2≈2352；mini的r需pilot估计，不能直接套旧值。

因此推荐：128只做粗筛；500用来选可接受点并报告区间；若最终差异只有1–2pp，宁可明确统计不确定，也不要反复试seed/threshold直到p<.05。需要更强结论就扩独立题数或多seed，不能把多次相同greedy计时当增加n。

### 10.4 统计实现

- 主准确率差：按题配对bootstrap（建议2000重采样，seed固定）和McNemar/exact discordant test；区间与p值都报，主要看效应大小。[A/E，R2]
- 延迟：相同题的paired bootstrap，同时报`sum(base_time)/sum(joint_time)`和逐题ratio分布；不要混用mean speedup与ratio of means。
- 同题多个timing repeats先归为一个题级统计；训练多seed先分seed展示，再使用明确层级bootstrap，不能把所有重复当独立题。
- dev用于选择会带来乐观偏差；test只执行冻结的比较。不对test做ASHA式淘汰，不反复修改scorer直到赢。

## 11. 对照实验：最低限度与可选扩展

### 11.1 资源受限也必须有的对照

| 对照 | 解决的问题 | 开销 / 来源 |
| --- | --- | --- |
| mini base，tau=.95 | 模型自身baseline | 一次dev500缓存；M7 |
| zero-injector / group-readout | hook/readout是否改结果，额外开销是多少 | 小集parity，全量只需一个关键点 |
| Joint selected，tau=.95 | 同协议变化 | 主结果 |
| base tau=.90/.85 | 只调阈值能否获得同样速度 | 无训练；M7/F2 |
| Joint selected tau=.90/.85 | accuracy—latency前沿是否整体移动 | 最终点，不每checkpoint扫 |
| 同checkpoint消息替换诊断 | 是否使用token身份条件 | analysis固定states；见下 |

### 11.1.1 若要归因：优先增加同数据reference-CE对照

这是首轮有信号后、论文要声称“LOO条件分布超过正确路径模仿”时的最小训练对照，不自动提交或并行开网格。STaR[A1][A2]提示成功路径学习是合理替代解释；本项目的具体对照由假设决定，不冒称作者规定此消融。

从主线保存的训练题bank冻结完全相同的(S,G,pi,q_LOO)，用同一readout、注入层、初始injector、数据顺序、state/位置权重、优化器和更新预算训练两个offline诊断分支：

```text
J-KL: mean_i KL(q_LOO_i || p_phi_i)
Ref-CE: mean_i -log p_phi_i(pi_i)
```

两者都能更新injector，都不能访问额外root samples；主干冻结。这样比较的是在**相同已通过LOO的参考支持集**上，条件软分布是否超出直接模仿参考token的价值。保存实际训练时间并报告，不能把CE算力差异藏掉。它不隔离LOO筛选本身，也不等于完整在线算法的消融：bank由已记录的Student生成，固定bank使两分支都成为offline。若另做在线对照，只能严格匹配题序/root上限/参数预算，不能声称分叉后state完全相同。

首轮只预留这些可重放数据和预算条目，不强制多训。若J-KL与CE近似，不代表方法没用，但“主要是正确参考学习/置信度提升”的解释仍成立；若J-KL更好，还需§11.3同伴消息干预才能支撑消息身份的作用。只有后续要研究LOO筛选本身时才增加“未筛成功参考”的新对照，当前不扩成多因素网格。

### 11.2 有初步信号后应补的强参照

**mini-CAP官方权重** [C1]：它专门对正确预测增加置信度，和当前Joint“更快过阈值”的现象直接重合。主表至少在相同prompt/cap/tau/backend上测一次。它经历额外官方训练，不是我们同预算的受控消融；若CAP与Joint比较，不得声称差值全来自训练目标。`mini-CAP+Joint`是后续新实验，不与本次迁移绑在一起。

**KLASS适配mini** [K2][K3]：用同block32、同mini状态和scorer，正确实现KL方向、history初始化/窗口、conf+KL AND、fallback。LLaDA MATH官方(.6,.01,2)只是移植起点，不声称对mini最优。先固定这组，若调参只允许独立dev上很小预先登记范围；报告调优预算。模型换成mini后标为`KLASS-adapted`，不是KLASS论文复现。

**Fast-dLLM/dInfer** [F1][I1]：至少保证base和Joint共享prefix-cache/fused条件。若仅实现no-cache，结论限定为“参考实现下加速”，不能宣传相对官方高效引擎的部署加速。Fast-dLLM v1面向旧LLaDA，其近似cache假设与mini原生block-causal不同，不直接套同名开关。

### 11.3 消息机制诊断

#### 11.3.1 本轮已核对的真实调用链

```text
model.py:_scoped_hook -> backend.blocks[early_layer - 1] 的forward hook
    _select_group 在修改hidden前计算全组的自然r_i
    modules.py:soft_message -> m_i = sum_v r_i(v) E[v]
    modules.py:MessageInjector -> delta_i = U silu(V [h_i.detach(); m_i])
    model.py:_hook -> backend.with_block_hidden(output, h + add)
后续原生block读取被替换后的hidden -> attention传播 -> final_norm -> LM head
```

`with_block_hidden`确实返回新的tensor/tuple首项，既不是只记录log，也不是只改一个未被使用的局部变量。旧LLaDA在L28后注入，L29–L32收到修改后的residual；mini在§7.1实测确定的层后传播。`h_i.detach()`只截断injector输入端的上游图，不会自动把delta和下游输出detach。base参数冻结仍可以有输入梯度；mini routed expert的no_grad是另外一个实际阻断，必须按§7.4修复。

但Injector读取`[h_i; m_i]`，它**可能主要使用h、忽略m**。参数非零、delta norm非零、训练loss下降都不能替代对消息通道的干预验证。单改D自己的m_D也不能证明ABC影响D。

#### 11.3.2 已执行的CPU打印：证实结构路径，尚未检验真实模型效用

新增 [analysis/audit_joint_message_path.py](../analysis/audit_joint_message_path.py)，可对已加载的`GuidedModel`和真实state调用`audit_message_path(...)`。CLI只创建tiny fake模型，不下载/加载预训练权重；结果在 [message_path_cpu_audit.json](../results/joint/message_path_cpu_audit.json)。运行命令：

```bash
OMP_NUM_THREADS=1 /tmp/tvenv/bin/python analysis/audit_joint_message_path.py --output results/joint/message_path_cpu_audit.json
```

这次用6层、32维CPU fake模型，在L2后注入，留4层传播；走的是生产`GuidedModel`、backend和injector。先测原始零初始化，再将injector输出矩阵设成随机非零值，仅为激活路径。固定state/G/D自身概率，仅将一个同伴消息替成另一token的one-hot。seed=0结果：

| 检查量（max absolute difference，梯度除外） | 实际值 |
| --- | ---: |
| 重复相同自然forward的logits差 | 0 |
| scale=0与off的logits差 | 0 |
| 只改同伴消息，下一层入口同伴位置hidden差 | 0.2901288271 |
| 同一次干预，下一层入口D位置hidden差 | 0 |
| D经过L3后的hidden差 | 0.0112613440 |
| D经过L4后的hidden差 | 0.0196838379 |
| D经过L5后的hidden差 | 0.0236415863 |
| D经过L6后的hidden差 | 0.0267477036 |
| D最终logits差 | 0.0772640705 |
| D固定目标logit对同伴概率向量的梯度L2 norm | 0.1567162126 |

零初始化负对照中上述消息效应和概率梯度均为0；这是预期行为。这里的0.077不是准确率、概率改善或真实权重效果；one-hot替换也是分布外的路径压力测试。还运行了现有`tests/test_selective_draft_joint.py`的7项CPU契约测试，全部通过。

另外仅在CPU读取了已有Joint snapshot的小型adapter权重，未加载主干：[injector_weights_cpu_audit.json](../results/joint/injector_weights_cpu_audit.json)。`down.weight`为64×8192，前4096列对应h，后4096列对应m；两部分L2 norm分别5.84455、5.74456，`up.weight`为4096×64、L2 norm=0.787687，均finite。这排除了“保存的injector输出矩阵仍全零/没有消息输入列”这类问题，但不能单凭权重范数推断实际贡献，真实干预仍待运行。

#### 11.3.3 真实checkpoint的验收必须拆成“到达、依赖、有益”三层

不在WSL跑真模型；在TACC计算节点对旧LLaDA已有snapshot及mini首个短训snapshot执行。使用mechanism中固定32题，每题1个自然state；每题同时保留答对/答错自然轨迹的实际标签，不先过滤。先挑8个state打印逐层路径，再对整个固定probe聚合。样本数是[E]预算选择，不构成显著性保证。

1. **到达**：每个state保存实际G、层号、hook触发次数、注入位置delta/h norm、下一block输入差、尾层每层D的hidden差、最终logits/commit变化。`scale=0`、readout-only、重复自然forward是数值噪声与parity对照。若只有A入口改变、D入口不变、后续D改变，才定位到尾层跨位置传播。不要把一次未改变argmax当作完全没效应。
2. **依赖消息身份**：固定D自己的h/m、G、所有其他输入，分别只替换A、只替换B、只替换C、同时替换ABC。优先用同位置受控token替换或相近熵的概率混合；若需要同state另一个成功reference，须作为单独计费的诊断采样，主训练不会自动提供候选银行；再做零向量、组内交换、跨题消息、等范数换方向这些破坏性压力测试。区分置零m与关闭injector：前者仍可通过h产生delta。记录最终分布差、目标log-prob差、KL、候选identity/commit变化；与重复forward的误差底相比，不使用任意绝对1e-6作为所有BF16内核的通用标准。
3. **有益的同伴条件依赖**：先固定成功reference pi，对自然Student、off、peer-shuffle计算`KL(q_PI_D || p_D)`和`log p_D(pi_D)`，观察自然消息是否优于off且优于shuffle。如额外诊断获得另一同state相容pi'，只改ABC，分别重算Teacher q_D/pi与q_D/pi'，检查Student响应方向；没有第二reference就报告该子项未测，不能假装主训练保留K8银行。这个测试不能把D自身消息或目标token也悄悄改掉；hard message oracle与自然soft message结果分开。

checkpoint至少比较step0/短训后/候选best；若只有一小部分state有非零效应，报告覆盖而不是挑一个漂亮打印。注入强度大但消息干预基本无效，判为“有hidden改写，消息通道使用不足”；消息干预有效但自然消息不改善目标或任务，判为“依赖存在，收益未证实”。通过路径检查是进入训练的实现条件；声称学会有益协调还需自然消息相对off和破坏性对照的配对指标，以及后续任务dev/时延证据。

少量完整continuation做同state配对确认；greedy不需采样配对，stochastic要保存并共享随机数/说明耦合方式。掉分只能说明依赖，不自动证明依赖方式有益；联合reference、同伴PI与纯sharpness的贡献还需单独训练消融，推理干预不能完全替代。当前通用审计函数只覆盖no-cache参考路径；生产prefix-cache还必须重跑cache路径专项检查，不能拿no-cache通过证明cache正确。

### 11.4 何时引入文档中的Receiver目标

[Joint分析](../results/joint/math_v1_analysis.md)及已合并的历史讨论提出了不同细化目标；它们是建议而非当前实现。只有v1迁移正确且G外漂移成为瓶颈，才选择**一个**版本做下一实验：同参考的组外PI监督，或有依据纠正位置用qPI、其余用q0。后者对当前block全部MASK位置构造单一KL目标：仅当PI将首选改到同一参考token且提高其概率时用qPI，其余用q0；组内仍留一，组外揭开整个G。`p-q目标`只给出logit局部梯度方向，不保证共享参数更新后逐位置改善或保持速度。历史“固定8次续写后先改loss、再微调主干”的顺序不再适用，首轮仍按root_success_v2执行。

来源[S1]/[G1]支持“Teacher更少MASK不保证更正确、目标可靠性与状态分布重要”，没有为lambda=1或具体gating规则提供mini有效性保证。需同预算、同state采样、同初始化对照；不要在首个mini结果里同时换目标后把改善全归于跨模型泛化。

## 12. 延迟测量与实验可比性

### 12.1 主计时定义

[A] 主指标是batch1、单请求端到端生成时间：从已上GPU的prompt开始，到合法completion形成后CUDA同步结束；包含readout、消息计算、注入、全部去噪/prefill/cache seal以及EOS检查。tokenizer/scorer和文件写出分别计时，不放入核心GPU latency。[F1][I1]

主报告同时给模型侧生成时间和可选用户侧含tokenization/decode的时间，注明边界。trace另跑同RNG重放，不能在timed path保存CPU logits/逐步打印。warmup先做3个代表长度，计时重复3次用于最终候选；首轮筛选可1次以省预算。[E，重复只减少计时噪声]

同GPU按base/Joint交错或随机顺序执行，固定设备、kernel、dtype、cap、batch；隔离其他进程，记录GPU温度/时钟/功耗策略。原`tacc/joint.slurm`每模式一张GPU适合吞吐跑分，但卡间差异未被消除，最终小幅11%级延迟结论宜在同卡交叉确认。

不要用`gpu % NG`让三种评测在只分配1卡时同时运行。并发多卡可做accuracy吞吐，精确latency应串行独占GPU。

### 12.2 NFE和token口径

记录`denoise_calls`、`prompt_prefill_calls`、`cache_seal_calls`、`teacher_calls`、`root_rollout_calls`分开；主训练`continuation_calls=0`必须断言，额外机制诊断续写另记。训练Teacher不计推理NFE，但计训练成本。cache里的短query forward不等价完整前缀forward，NFE只是机制量，不是FLOPs/时延替代。

tokens/s同时报告有效completion token（首EOS前）与实际计算/写入token，特别说明末block舍入；time-to-first-token若未设计可安全流式输出，不伪报成AR同定义指标。答案变短、提前停止、block并行度、单步成本分别分解。

## 13. TACC预算与作业划分

### 13.1 当前可核验费率

本次公开官方文档仍给出：`gpu-a100`/`gpu-a100-dev`为3 SU/node-hour；`gpu-a100-small`为1.5；`gpu-h100`为6；至少15分钟计费。[Q1] 提交时仍需在TACC核对qlimits与实际allocation；文档没有提供当前个人余额。

```text
SU_job = nodes * max(actual_elapsed_hours, 0.25) * partition_rate
```

本计划无sbatch执行、无自动授权训练。延续用户约定：单作业估算>30SU、多节点或H100先询问；所有作业提交前报预算。拆成小作业不应被用于规避一个已知大实验的总预算沟通。

### 13.2 从题级实测成本预测覆盖和SU

pilot按题记录root每次尝试时间/NFE、首次成功attempt、trace/state抽取、LOO各位置/早拒绝、反向、同步等待、加载和eval时间。没有同state候选模块。多卡wall time取最慢rank；root retries易造成失衡，不能简单把三卡GPU秒除3当墙钟。

```text
C_q = sum(a=1..A_q, t_root(q,a)) + t_verify_and_trace(q)
      + I(has_selected_state) * t_own_reference_and_LOO(q)
      + I(accepted) * t_student_forward_backward(q)
A_q <= 8；成功即停；每题selected/accepted state均至多1
t_round_wall = max_rank(C_q) + sync_overhead
y_question = accepted_states / question_visits
e_round = 全局无目标round数 / attempted_rounds

预计rounds(U有效updates) ≈ U / (1-e_round)
预计question_visits ≈ 3 * rounds      # 3rank且每round有足够题时
预计accepted_states ≈ question_visits * y_question
预计train_wall ≈ rounds * mean(measured round_wall)
预计SU = 队列费率 * (train_wall + load + eval + checkpoint时间)/3600
```

若先给训练SU预算B，先扣独立评测/加载开销，使用实测wall_round估算可执行rounds，再算题数、root attempts与accepted states；第一遍distinct题数不能超过6000。对题级成本/yield做bootstrap，报告区间与慢rank/全失败题的保守场景。冻结pilot与训练过程不同，每50updates更新预计剩余覆盖；未生成的attempt不得填零成本后当固定8次平均。

别把旧LLaDA的5:32小时或503489次续写前向线性换算成mini训练时长。旧32条短续写与新最多8条完整root、不同模型/cap/cache不可直接比较。新协议是否让单位SU贡献更多题，必须实测；省下的预算首先用于未访问题，而不是保证跑完整6000题。

### 13.3 阶段预算与覆盖交付 [E]

| 阶段 | 预算规定 | 可交付物 |
| --- | --- | --- |
| P0官方MATH | 取得完整recipe/题数/cap后按实测预估；目前无可承诺SU数 | 官方复现报告、原始差异、accuracy/NFE/latency、frozen manifest |
| P1工程smoke | 3×A100 dev最多2h约6SU作为初始资源上限示例，提交前按当日费率复核 | parity/梯度/消息路径/显存/root采样契约 |
| P2 profile/readout/root pilot | 依据384条readout轨迹与64–512条root轨迹等逐项算，不再把旧12SU当保证 | layer/head确认、yield/成本、coverage_budget.json |
| P3至50，再至100 | 用pilot预测访问题/贡献题/实际SU；每段单独评审，超既有授权边界先报 | checkpoints、dev_fast、覆盖与实际账本 |
| 100→200→400 | 根据已发生yield和dev变化估边际价值 | 同一训练链更多独特题，随时保留较早best |
| 首遍结束或超过400 | 重新说明剩余题、重复采样收益、held-out曲线和预算 | 不默认第二遍、不固定800为目标 |
| P4/P5确认与必要对照 | 初步建议预留总预算约25%，以实际评测工作量为准 | dev500选点、锁定test、必要相近时延base；归因对照另列 |

25%是工程预留建议，不是论文常数。预算不足先减少长训/外部baseline组合，保留正确官方复现和独立选择；不通过压缩test/隐去成本制造成功。相同预算先覆盖更多训练题，不自动增加group网格、额外candidate银行或多个seed。

P2数量清单：readout初始train256+val64+confirm64，共384条greedy完整轨迹；layer profile复用其中128题的固定state bank，必要head缓存计单独前向成本。sampling64题各1–8次，共64–512条root完整轨迹、至多64次自身tuple认证。两者合计448–896条完整轨迹，**不含**protocol128、官方完整MATH、base dev500、梯度/消息probe、readout训练和checkpoint评测；各项复用必须按相同协议/输入hash证明，不能把温度不同的trace复用成同一样本。旧“64条自然轨迹+512条续写”的pilot账目删除。

旧LLaDA checkpoint sweep和reference-CE归因分支单列估价，未获任务授权不自动提交。作业结束用sacct更新results/EXPERIMENTS.md与本run账本。全局预算应包括多段作业，不能拆job规避>30SU/多节点/H100约定。

### 13.4 把评测也列成可计算的预算

以下是model-question pairs，尚未乘每题实测秒数。greedy输出在相同协议下可复用；计时确认可额外重复，不必重复做accuracy生成。

| 事项 | 规模计算 | 解释 |
| --- | ---: | --- |
| mini base dev500 | 500 | 同时提供base dev_fast128 |
| 25/50/100/200/400各dev_fast128 | 5×128=640 | 到哪一段算哪一段；不提前承诺跑完400 |
| 2–3个finalist扩到dev500 | 2–3×372=744–1116 | 复用同checkpoint已有128题，避免重复500 |
| 上述checkpoint选择合计 | 1884–2256 | 不含readout/长度审计/额外tau/机制probe |
| 最小MATH最终确认 | base500 + Joint500 + matched-latency-base500 =1500 | 第三个点若恰等于主base则去重；先在dev冻结其阈值 |
| mini-CAP相同协议 | +500 | 正向信号后优先增加的置信度训练参照 |
| KLASS-adapted相同协议 | +500 | 参量先在dev锁定，不能用test调参 |
| 第二任务GSM8K | 2×1319=2638 | 只测base和一个主Joint，避免所有checkpoint交叉 |
| 第二Joint checkpoint全MATH | +500 | 只有两个选点回答不同研究问题时才值得 |

τ=.85/.90的完整前沿先在dev_fast上比较：base和一个finalist额外4×128=512题次；这能筛出相近时延的base参照。预算允许才在test画全部曲线，不用所有threshold组合填满表。主协议test结果和dev曲线不能混画为同一个统计集。

若每题实测20s，1884–2256题次约10.47–12.53 GPU-hours；在三张独立完整模型副本上均衡分配，理想node-time约3.49–4.18h、约10.47–12.53SU，另加加载/失衡/重测。单卡small逐个跑则约15.7–18.8SU。这里三卡的低SU/GPU-hour可能更划算，但**latency测量仍需独占各卡并进行同卡交叉**。具体是否三卡并行由实际显存与队列决定。

最低可交付是正确的mini迁移 + 固定dev选择 + MATH最终配对结果；若GSM/CAP/KLASS预算不足，报告为pilot并列明待补工作。不要压缩到只测几十题，却声称完成了可发表的跨模型外部验证。[E，R1/R2/Q1提供资源与统计原则]

### 13.5 诊断失败时的动作表

| 症状 | 首先排查 | 允许的下一步；不是直接做什么 |
| --- | --- | --- |
| base远低于模型卡 | T0 bug、prompt、EOS切片、cap、block mask、scorer、模型revision | 先补齐官方MATH recipe并逐项parity；未复现不进入方法训练，不调参数凑93.22% |
| 大量`length_cap` | role_end/EOS、终止时机、是否语义答案已完成 | 先修停止契约，再长度审计；不直接缩短cap，也不以自定短cap替代官方复现 |
| injector.grad非零但学得异常 | routed experts no_grad、hook重算、norm两次、label shift | 单路径Jacobian/finite difference；不先加学习率 |
| LOO率低但root成功率高 | G含cap外/EOS后位置、坐标偏移、目标自身泄漏、自身tuple局部不相容 | 对单state逐位置打印diff；不随意放宽全员认证 |
| root重试耗尽/成本暴涨 | T过高、scorer解析、cap不足、难题集中、policy lag | 固定32题pilot排查；必要时另登记root预算/temperature版本，不自动恢复分支搜索 |
| 步数降但速度不降 | readout大vocab投影、FP32 embedding临时副本、MoE Python循环、CPU同步 | module profiler、selected-head/cache/fused；不只报NFE |
| off与readout-only不一致 | hook改写、autocast、RNG被读出消耗、不同mask/position_ids | 修parity后再训练 |
| resume轨迹变化 | world_size、样本顺序、root/state RNG、数据cursor、warmup、checkpoint完整性 | 保存并核对完整state；不叫精确恢复 |
| 多卡hang | 空rank未参与collective、FSDP前向次数不齐、rank先异常退出 | 协同错误传播/同步退出，先CPU多进程fake；不只无限加timeout |
| 更高confidence伴随修复≈损伤 | 目标近one-hot、G外无监督、训练只覆盖已可解题 | 先tau/base/CAP与同state干预，再考虑Receiver目标 |

动作表中的故障判断来自[M3][D2][G1][S2]及本仓库已有日志；需要实测确认，不以症状直接定因。

## 14. 给Claude的实施工作包与验收条件

| 工作包 | 修改/新增位置（建议名，尚不存在的不视为可执行） | 完成定义 |
| --- | --- | --- |
| W1 config/schema | `config.py`, 新`configs/selective_draft/llada2_joint_math.json` | 官方manifest闸门、新backend/stop/grid、训练root温度与评测温度、steps_per_block、绝对warmup；G阈值绑定commit，layer未解析拒绝长训 |
| W2 loading/backend | `loading.py`, `backends/llada2_mini.py` | 官方固定权重类和路径，模型shape/token断言，原生logits parity |
| W3 sampler/grid | 新`llada2_generation.py`或策略化generation | 正确T0、global block grid、growth、EOS、lastblock、batch1轨迹 |
| W4 state/replay | `trace.py`, `generation.py`, `joint.py` | root最多8且首次成功停、单state/自身tuple；动态窗口绝对坐标、已知token/EOS相容；不回落旧continuation分支 |
| W5 autograd | mini backend/审计过的MoE适配模块 | 尾层expert输入梯度正确，Teacher不保留图，冻结主干不更新 |
| W6 training/checkpoint | `training.py`, `checkpoint.py` | global有效state平均、首遍跨rank无重复、覆盖计数、warmup/空rank/完整resume；readout选点确认；采样协议hash匹配 |
| W7 data/scoring | `data.py`, 建议新`scoring.py` | test revision独立固定、官方prompt/scorer及Math-Verify交叉审计、题文去重、split身份核对 |
| W8 evaluation | `evaluation.py`, 建议`analysis/checkpoint_sweep.py`；已有`analysis/audit_joint_message_path.py` | dev_fast嵌套ID、配对区间、Pareto、真实计时、缓存key；真实snapshot的同伴消息干预 |
| W9 inference speed | mini prefix-cache/selected-head/fused接口 | base/Joint同优化；cache seal正确；速度包含所有开销 |
| W10 TACC | 新`tacc/llada2_joint.slurm` | 默认dev；环境隔离；独占计时；job预算；恢复；无隐式test sweep |

先完成P0所需W1–W4/W7/W8的无训练官方复现，再P1/P2训练契约与head，W9优化通过同语义核对后才长训。不要为了“先交代码”跳过MoE梯度和EOS契约。

新增配置样例是**目标schema，不可直接传给当前CLI**：

```yaml
protocol_id: mini20_thr095_b32_greedy_root_success_v2
official_math_reproduction:
  status: awaiting_official_math_recipe
  manifest: null                   # 取得并复现完整官方recipe后填写hash
  require_passed_before_training: true
model:
  path: inclusionAI/LLaDA2.0-mini
  revision: dad945cac317da394b390f82c7b40691d8a881ed
  backend: llada2_mini
  dtype: bfloat16
  attn_implementation: sdpa
  quantization: none
generation:
  mode: threshold
  commit_threshold: 0.95
  threshold_comparison: strict_gt
  min_commit: 1
  steps_per_block: 32
  block_length: 32
  block_origin: sequence_absolute_zero
  max_new_tokens: null              # 必须由官方MATH manifest解析，禁止默认2048
  window_mode: current_block_prefix
  last_block_policy: native_round_up
  temperature: 0.0                  # 主评测温度；训练root显式覆盖
  confidence_space: raw_model
  top_k: 0
  top_p: 1.0
  stop_mode: prefix_complete_eos
  mask_candidate_policy: forbid_without_confidence_renormalization
  cache_mode: none                  # parity后单独升级prefix并更新版本
guidance:
  early_layer: null                 # 由layer_decision.json解析，不允许默认层
  layer_decision: null
  readout_rank: 64
  injector_rank: 64
  group_policy: confidence_threshold_all
  group_threshold_source: generation.commit_threshold
  group_candidate_policy: greedy_raw_probability
  min_group_for_communication: 2     # LOO至少需要一个同伴，非人数预算
  scale: 1.0
  mode: group_soft
  readout_position_chunk: 8
  message_vocab_chunk: 16384
teacher:
  mode: joint
  reference_source: own_successful_trajectory
  acceptance: all_coordinate_argmax
  loo_batch_size: 1
training:
  sampling_protocol: root_success_v2
  root_rollout_temperature: 0.9
  max_root_attempts_per_question: 8  # 包含第一次；首次答案通过即停
  stop_on_first_correct: true
  states_per_successful_rollout: 1
  state_sampling: uniform_eligible_group_states
  reference_failure_action: skip_question
  continuations_per_state: 0
  tuple_ranking: none
  questions_per_rank_round: 1
  question_order: seeded_shuffle_without_replacement
  train_pool_size: 6000
  stop_at_end_of_first_pass: true
  refresh_updates: 1
  learning_rate_guidance: 0.0001
  adam_beta1: 0.9
  adam_beta2: 0.99
  weight_decay: 0.1
  grad_clip: 0.2
  warmup_updates: 5
  scheduler: constant_after_warmup
  max_updates: 50                   # 首段上限，后续100/200/400需通过评审
  approved_su_budget: null          # 实测成本和预算批准后填写
  save_every: 25
  seed: 42
evaluation:
  split: dev
  batch_size: 1
  warmup: 3
  repeats_screening: 1
  repeats_final_timing: 3
  primary_scorer_manifest: null     # 官方MATH scorer，不默认Math-Verify
  crosscheck_verifier: math_verify_pinned
  verifier_timeout_s: null          # 按已冻结主scorer契约配置；新增超时单列
```

schema必须拒绝root_success_v2下非零continuations_per_state、非none tuple_ranking、旧state_temperature/candidate_temperature/candidates_per_state/states_per_rollout字段；拒绝独立group_size/group_max_size/group_select_threshold。未解析official manifest/cap/layer/scorer/SU预算时只可审计，不可启动正式训练。当前CLI还不支持此schema，交给Claude实现时需迁移验证而非直接运行。

chunk8/16384是当前实现的显存分块参数[E]，须做chunk大小变化parity；参数本身不是论文贡献。无效旧字段（如message_threshold=.8、global steps=128、legacy filter_correct_only）不能悄悄留下造成误读；配置加载后输出“实际生效字段”和“被拒绝字段”，不要只dump默认dataclass。

### 14.1 必须有意义的测试

CPU fake mini测试：20层结构无需真实20层大参数，最小MoE即可；block0/1/31/32/33边界；P%B不同；EOS在prompt/候选/已提交位置；EOS左侧有MASK；填洞后无新EOS仍应停；cap外EOS；同位置logit；zero injector；routed expert input梯度；target自身不泄漏；自身reference known-token一致性；root首次成功停/8次耗尽/每次从prompt重置；成功后无组/LOO失败不得补采或重抽；每题最多1state；跨rank首遍题ID不重复；结束首遍正确停；empty rank；exact resume。

GPU真实权重测试：同state官方原生forward vs wrapper；修正过的T0 reference vs新decoder（两边使用相同明确补丁）；SDPA/eager小state；no-cache/cache；batch1；短/长长度OOM峰值；1–3次参数更新；base frozen checksum。将官方原始sampler与修正版的差异列为预期差异，不能要求两个不同算法输出parity。

协议测试要有对行为的外部断言，不只复写实现。尤其测试“未来block token改动不影响当前结果”和“MoE routed路径Jacobian存在”，能发现仅看shape无法发现的错误。

## 15. 必须产出的文件和运行记录

```text
run/
  meta/
    source_revisions.json
    environment.txt
    code_patch.diff
    resolved_config.json
    protocol.json
    official_math_manifest.json
    official_math_reproduction.json
    sampling_protocol.json
    layer_decision.json
    coverage_budget.json
    model_inspect.json
    split_manifest.json
    prompt_fixtures.json
    scorer_manifest.json
    parity_report.json
    gradient_report.json
    cost_profile.json
  readout/
    checkpoint-step-*/
    heldout_metrics.jsonl
    checkpoint_selection.json
    confirmation.json
  joint/
    checkpoint-step-000000/          # 注入前baseline模块状态
    checkpoint-step-000025/...
    log_rank*.jsonl
    rollouts/                      # root尝试和首次成功，含失败账本
    references/                    # 每题至多1个自身tuple，无candidate银行
    question_attempts.jsonl
    train_coverage.csv
  eval/
    dev_fast/<checkpoint>/<protocol>/rows.jsonl
    dev_full/<checkpoint>/<protocol>/rows.jsonl
    test_locked/<checkpoint>/<protocol>/rows.jsonl
    checkpoint_selection.json
    pareto.csv
    paired_statistics.json
  accounting/
    sacct.txt
    su_ledger.json
```

每个输出row至少含example_id/question_hash、prompt token hash、snapshot hash、protocol hash、model revision、seed、raw token IDs、stop reason、EOS位置、有效/物理长度、parsed answer、verifier状态、correct、latency repeats、NFE细分。

现有脚本用“summary.json存在就跳过评测”不够：文件必须匹配checkpoint hash、scorer、数据IDs、protocol、timing backend，且记录complete。配置改了但旧summary还在必须拒绝复用。

## 16. 最终允许写出的结论模板

只有完成对应实验才填入数值：

```text
我们在固定revision的LLaDA2.0-mini上验证Joint sibling PI，训练采样使用root_success_v2。
使用block32、greedy、阈值0.95及明确的EOS/长度/评分协议。
在独立dev上按预注册规则从[候选steps]选出[step]；
每题从prompt最多8次、首次成功停、抽1个state、自身tuple经LOO后训练injector。
训练共[更新数]次、访问[独特题数]题、其中[贡献题数]题提供监督、
[root attempts]条完整轨迹、[accepted states]个state、[GPU小时/SU]。
主干冻结，injector有更新；不将整体收益单独归因于LOO或同伴信息。
固定测试集上，accuracy差值[...，paired CI ...]，
平均生成时间变化[...，paired CI ...]，NFE与有效长度[... ]。
与base阈值曲线/mini-CAP/KLASS-adapted相比[仅陈述实际测得结果]。
```

若比较历史LLaDA-v1与mini-v2，明确模型与采样协议均有变化，不称严格单因素外部泛化证明；若只有no-cache结果，明确实现范围；若只有一个训练seed，明确seed；若CI跨0，报告观察到的速度/精度变化而不写“显著”；若预算到400结束，称“本预算与所测checkpoint中最佳”，不写“找到了全局最优/证明收敛”。

这项实验的价值是判断已有Joint机制能否在**原生block diffusion、MoE、更强基础模型、不同词表和层数**上维持或改善准确率—实际延迟权衡。收益可能比旧LLaDA小：mini已更强，CAP也已专门优化confidence；没有依据保证原来的11%延迟收益或某个准确率增幅会复现。

## 17. 核心证据的源码定位

下面行号基于已固定revision；引用正文函数名优先于未来分支行号。

| 事实 | 精确定位 |
| --- | --- |
| mini T0始终multinomial | [M3，第1207行起](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L1207)：第1213行仅处理非零温缩放，第1218行multinomial |
| 官方2.1修复零温 | [M9，第1218行](https://huggingface.co/inclusionAI/LLaDA2.1-mini/blob/20e64e2ad21644d0e5248586ed9c942cdd45de0f/modeling_llada2_moe.py#L1218)：argmax并读取原softmax概率 |
| eval专家断梯度 | [M3，第306行](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L306)：training/eval分支；第332行no_grad |
| 4D block mask输入要求 | [M3，第866行](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L866) |
| 同位置logits与loss | [M3，第1068行](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L1068)：lm_head后转FP32，labels不右移 |
| block边界/steps | [M3，第1303行](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L1303)：总长度向上舍入、prefill_blocks=P//B、per-block schedule |
| EOS早停/普通返回差异 | [M3，第1377行](https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py#L1377)，以及第1392行后 |
| dInfer原始概率confidence | [I2，第354行](https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/python/dinfer/decoding/parallel_strategy.py#L354)：候选Gumbel，confidence来自未加噪logits |
| THU选checkpoint/崩溃 | [T1] §4.1 Evaluation Details（每25步、前501步）与§4.5 Failure Modes；不能据此在我们test上选best |
| THU从prompt重试并首次成功停止 | [T6，第479行起](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/d_opsd_trainer.py#L479)：iter_num从1开始，while答案未通过，generate仍传入batch_prompt_ids；配置上限8见[T3] |
| CAP与阈值、block推荐 | [M7] §5.2、§6.1及Analysis of Inference Hyper-parameters |

## 18. 来源链接

[M1]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/README.md
[M2]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/config.json
[M3]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/modeling_llada2_moe.py
[M4]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/model.safetensors.index.json
[M5]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/tokenizer_config.json
[M6]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/blob/dad945cac317da394b390f82c7b40691d8a881ed/tokenizer.json
[M7]: https://arxiv.org/html/2512.15745v1
[M8]: https://github.com/inclusionAI/LLaDA2.X/issues/6
[M9]: https://huggingface.co/inclusionAI/LLaDA2.1-mini/blob/20e64e2ad21644d0e5248586ed9c942cdd45de0f/modeling_llada2_moe.py
[D1]: https://github.com/inclusionAI/dFactory/blob/e4a7b5f92b62cdad8b9d1240b51ac09a807bd0a4/configs/sft/llada2_mini_bd_sft.yaml
[D2]: https://github.com/inclusionAI/dFactory/blob/e4a7b5f92b62cdad8b9d1240b51ac09a807bd0a4/tasks/train_llada2_bd.py
[D3]: https://github.com/inclusionAI/dFactory/blob/e4a7b5f92b62cdad8b9d1240b51ac09a807bd0a4/models/llada2_moe/modeling_llada2_moe.py
[I1]: https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/evaluations/eval_llada_mini.sh
[I2]: https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/python/dinfer/decoding/parallel_strategy.py
[I3]: https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/python/dinfer/decoding/generate_cache.py
[C1]: https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP/blob/2deebcd70296188924fc59fb749ac02127c7b005/README.md
[C2]: https://huggingface.co/inclusionAI/LLaDA2.0-mini-CAP/blob/2deebcd70296188924fc59fb749ac02127c7b005/modeling_llada2_moe.py
[T1]: https://arxiv.org/html/2606.18195v2
[T2]: https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/README.md
[T3]: https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/opsd.yaml
[T4]: https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/run/math/opsd.sh
[T5]: https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/eval/run/math/opsd.sh
[K1]: https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/README.md
[K2]: https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/scripts/llada_math.sh
[K3]: https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/src/model/llada_klass.py
[L1]: https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/EVAL.md
[L2]: https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py
[L3]: https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/opencompass/opencompass/configs/datasets/math/math_0shot_gen_11c4b5.py
[L4]: https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/eval_illada.sh
[L5]: https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct/blob/5769f049224b2cff61106ebce0429ede5deeb434/README.md
[F1]: https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md
[F2]: https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py
[G1]: https://arxiv.org/html/2306.13649
[S1]: https://aclanthology.org/2026.acl-long.1008/
[S2]: https://aclanthology.org/2026.acl-long.1008.pdf
[H1]: https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating
[H2]: https://github.com/huggingface/Math-Verify/blob/ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b/README.md
[H3]: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_attn_mask_utils.py
[H4]: https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/main_classes/trainer.md
[R1]: https://arxiv.org/html/1810.05934
[R2]: https://github.com/google-research/rliable
[Q1]: https://docs.tacc.utexas.edu/hpc/lonestar6/#running

[M10]: https://huggingface.co/inclusionAI/LLaDA2.0-mini/discussions/3
[M11]: https://github.com/inclusionAI/LLaDA2.X/issues/4
[I4]: https://github.com/inclusionAI/dInfer/blob/1ffeb961cd258bede74fcf5ca8a416ae6d57b18f/README.md
[I5]: https://github.com/inclusionAI/dInfer/issues/29
[T6]: https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/d_opsd_trainer.py
[E1]: https://arxiv.org/html/2207.07061
[E2]: https://aclanthology.org/2024.acl-long.681/
[E3]: https://arxiv.org/html/2303.08112
[E4]: https://github.com/AlignmentResearch/tuned-lens
[E5]: https://github.com/google-research/t5x/tree/main/t5x/contrib/calm
[E6]: https://github.com/facebookresearch/LayerSkip
[A1]: https://arxiv.org/html/2203.14465
[A2]: https://github.com/ezelikman/STaR/blob/6ffd3c4743da6c7a7369834c434359d9f7e0ac38/README.md
[A3]: https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/README.md
