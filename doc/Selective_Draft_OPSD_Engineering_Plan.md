# Selective Draft OPSD：给 Claude 的工程实施计划

**日期：2026-10-14。交付物类型：代码实施计划，不是已实现功能或实验结果。**

方法依据：[Selective_Draft_OPSD.md](./Selective_Draft_OPSD.md)。请先读该文和根目录 [AGENTS.md](../AGENTS.md)，再按本计划实施。用户最新指令优先；CLAUDE.md 的 TACC 操作约定可用，旧代码地图、环境版本和 steps == tokens 要求不适用于本方案。以下新文件、接口、命令均为拟实现内容，不能当作当前已存在的功能。

目标是交付一条可以独立训练、恢复、评测和诊断的流水线：**一个中间读出适配器 → 共享冻结 LM Head → 单个最高置信 soft draft → 一次消息注入 → 原生尾层 → 最终提交**。从第一轮真实模型实验就采用阈值解码，同时实现固定预算对照；LLaDA 先打通，iLLaDA 作为必须支持的后续强模型验证。

本次代码工作按里程碑交付，不因实施计划存在就自动启动长训练或批量模型评测。实际 TACC 作业沿用项目预算约定。

## 0. 必须保持的方法语义

1. 默认读出层为完成第 28 个 Transformer block 后的 residual state，即零基 blocks[27] 的输出；不是第 28 层输入，也不是最终归一化后的隐藏态。层号必须可配置。
2. 连接顺序固定为 hidden → 轻量适配器 → 模型自身 final norm → 原 LM Head。一个早期预测分布同时用于选位置与构造消息，不加位置头和位置 loss。
3. 每个样本、每个去噪步最多一个指导位置 A。只在当前可解码 block 的 MASK 内选取，平分时选绝对位置最小者。
4. 消息门控是 max(r_A) >= 0.8；不是末层提交阈值。当前 block 少于两个 MASK 时跳过消息。
5. Student 消息使用完整 soft distribution 的期望 embedding，不默认做 argmax、采样或 top-K 截断。Teacher 的候选近似与 Student 的消息定义分开。
6. 只在 A 的中间 residual stream 添加消息，靠后续原生双向 attention 传播；不增加独立组内 attention、不做 Partial Loop。
7. A 的最终 token 可变、可不提交，不锁定硬 Anchor、不预占提交额度。固定的是本轮指导消息。
8. 阈值模式默认最终候选概率 > 0.9 时提交；为空则提交末层最高置信的一个。固定预算模式在末层选 k_t 个。两种模式都由末层决定内容与集合。
9. 原 backbone、embedding、final norm、LM Head 均冻结。阶段一只更新读出适配器；阶段二只更新消息注入模块。不默认添加 LoRA 或更新原 LM Head。
10. Teacher 与 Student 使用同一冻结 base。条件 Teacher 只将 A 从 MASK 改为候选 v，其余输入、attention mask、位置编码均不变；不揭示 Receiver、不添加未来 rollout。
11. 消息关闭时必须恢复相同基础模型的输出路径；关闭消息不等于消除了中间读出的计时开销。
12. 有效性看准确率和实测时延。可有质量收益或速度收益；若存在取舍，报告曲线。置信度上升、KL 下降、前向次数减少均不能单独代替任务结论。

## 1. 已核查的仓库状态与复用边界

| 当前文件 / 符号 | 实际行为 | 本次处理 |
|---|---|---|
| eval/generate.py::generate | 已有 threshold 分支、至少一个 token 的兜底；另有固定预算分支 | 作为参考和回归对象，新流水线共用新解码核心 |
| d-opsd/utils.py::generate | 独立的固定预算训练 sampler；返回轨迹；最后一个 block 不加入训练轨迹 | 不作为新阈值训练的 sampler |
| d-opsd/d_opsd_trainer.py::compute_loss | 强制每行 exactly 2 positions；teacher 选 top-2；按 batch_divide 切固定轨迹 | 不在该类上堆新分支，新建训练入口 |
| d-opsd/d_opsd_trainer.py::_generate_and_score_completions | pass@k 筛轨迹；以固定步数算 block；future PI；硬编码 EOS=126081 | 参考数据和日志格式，不复用 PI 构造和步数推算 |
| d-opsd/d_opsd_train.py | 4-bit + PEFT LoRA；AdapterSnapshotCallback 只保存名字含 lora_ 的参数 | 不复用该保存回调，新模块有独立 checkpoint 格式 |
| d-opsd/d_opsd_config.py、opsd.yaml | 旧 GRPO/OPSD 配置、batch_divide、JSD/top-k/clip 等 | 新建小型配置，不继承无关默认值 |
| analysis/llada_trace.py::LLaDATraceRecorder | 保存 S_t、C_t、各 MASK 的摘要；不存完整词表 | 参考旧 schema，新版扩展指导信息与变长状态 |
| analysis/pi_sibling_analysis.py | hard reveal、概率混合和对照分析 | 复用数学思路；其中 future PI、最终 rollout label 不能混入新 teacher |
| eval/gsm8k.py、d-opsd/data_utils.py | 训练/评测 prompt 与前缀处理有差异；评测 train split 默认只取前 1000 条 | 新建统一 prompt/data adapter，按显式样本 ID 选数据 |
| eval/eval.py | mask id 在 recorder 中硬编码；未显式传 attention_mask；wall_time 包含解析/记录且无明确 CUDA 同步 | 新评测入口统一 token IDs、mask、采样与计时 |
| analysis/test_llada_trace.py、test_pi_sibling.py | 无预训练模型的 CPU fake-model 回归 | 保留；新增行为测试，避免把固定两 token 的断言搬过来 |
| tacc/env.sh、setup_env.sh | WORK 缓存、offline 配置、CUDA/GLIBC/TRL 补丁 | 沿用，不升级或覆盖现有环境 |

**建议实现成根目录独立 Python package：selective_draft/**，用 python -m selective_draft.cli 运行。避免 d-opsd、eval 和根目录三个 utils.py 的同名导入问题。首版不改旧训练入口和默认脚本；后续如要接入 eval/eval.py，只做显式开关下的薄适配，不复制解码逻辑。

旧 eval 的 EOS 标志主要控制统计，没有真正终止后续 block 计算；新实现若修正停止策略，base 与 guided 必须一起采用新策略，并在旧路径回归时显式关闭新停止策略。不能把停止规则变化的收益记成指导收益。

## 2. 文件组织与交付接口

文件可以按实际代码规模合并，但职责与复用关系必须保持：

| 拟新增位置 | 职责 |
|---|---|
| selective_draft/config.py | dataclass 配置、JSON 读写、交叉字段校验 |
| selective_draft/backends/base.py | 统一模型访问、边界 hook、归一化与投影契约 |
| selective_draft/backends/llada.py、illada.py | 各模型的具体模块路径与版本检查 |
| selective_draft/modules.py | EarlyReadoutAdapter、MessageInjector |
| selective_draft/model.py | FrozenBaseWithGuidance，注册新增模块，组织各阶段 forward |
| selective_draft/generation.py | 阈值 / 固定预算共同核心；输入、提交、停止与统计 |
| selective_draft/data.py | GSM8K 样本 ID、split manifest、统一 prompt、答案解析适配 |
| selective_draft/trace.py | CPU 状态 shard、指导诊断摘要、schema 与回放 |
| selective_draft/teacher.py | 单位置 reveal、MC / top-K Teacher mixture |
| selective_draft/losses.py | 全词表 forward KL 与有效状态归一化 |
| selective_draft/training.py | 阶段一 / 二训练循环、rollout 刷新、DDP、恢复 |
| selective_draft/checkpoint.py | 模块快照和完整可恢复训练状态 |
| selective_draft/evaluation.py、diagnostics.py | 任务与 latency 评测，读出/teacher/消息干预诊断 |
| selective_draft/cli.py | inspect、collect、train-readout、diagnose、train-guidance、evaluate |
| configs/selective_draft/*.json | LLaDA 阈值首版、固定预算覆盖项、iLLaDA 配方 |
| tests/test_selective_draft_*.py | CPU 行为、梯度与数据契约测试 |
| tacc/selective_draft_smoke.sh、selective_draft.slurm | 分阶段计算节点 smoke、参数可覆盖的批作业入口 |
| results/selective_draft/README.md | 实验索引与结果口径；大产物放 outputs 或集群目录 |

新入口支持普通单进程运行；只有在 WORLD_SIZE > 1 时初始化分布式，不要求单卡也必须设置 LOCAL_RANK 或启动 NCCL。

拟定统一调用结果：

```python
@dataclass
class StepContext:
    block_start: Tensor       # [B], absolute token position, inclusive
    block_end: Tensor         # [B], exclusive
    eligible_mask: Tensor    # [B, N], current block & MASK & live row
    active_rows: Tensor      # [B], not finished
    step_id: Tensor          # [B], actual denoising step, not inferred from schedule

@dataclass
class GuidanceInfo:
    source_pos: Tensor       # [B], -1 means no eligible source
    source_confidence: Tensor
    gate_open: Tensor        # [B]
    skip_reason: list[str]   # no_receiver / below_threshold / inactive / disabled
    # Full source probabilities may exist transiently; don't serialize by default.

@dataclass
class StepOutput:
    logits: Tensor           # initially [B, N, V], raw final logits
    guidance: GuidanceInfo | None

@dataclass
class GenerationResult:
    sequences: Tensor
    completion_lengths: Tensor
    stop_reasons: list[str]
    stats: dict              # NFE, actual steps, commit counts, gate statistics
    trace_refs: list[str]    # empty in benchmark mode
```

生成代码通过 step_model(input_ids, attention_mask, context, guidance_mode) 获取 StepOutput。不要让模型 wrapper 自己猜测当前 block 或缓存上一个样本的 eligible_mask。

## 3. Backend 与中间层注入：先保证原计算路径一致

### 3.1 已做的源码核查

本计划静态核查了以下公开模型源码；未在本地加载模型或执行模型测试。实施时使用实际缓存的 pinned revision，检查是否与下表一致，不直接依赖浮动 main：

| 项目 | LLaDA-8B-Instruct | iLLaDA-8B-Instruct |
|---|---|---|
| 静态核查 revision | 08b83a6feb34df1a6011b80c3c00c7563e963b07 | 5769f049224b2cff61106ebce0429ede5deeb434 |
| HF 类 | LLaDAModelLM | ILLaDAForCausalLM |
| block 列表 | base.model.transformer.blocks | base.model.layers |
| 最终 norm | base.model.transformer.ln_f | base.model.norm |
| input embedding | base.model.transformer.wte | base.model.embed_tokens |
| LM projection | base.model.transformer.ff_out；weight_tying 时 F.linear 到 wte.weight | base.lm_head，配置 tie_word_embeddings=true |
| block 输出 | (hidden, cache) | (hidden,) 或包含 attention 的 tuple |
| 层数 / d / V | 32 / 4096 / 126464 | 32 / 4096 / 155136 |
| attention | 原生双向；不要改为 causal mask | GQA，SDPA 路径 is_causal=False |
| 特殊 token | config 中 mask=126336、eos=126081 | config 中 eos=2；mask 需查 tokenizer/官方配置，官方用法为 5 |

源码链接：[LLaDA](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/08b83a6feb34df1a6011b80c3c00c7563e963b07/modeling_llada.py)、[iLLaDA](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct/blob/5769f049224b2cff61106ebce0429ede5deeb434/modeling_illada.py)。

LLaDA 的 get_output_embeddings() 在 tied 模式可能返回 Embedding，不能把它作为 Linear 直接调用。backend.project_hidden 必须复刻 native final norm、weight tying、bias 和 scale_logits，且只归一化一次。iLLaDA 的原 forward 提供 labels 时做 shift CE，本方法绝不借用该 loss。

这些模块路径仅适用于核查版本。block_group_size != 1、未知类、norm/head 不匹配时给出明确错误，不静默猜路径。

### 3.2 首版采用 scoped forward hook

优先在原模型第 ell 个 block 的输出处安装一个作用域内 hook，保持模型原 forward、attention mask、RoPE 和尾层调用不变：

1. boundary = backend.blocks[ell - 1]。
2. 从 block 原输出中取 residual hidden；仅替换 tuple 中的 hidden，其余 cache/attention 原样保留。
3. readout-only 模式捕获 detached H^ell，但不替换原 hidden。
4. guided 模式选择 source、检查 gate、生成增量，并返回 H^ell + scatter(delta_A)。
5. hook 用 try/finally 或 context manager 移除；Teacher 前向、异常返回后均不残留。
6. 不原位覆盖 base 的输入 token 或被其他路径引用的 hidden；用可微的 out-of-place scatter/index_add 构造改变量。
7. 同一实例首版不支持并发前向；不要使用进程级全局变量存当前 A 或 gate。
8. 新模块必须在 DDP/Accelerate 包装前注册为 nn.Module 子模块，hook 不得偷偷持有未被 optimizer/DDP 管理的可训练参数。

**第一版关闭 activation checkpointing 和 torch.compile。** 它们可能在 backward 重放 hook，导致读取已清理的上下文或重复注入。若真实显存需要 checkpointing，优先对边界后的尾层做显式 non-reentrant checkpoint，并证明有/无 checkpoint 的输出与梯度一致；不能简单开启旧 whole_layer 开关。

后续需要减少无用 logits、hook 开销时，再实现精确的 split/prefix/tail backend。不能先重写整套模型 forward，再用“差不多相同”代替零注入一致性。

### 3.3 梯度与模式管理

- 基础权重全部 requires_grad_(False)，base 始终 eval；wrapper.train() 不能把冻结 base 的 dropout 打开。首版新增模块 dropout=0。
- 原生前向显式关闭 KV cache。不同去噪步的双向上下文会变化，首版不做跨步 prefix/KV 复用，避免把 AR cache 语义误用于 masked diffusion。
- 阶段一：native base 在 no_grad 下运行并捕获 H^ell、q^0；退出 no_grad 后，用 g_phi + 冻结 norm/head 计算读出 KL。**冻结 head 仍需对输入 hidden 求梯度**，该分支不能 no_grad。
- 阶段二：r_A、选择、Teacher targets 全部 detached；J_psi 及注入后的尾层必须在 grad enabled 下运行。冻结 tail 参数不等于关闭 tail 的 autograd。
- 冻结 prefix 没有需要梯度的输入和参数，通常不会保留 backward 图；无需为了省显存把整个 Student 包在 no_grad 中。
- 用 no_grad 计算会进入可训练模块的 detached features；不要未经处理把 inference_mode 创建的 tensor 交给需保存输入的 backward。
- Teacher 调用走独立 base_forward，没有 readout/inject hook。同一份冻结 base 可串行服务 Teacher/Student，节省一份 8B 权重；Teacher 前向安排在 Student 的带梯度 forward 之前。
- 训练调用必须经过包装后的 Module.forward(task=...)，不能在 DDP 外直接调用子模块方法而绕过梯度同步。

## 4. 轻量模块与显存实现

### 4.1 可直接编码的首版参数化

工程起点，不声称为调优后的最优设计：

```text
readout rank = 64
g(h) = h + U_r SiLU(V_r h)
V_r: d -> 64，随机初始化
U_r: 64 -> d，零初始化

injector rank = 64
J(h_A, m_A) = U_j SiLU(V_j concat(h_A, m_A))
V_j: 2d -> 64，随机初始化
U_j: 64 -> d，零初始化
guidance_scale = 1.0（固定；0 用于 parity）
```

不把上下投影全部初始化为零，否则可能没有有效梯度；也不同时令 output projection 和可学习乘法 gate 为零。实际门控 z_t 是离散选择，不回传梯度。阶段一训练 g，阶段二冻结 g 后训练 J。

以 d=4096 计算，两个无 bias 的 rank-64 模块合计约 131 万参数。统计和 checkpoint 应报告实际 trainable 参数数，禁止意外解冻数亿参数的词表矩阵。

### 4.2 只在所需位置做中间词表投影

- 为了选全局最高置信 source，必须比较当前 block 全部 eligible MASK 的早期分布；不能先随意选一个位置才投影。
- 按位置分块投影，例如 readout_position_chunk=8。FP32 softmax/logsumexp 后计算 max probability；跨 chunk 保持确定性平分规则。
- 推理/阶段二只保留每行获胜位置的完整 r_A 和必要摘要；不保存 [steps, sequence, vocab]。
- m_A 用完整 r_A 与 E 加权，至少做 FP32 累加；不要把整个 embedding table 每步复制成 FP32。可按词表 chunk 累加，或使用经数值验证的混合精度 matmul。
- full-soft message 与 top-K message 不是同一方法；首版实现 full-soft。Teacher 候选 top-K 不影响 Student 消息定义。
- 阶段一的读出图需保留给梯度；按位置 chunk 算 KL，明确 loss 归一化。真正为省显存逐 chunk backward 时，要配合 gradient accumulation/DDP no_sync，不能重复反传已释放的图。
- 学生和 Teacher 目标均优先用完整 Receiver 词表 KL；先减小 position/state microbatch，再考虑近似词表 loss。

首版从 bf16 冻结 base + 单卡 microbatch=1 做显存实测，不默认加载 4-bit，也不默认复制 Teacher。若确需量化，Teacher、Student、base 对照必须使用匹配精度，并额外验证输入梯度与零注入 parity；不要把量化差异归因于方法。

## 5. 一个解码核心服务训练、采集和评测

### 5.1 每步的数据顺序

```python
while current_block_has_masks:
    context = build_context(state, block_bounds, active_rows)
    step = step_model(state, attention_mask, context, guidance_mode)
    candidates = choose_tokens(step.logits, temperature, sample_rng)
    confidence = raw_probability_of_chosen_token(step.logits, candidates)
    commit = threshold_or_budget_select(confidence, context, config)
    record_state_before_write(state, commit, step.guidance)  # optional, CPU
    state = commit_tokens(state, candidates, commit)
    update_progress_and_stop_metadata()
```

同一次 step 的 logits 必须来自指导之后的尾层。选 source 的置信度始终来自早期 softmax，不使用 Gumbel 噪声；末层提交置信度来自实际候选的原分布概率。输出采样 RNG、Teacher 候选 RNG、数据采样 RNG 分开管理，开启 trace/Teacher 诊断不能改变 Student rollout。

### 5.2 阈值与固定预算契约

**阈值模式：**

- eligible = live row & 当前 block 范围 & state == mask_id。
- 提交全部 candidate probability > commit_threshold 的位置；没有则选 eligible 中最高者。
- commit_threshold 与 message_threshold 分开配置；默认分别为 0.9、0.8。
- 每个活动行有 MASK 时必须推进。检查新写 token 不是 MASK；遇到非法候选/非有限概率时记录并报错，不能进入无限循环或伪报已提交。
- 未触发 gate 不阻止正常提交；A 可以不在 commit set 中。
- 设置每 block 最大步数的 invariant guard；在合法单调提交下不应超过初始 MASK 数。触发说明代码错误，不静默改协议。
- threshold 模式 steps 不是循环上限；manifest 明确 requested fixed steps 不适用，另存实际 NFE。

**固定预算模式：**

- 在 block 开始用剩余 MASK 数和 steps_per_block 分配 k_t，保留 get_num_transfer_tokens 的整数分配语义。
- k_t 可为 0、1 或更多，不硬编码 2；k_t=0 允许调度空步，是否省略其 forward 必须在 base/guided 中一致并记录。
- 每步在末层按置信度选 k_t 个，不强制包含 A。
- 首轮回归用 256/128/32；对不能整除的配置，定义并测试分配或明确拒绝，不能静默丢失剩余 token。

首版支持 cfg_scale=0、low_confidence、单次 pass@1；CFG、pre_answer、add_ref 等组合明确拒绝，不借用旧分支产生未定义语义。随后增加温度采样，以相同随机状态回归现有生成器；使用数值稳定实现时记录它与历史采样器的差别。

### 5.3 Padding、block、EOS 与变长生成

- Attention mask 的 0 仅表示 padding，生成区域的 MASK token 仍是有效 attention 位置，不能屏蔽。
- prompt_len、block_start/end、active row 必须显式传递，不能从第一个 MASK 猜 prompt 边界，也不能用 t // 固定步数推算当前 block。
- 首轮真实模型 smoke 使用 batch=1，随后测试混合长度 prompt 与完成状态不同的 batch；不支持时显式限制 batch，不静默忽略 padding。
- position_ids/RoPE 沿用 backend 原逻辑，base 与 guided 相同。不要未经验证把 native arange 换成 padding-aware cumsum。
- 支持 window_mode=fixed 和 growing：fixed 预分配生成窗口；growing 每完成一个 block 才追加下一个。未来 MASK 可见性不同，是不同输入协议，不能混着比较。
- 新主流水线默认 stop_mode=block_end：只检查生成区；当前 block 完成后，如果生成区已有 stop token，则该行结束，否则继续下一块。不能因为 prompt/padding 含 EOS 就停止，也不能在 EOS 左侧仍有 MASK 时直接结束。
- 保存实际有效完成长度、计算到的块尾长度、stop_reason；返回的未使用容量不能被答案 parser 当作文本。
- 旧代码 parity 用 window_mode=fixed、stop_mode=none，先对齐旧完整窗口输出。实际质量/速度实验由新 baseline 和 guided 同时采用 block_end。
- iLLaDA 实验加入 growing + block_end，base/guided 同配；先验证官方 prompt/stop 语义。总长度不能超过模型位置上限。
- generation 的 bf16/fp16 autocast 必须显式配置。旧 eval 默认 CUDA autocast 可能是 fp16，与加载权重的 bf16 不同，不能混用后要求逐 token 完全一致。

## 6. 统一数据、状态记录与独立数据划分

### 6.1 数据与 prompt

首版任务为 GSM8K。统一使用一个明确的数据集 ID/revision，建议 openai/gsm8k/main；保留源 split 和 row ID，不能直接依赖 eval/gsm8k.py 的前 1000 条逻辑。

在官方 train split 上持久化训练集、开发集、独立分析集 ID manifest；官方 test 留给最终评估。模型/阈值/层位选择不能反复使用 test。历史选 checkpoint 的 300 题不作为新方法独立最终证据。

新 data adapter 统一训练、rollout、诊断、评测的 prompt builder 与答案 parser。LLaDA 可提供 legacy_boxed_reasoning 配方，明确是否追加 <reasoning>；iLLaDA 提供经官方模板核验的独立配方。保存最终 prompt_ids 和模板 hash，禁止 Teacher 增加参考答案。不同配方分别重跑 base，不直接比较历史准确率。

首版数据过滤做成显式 filter_correct_only 配置，建议保留旧方案的正确轨迹训练选项，同时始终保存错误 rollout。评测所有预定样本，不过滤答错、截断或拒答。pass@k 不作为默认数据补救；如另行启用，记录每次尝试、温度和最终筛选过程。

### 6.2 状态 schema

每个 rollout 必须包含：

```text
schema_version, run_id, model_revision, code_revision, policy_checkpoint_id
dataset_id/revision, split, example_id, attempt_id, seed
prompt_ids, prompt_attention_mask, prompt_template_hash
generation_config, sampling_dtype, window_mode, stop_mode
steps:
  actual_step_id, block_id, block_start, block_end
  pre_commit_state_ids, valid_length, attention_mask
  eligible_positions, committed_positions, committed_token_ids
  final_candidate_confidences
  source_pos, source_confidence, gate_open, skip_reason
  source_topk_ids/probs, source_topk_mass, source_entropy
  source_final_argmax, source_was_committed, source_identity_changed
completion_ids, completion_length, stop_reason, parsed_answer, is_correct
```

位置统一存全序列绝对索引；若向旧 trace 导出 completion-relative 索引，显式减去 prompt_len 并验证。growing 模式每步 N 可不同，使用逐步 tensor/list 或有 valid_length 的 padding schema，不能强行 stack 为固定长度。

默认训练缓存只保存 CPU token state 与元数据，按需重新算 hidden/distribution；完整 logits 和所有层 hidden 不落盘。诊断 trace 可保存 top-K 摘要，但它们不能直接冒充完整 r_A 或完整 KL target。指导位置没有提交时也必须记录 source；不能用 C_t 反推出 A。

trace 模式提供 state_only 与 diagnostic 两档；benchmark 使用 none，只保留低开销统计。旧 LLaDATraceRecorder 会保留 GPU tensor，不能直接作为长 rollout 的默认训练 buffer；新 recorder 在记录时转 CPU 或流式分 shard。

rollout 应包含最后 block 的合法训练状态，不沿用旧 sampler 排除最后 block 的做法。transition 回放须验证：恰好 commit set 发生变化，其他 token 不变。

## 7. Teacher mixture 与 loss 的精确定义

### 7.1 普通 Teacher

在相同 S_t 上关闭指导，计算 q_i^0。阶段一训练 KL(q_i^0 || r_i)，只在当前 block 的 MASK 位置计算，不监督 prompt 或未来 block。可均匀抽样 MASK 以控成本，记录抽样规则；监测实际 source 的读出表现。

“与末层 argmax 一致率”“与 rollout 最终 token 一致率”分别命名，它们不是 ground-truth token accuracy。GSM8K 自由推理没有天然逐 token 标准答案，不能给这些 proxy 换上“真实正确率”的名字。

### 7.2 Soft 条件 Teacher 的首版实现

Stage 2 固定早期读出。对同一个 S_t 重算出确定的 A、r_A、gate。仅在 gate_open 且存在 Receiver 时构造 Teacher 输入；Teacher base eval/no_grad、指导关闭：

```python
view = state.clone()
view[row, A] = candidate_v
# Other positions, prompt, attention mask and position IDs are unchanged.
q_v = teacher_base(view).distribution_at(receiver_positions)
```

建议同时实现两种明确命名的近似：

- **mc（首版训练默认，samples=4，候选采样有放回）：** v_j ~ r_A，q_bar = mean_j q_vj。重复候选可合并，仅跑一次 Teacher，再按频数加权。独立 Teacher RNG。
- **topk（诊断/消融）：** 取 K 个候选，权重 r_A(v)/retained_mass，计算截断重归一化 mixture。保存 retained_mass 与截断误差代理；它不是完整 mixture 的无偏估计。

每个候选的一次 Teacher forward 同时读所有 Receiver，不能对每个 Receiver 单独运行完整模型。candidate microbatch 初始为 1，增大前测显存。MC 或 top-K 的数字是工程起点，应在独立分析状态上做样本数/质量诊断。

概率先混合再算 log：q_bar = sum w_v softmax(logits_v)。不能平均 logits，不能把不同候选的 top-K Receiver 词表分别归一化后直接拼接。可用 logsumexp(log_weight + log_qv) 稳定累积。

给定同一 soft 条件与 detached 权重，平均各 hard 条件的 forward KL 与混合目标 KL 对 Student 的梯度相同，但标量相差一个与 Student 无关的项。首版直接实现 mixture KL 以便清晰记录；若将来流式算交叉熵，日志必须注明目标形式。

### 7.3 全词表 forward KL 与归一化

使用 FP32 log_softmax，在词表维求和，再按有效位置平均：

```python
per_position_kl = (q * (log_q - log_p)).sum(dim=-1)
per_state_loss = per_position_kl[receiver_mask].mean()
loss = mean_over_valid_states(per_state_loss)
```

阶段一位置集合为 M_t，阶段二为 M_t 去掉 A；不只监督最后提交的位置。Teacher probabilities 在两个阶段都 stop-gradient。**阶段一的 r_i 必须保留到 g 的梯度；只有阶段二作为指导条件的 r_A、选择与 gate 才全部 detached。**

不要复用旧 generalized_jsd_loss 的默认 top_k=20 或逐词表项 clip；它们会改变目标，历史 clipped loss 甚至可能为负。只允许浮点误差量级的轻微负 KL；明显负值视为数值/公式错误。

Receiver/状态数可变。明确默认“每状态先对 Receiver 平均，再对有效状态平均”，而不是大 block 自动获得更大权重。数据采样先按问题，再按该 rollout 的实际 step 均匀抽样，避免长轨迹天然主导训练。

### 7.4 Teacher 诊断必须先于规模化阶段二

在固定的一批分析状态上报告 gate coverage、MC/top-K mixture 相对 q^0 的 JS/KL、候选间响应差异、近似随样本数的稳定性。报告 source 的末层一致率与最终任务正确/错误分桶，保留错误 rollout。

在同 S_t、同 A 上替换候选分布，Teacher 和 Student 都使用这个新条件进行响应比较。自然推理之外的人工干预单独标注。若绝大多数目标接近 q^0，或 source 错误造成明显损害，应反馈研究判断；不要仅靠延长训练掩盖没有信号的问题。

## 8. 两阶段训练与 on-policy 刷新

### 8.1 独立训练循环，避免继承旧 GRPO 假设

建议使用现有 Accelerate + 原生 AdamW 实现小型训练循环，不继承 dOPSDTrainer。新模块参数少，首版不需要 DeepSpeed optimizer offload。先单 GPU 跑通，再用单节点 3 卡 DDP；旧 accelerate.yaml 默认 4 processes，不直接复用。

只把当前阶段参数交给 optimizer。打印并断言 trainable parameter names；记录 base 参数不变的抽样 checksum、norm/head/embedding 的 requires_grad 和 tied storage 状态。model.eval() 与 requires_grad 管理分别处理。

### 8.2 阶段一：EarlyReadout

1. 冻结 base 和 J，训练 g。
2. 收集 base 在主阈值协议上的 rollout，保存正确/错误结果及状态。此时 g 不参与实际输出决策，base 状态可缓存复用，不必每次更新 g 都重跑 rollout。
3. 对每个训练状态，一次无指导 base forward 同时提供 detached H^ell 和 q^0。g 的预测与 KL 在 grad enabled 区域计算。
4. 统一当前 block 的 MASK 位置；必要时均匀抽样位置，不只挑最终 Teacher 最自信的几个，避免其余位置置信度失控。
5. 开发集选择 early checkpoint 时同时看读出 KL、top-1 source 一致率、0.8 gate coverage；不能只把 sharpness 提升当作可靠性提升。
6. 验证无消息时训练 g 不改变完整模型的最终 logits、解码轨迹或停止行为。
7. 保存 readout snapshot，供阶段二冻结使用；J 保持零初始化。

### 8.3 阶段二：MessageInjector

1. 加载已选定 g 并冻结，只训练 J，base 保持冻结。
2. 使用当前 Student + 当前门控 + 当前阈值生成真实 rollout。按问题和实际 step 抽取训练状态；保存生成该状态的 policy_checkpoint_id。
3. 对状态重算 early source。可以先做一次无指导 base capture 获得 H^ell，再算 r_A 和 Teacher targets；首版允许这次训练预计算，避免引入不正确的 prefix cache。
4. Teacher mixtures 全部 detached 后，再做带梯度 Student forward。允许向其传入同一状态算出的不可变 GuidanceSpec（A、r_A、gate），并验证与推理内部选择相同；J 在 Student hook 中求值。
5. 训练成本因此可能包含 base 预计算 + 多个 Teacher forwards + Student forward/backward。**“一次主干前向”只指 Student 推理路径**，不能将 Teacher 计算漏出训练预算。
6. 指导开始为零也必须保留尾层梯度。不能因为 delta 当前数值为零而在训练时走断梯度的 shortcut。
7. 更新 J 后刷新 rollout。首版 refresh_updates=1：每个 optimizer update 使用更新前当前 policy 新生成的状态；同一 update 的梯度累积阶段不改变参数。
8. 若性能原因改成一个 rollout batch 支持多次 update，必须记录 policy lag，并称为有限缓存近似；设置最大 lag/缓存生命周期，不能把长时间复用旧状态称为严格 on-policy。
9. 推理、采集和训练状态重算必须调用同一个 source/gate/injection 实现，不能一处 full soft、一处 hard 或 top-K。
10. 对 gate 长期全关、正确轨迹筛选后长期为空设置诊断退出条件；输出 coverage/筛选统计。不要自动降低门控或强行指定 A 掩盖问题。

首版工程训练超参可用学习率 1e-4、weight_decay=0、grad_clip=1.0、rank=64，作为 smoke 起点；规模化训练前根据开发集与梯度尺度再决定。不要直接复用旧 LoRA 的学习率、beta/JSD clip 和 batch_divide。

### 8.4 变长数据与 DDP 空样本问题

- rollout 长度、Receiver 数、gate 开启数在不同 rank 可以不同；不能对 ragged tensor 直接 gather 或依赖相同去噪步数。
- rollout 用每 rank 本地冻结 base/当前同步模块独立完成，不在每个自适应去噪步插入需要所有 rank 同时进入的 collective。
- 各 rank 在同一个 update 边界汇合。统计有效状态数的 scalar all_reduce，不能按本地平均 loss 再直接平均而忽略样本数不同。
- 如果目标是全局有效状态平均，DDP 自带 world-size 平均时，本地 loss 使用 local_state_loss_sum * world_size / global_valid_state_count。有效状态内先对 Receiver 平均。
- 梯度累积时以整个 optimizer update 的有效状态数为分母；核对 Accelerate 是否还会自动除以 accumulation steps，避免重复归一化。测试同时覆盖不同 microbatch 切法，并要求等价于一次合并 batch。
- 某 rank 无有效状态、其他 rank 有：该 rank 仍执行同步 backward，使用连接到所有本阶段训练参数的零 loss。不要本地 continue 导致其他 rank 死锁。
- 所有 rank 都无有效状态：共同跳过 optimizer 和 scheduler，不推进 optimizer_step；AdamW 在零梯度下仍可能 weight decay，所以“执行零 loss step”不是无变化。
- 固定每个 optimizer update 的 collective 次序；样本重试在本地准备阶段有上限，全局跳过决策在边界同步。
- 最小 DDP 测试用两进程 CPU fake model 或 TACC 两 rank，覆盖“一边 gate 全关、一边开启”“全部关闭”“状态数不同”，检查梯度与单进程合并 batch 一致。

## 9. Checkpoint、恢复与结果可追溯性

区分两类保存：

**模块快照，供推理/评测：**

```text
snapshot/
  modules.safetensors       # only g_phi and J_psi, no duplicated frozen base
  config.json
  manifest.json             # base/tokenizer revisions, stage, dimensions, code hash
  metrics.json              # dev metrics used to choose this snapshot
```

**完整恢复点，供续训：**

```text
checkpoint-step-N/
  snapshot files
  optimizer.pt
  scheduler.pt              # when used
  trainer_state.json        # stage, optimizer_step, rollout_round, counters
  rng_state_rank-N.pt       # Python/NumPy/Torch CPU/CUDA + independent generators
  sampler_state_rank-N.pt   # dataset ordering/cursor
  pending_state_refs.json   # replay buffer, policy version/lag, if any
```

在 optimizer update 边界保存，避免恢复到半个累积步。临时目录写完后原子 rename；记录成功标记，再按保留策略处理旧 checkpoint。不要把原模型 state_dict 全存一遍，也不要使用旧只筛 lora_ 参数的回调。

加载时校验 base revision、tokenizer/vocab、backend、hidden dim、early layer、module ranks、tied embedding 和 precision。推理允许显式覆盖 message/commit 阈值等实验参数，但原配置和覆盖项必须同时写入新 manifest；不允许静默加载不匹配形状或漏掉 J。

阶段一 snapshot -> 阶段二是显式初始化，不是继续使用阶段一 optimizer。阶段二 full checkpoint -> 阶段二恢复则必须恢复 optimizer、scheduler、RNG 和数据游标。

验收：连续运行 n+m 个小更新，与运行 n 个更新、保存、恢复 m 个更新对比损失、参数和后续 rollout。CPU fake 测试应确定一致；真实 GPU 按明示数值容差比较，不承诺所有 kernel 跨硬件逐位复现。

## 10. 配置、命令与启动脚本

### 10.1 配置起点

实现统一 JSON + typed CLI overrides；CLI 参数优先，并在启动时输出最终解析配置。以下是首版起点，不是已验证最优超参：

| 字段 | 默认 / 要求 |
|---|---|
| model.path / backend / revision | GSAI-ML/LLaDA-8B-Instruct / llada / pinned revision |
| model.dtype / quantization | bfloat16 / none |
| data.split_manifest | 必填，固定样本 ID；训练不能默认 test |
| data.prompt_recipe | 显式配方名；训练和评测一致 |
| guidance.early_layer | 28，取 block 输出 |
| guidance.readout_rank / injector_rank | 64 / 64 |
| guidance.message_threshold / scale | 0.8 / 1.0 |
| guidance.mode | off / readout_only / single_soft |
| guidance.readout_position_chunk | 8，可覆盖 |
| generation.mode | threshold |
| generation.commit_threshold | 0.9，仅 threshold 生效 |
| generation.steps | 128，仅 fixed_budget 生效 |
| generation.max_new_tokens / block_length | 256 / 32；强模型任务可另配更长 |
| generation.temperature | 0.0 起始；随机解码单列实验 |
| generation.window_mode / stop_mode | fixed / block_end；iLLaDA 增加 growing 配方 |
| generation.cfg_scale / remasking | 0 / low_confidence |
| teacher.mode / mc_samples | mc / 4 |
| teacher.topk / candidate_batch_size | 4 / 1；topk 仅截断近似模式生效 |
| training.learning_rate_readout / guidance | 1e-4 / 1e-4，smoke 起点 |
| training.weight_decay / grad_clip | 0 / 1.0 |
| training.microbatch_states | 1 |
| training.states_per_rollout | 4，均匀抽实际 step，可覆盖 |
| training.rollouts_per_update | 1 per rank 起始 |
| training.filter_correct_only | true 作为旧筛选口径起点；所有 rollout 仍落盘 |
| training.refresh_updates | 1；阶段二默认每次更新后刷新 |
| training.activation_checkpointing | off，需独立通过梯度验收后才启用 |
| evaluation.batch_size | 1，latency 首版按单请求测 |
| trace.level | collect 时 state_only，diagnose 时 diagnostic，benchmark 时 none |

配置必须拒绝：非法层位、阈值越界、无效 block/长度、未知 backend、未支持 CFG、指导缺失 checkpoint、训练多个阶段参数同时解冻。0 阈值本身可合法使用，不能套用旧 --threshold=0 表示 fixed 的魔数；新接口用 generation.mode 区分。

### 10.2 拟实现 CLI 示例

以下命令仅供实现完成后使用。除 CPU fake tests 外，涉及真实模型的命令都在 TACC 计算节点运行。

```bash
# 纯数据划分：生成可复用样本 ID manifest，禁止默认覆盖已有文件。
python -m selective_draft.cli prepare-data \
  --config configs/selective_draft/llada_threshold.json \
  --output-dir outputs/selective_draft/data

# 打印/验证 backend、token IDs、权重冻结、模块形状与 resolved config。
python -m selective_draft.cli inspect \
  --config configs/selective_draft/llada_threshold.json

# 基础模型阈值轨迹；split 在 config 中明确指定。
python -m selective_draft.cli collect \
  --config configs/selective_draft/llada_threshold.json \
  --set guidance.mode=off \
  --output-dir outputs/selective_draft/base_states

python -m selective_draft.cli train-readout \
  --config configs/selective_draft/llada_threshold.json \
  --states-dir outputs/selective_draft/base_states \
  --max-steps 2 \
  --output-dir outputs/selective_draft/readout_smoke

python -m selective_draft.cli diagnose \
  --kind teacher \
  --config configs/selective_draft/llada_threshold.json \
  --snapshot outputs/selective_draft/readout_smoke/snapshot \
  --output-dir outputs/selective_draft/teacher_diag

python -m selective_draft.cli train-guidance \
  --config configs/selective_draft/llada_threshold.json \
  --readout-snapshot outputs/selective_draft/readout_smoke/snapshot \
  --max-steps 2 \
  --output-dir outputs/selective_draft/guidance_smoke

python -m selective_draft.cli train-guidance \
  --config configs/selective_draft/llada_threshold.json \
  --resume outputs/selective_draft/guidance_smoke/checkpoint-step-2 \
  --max-steps 4 \
  --output-dir outputs/selective_draft/guidance_smoke

python -m selective_draft.cli evaluate \
  --config configs/selective_draft/llada_threshold.json \
  --snapshot outputs/selective_draft/guidance_smoke/snapshot \
  --set guidance.mode=single_soft \
  --output-dir outputs/selective_draft/eval_guided

# 同一评测入口支持固定预算，不换生成函数。
python -m selective_draft.cli evaluate \
  --config configs/selective_draft/llada_threshold.json \
  --snapshot outputs/selective_draft/guidance_smoke/snapshot \
  --set generation.mode=fixed_budget \
  --set generation.steps=128 \
  --output-dir outputs/selective_draft/eval_fixed
```

实际 TACC 输出路径由脚本设为 SCRATCH，以上 outputs 只是相对路径示意。CLI 新增 flags 时同步更新 --help、示例和配置 schema，不交付失效的复制粘贴命令。

### 10.3 TACC 入口

新 smoke 脚本按显式 stage 运行：env、backend、decode、readout、teacher、guidance、resume、eval、ddp。不要默认 all 就启动长训练，也不要复制旧 dev_smoke.sh 中无提示删除整个训练目录的逻辑。

运行脚本从 repo root 启动，source tacc/env.sh；MODEL_PATH、MODEL_REVISION、CONFIG、OUTPUT_DIR、MAX_STEPS、NUM_GPUS、MASTER_PORT、RESUME_FROM 等均允许环境变量覆盖。单节点训练默认 3 卡，单卡评测可用 gpu-a100-small。日志目录在 sbatch 之前创建。

## 11. 评测、诊断与时间测量

### 11.1 必须保存的结果

每个问题一行：example_id、prompt/hash、generation、parsed_answer、is_correct、truncation/stop_reason、effective_output_tokens、实际 steps、model forward 次数、gate 次数、延迟、snapshot/config hash。每步/每状态详细信息另存，不塞进 benchmark JSON。

汇总至少包括：

- task accuracy、题数及按问题计算的不确定性；
- 单请求生成时延 median / p90、总生成时间、每题 NFE；
- 每步提交量分布、gate coverage（分别以全部活动步及存在 Receiver 的步为分母）；
- source 与 base final argmax 一致率、指导后 source 身份变化率、source 同步提交比例；
- 0.8/0.9 门控覆盖率与上述一致性 proxy，明确它们不是事实正确率；
- 输出长度与截断率，避免更短但更差的输出看起来更快；
- 峰值显存，训练 Teacher / rollout / Student backward 的时间分项。

### 11.2 计时协议

1. base 和 guided 使用相同硬件、dtype、attention 实现、batch、prompt、长度、停止规则与输出解析。
2. 加载模型、编译/首次 kernel 初始化、warmup、Teacher 诊断和文件写入不计入纯生成时间；另记整体作业耗时。
3. CUDA 同步后用 perf_counter 计完整 generate，结束时再同步；benchmark 关闭详细 trace、仅为日志新增的逐步 .item()/打印和磁盘写入。必要统计尽量在设备上累积，计时后取出；自适应循环和门控实际需要的 host/device 同步成本仍计入生成时间。
4. 每个方法使用相同预热与重复次数；运行顺序交替或随机化，避免温度/缓存偏差。
5. 报告实际前向调用；batch NFE、每行有效步数分开，不能把 batch time 除以 batch size 冒充单请求 latency。
6. iLLaDA growing 模式和 LLaDA fixed 模式分别重跑自己的 base。方法跨模型有效性看各自提升，不比较不一致协议的原始数字。
7. 同一个 commit_threshold 是初始公平比较；最终在独立开发集选阈值，报告相近准确率的速度或相近时延的准确率。

### 11.3 最小对照组

- base：新解码核心 + 原模型，无早期额外计算。
- readout_only / zero injection：保留早期读出开销，原输出不变，用于 parity 与开销测量。
- trained single_soft：正式候选。
- message intervention：固定状态、A 和 gate，改变消息身份/分布，检查是否依赖具体消息；需要做任务级对照时保持相同计算预算并报告其人工性。

Stage 1 的无消息模型在功能上应等价于 base，不能把它包装成另一个有质量变化的算法。低 KL、稳定梯度或一次 smoke 答对都只说明实现/优化可运行，不构成方法成立。

## 12. 必要测试与逐项验收

不写只检查类名/字段存在的测试；覆盖下列会改变研究结论的行为。CPU 测试使用可控 tiny Transformer / fake model，不下载或运行预训练模型；真实模型验收放 TACC。

| 测试组 | 输入与检查 |
|---|---|
| 边界与输出恢复 | L28 对应 block[27] 输出；tuple cache 不丢；捕获无注入与 native forward 对齐 |
| 原计算路径一致 | mode=off、scale=0、未过门控、只有一个 MASK；不同 dtype 分别测；参数不变 |
| source 与门控 | prompt/padding/未来 block 分数再高也不选；平分按位置；0.799、0.8、0.9 边界；活动/完成行混合 |
| soft 消息 | 小词表枚举计算 sum r(v)E(v)；与 full实现、chunk实现一致；改变概率确实改变消息 |
| soft proposal 语义 | A 末层改写、A 不提交、k_t=1、多个 Receiver 提交均合法；预算只计实际写入 |
| 阈值提交 | >0.9 全部提交；恰好0.9不自动提交；空集合兜底；真实步数可变；非法MASK候选不挂死 |
| 固定预算 | k_t=0/1/2/更多，余数分配、block尾部，不越界不丢token |
| Teacher 输入隔离 | 正常候选只改A；B仍MASK；无prompt/future改动；无指导hook污染；原state未原位改变 |
| mixture | tiny词表精确枚举；MC重复候选权重；top-K retained_mass；概率混合而非logit平均；梯度stop正确 |
| KL | q=p时0；词表求和后按位置/状态平均；ragged mask；不继承旧clip；对Student梯度正确 |
| 梯度路径 | Stage1仅g有grad；Stage2仅J有grad且通过冻结tail；零输出初始化的上投影可获得非零grad |
| 生命周期 | Teacher→Student→Teacher与异常抛出后，hook数量和base输出恢复；训练切换不污染base.eval |
| 状态与停止 | S_t回放只改C_t；包含最后block；EOS只查生成区；block完成才停止；growing长度与padding正确 |
| 恢复与DDP | 保存恢复一致；不同rank有效状态数、局部全关、全局全关；梯度等价于合并batch |
| RNG与trace | 开关trace、Teacher采样/诊断不改变固定seed的Student输出 |
| 新旧路径回归 | legacy配置+guidance off下对齐eval/generate.py；新停止策略和dtype另列，不混为parity失败 |

完整词表可能包含 MASK 这个特殊 ID。Teacher 如果从 r_A 抽到 MASK，数学上这是没有揭示的分支，目标为原 S_t 的预测；记录该事件，不能伪称输入恰好改了一处。提交器则不能将 MASK 当作成功解码。若要改变词表支持，必须形成显式协议配置并在 base/Student/Teacher 中一致处理，不能暗中重新归一化。

真实模型 parity 必须报告 max/mean logit error、source/commit set 差异和所用数值容差。FP32 tiny 测试要求严格；bf16真实kernel容差按观测制定并存档。若误差会翻转门控或提交选择，需要定位，而不是只看平均误差很小。

## 13. 建议实施里程碑与完成条件

### P0：接口与环境审计

交付 backend inspect、配置schema、split manifest、明确的模型/tokenizer revision 和环境版本报告。确认两个模型的 native module 路径、output norm、mask/EOS和权重共享。

完成条件：不假设不存在的文件/API；未知版本显式失败。iLLaDA config 记录 transformers_version=4.57.1，现有环境是4.49.0；这是兼容性检查项，不是自动认定必须升级。先在现有环境做导入/加载 smoke；需要新环境时隔离建立，不破坏旧环境。

### P1：解码核心 + 无注入 backend

交付阈值生成、固定预算策略、状态trace、scoped capture、CPU行为测试。新解码器训练/评测共享，旧入口不变。

完成条件：CPU测试通过；TACC少量真实状态的native/关闭指导parity通过；阈值首轮使用0.9，固定预算回归支持非固定2-token的k_t。

### P2：早期读出与阶段一

交付g、Stage1 KL、数据读取、snapshot与恢复、读出诊断。先2–4个更新验证梯度和恢复，再做有预算的小规模读出学习。

完成条件：仅g更新；原末层预测在无消息时不变；能够报告0.8门控的真实覆盖率和一致性proxy，未出现大量非有限输出。

### P3：注入、Teacher mixture 与条件信号诊断

交付J、单源soft注入、MC/top-K目标、消息干预工具。先做CPU梯度/mixture测试，再在有限真实状态上测teacher目标和消息响应。

完成条件：Teacher只改A、梯度只更新J；Student训练/推理消息一致。另给研究诊断结论：目标是否有信号、错误消息风险如何。实现通过与研究有信号分别报告。

### P4：阶段二 on-policy 与可恢复训练

交付完整刷新循环、CPU状态buffer、DDP空样本处理、统计与预算测量。单卡通过后再验证3卡，不直接跑主训练。

完成条件：训练/恢复/评测贯通；global empty不会误更新或死锁；policy lag与状态来源可追溯。

### P5：首个匹配协议的任务实验

在预定开发/分析集上比较base与trained single_soft，主协议threshold；完成准确率和真实时延测量，补消息对照。之后再做fixed_budget并比较协议差异。

完成条件：结论包含样本、checkpoint、阈值、停止、温度和实际时延，不仅有loss曲线；无论正负结果均保存。若只有步数下降但时延无收益，应照实报告。

### P6：iLLaDA 强模型验证

复用接口，不复制一整套训练器；用独立模型配方、tokenizer与growing/stop设置重新跑base、readout可行性和有限消息验证。

完成条件：iLLaDA自身的零注入parity、两阶段训练梯度、shared-head状态与匹配任务评测都通过。若兼容性阻塞，保留明确错误、最小复现与环境需求，不声称已支持。

**功能交付验收**是这些接口、测试与运行路径可复现；**科研成功验收**是独立分析/测试下质量或时延改善。没有正收益不代表代码没写完，也不能靠改测试集或隐瞒负结果完成研究目标。

## 14. TACC 运行边界与预算记录

沿用 AGENTS.md 和 CLAUDE.md 的有效操作约定：

- 本地只做代码、文档与不加载预训练模型的CPU测试；模型训练/推理/评测在计算节点。登录节点只做允许的文件、缓存准备和作业管理。
- 环境与HF缓存放WORK，运行输出放SCRATCH，重要snapshot、manifest和日志备份到WORK。
- source tacc/env.sh；显式HF_HOME、HF_HUB_OFFLINE=1、WANDB_MODE=offline，保留现有BNB_CUDA_VERSION等兼容设置。
- 先计算节点smoke再sbatch；默认单节点gpu-a100（3卡），单卡评测可gpu-a100-small。
- 提交前报告节点/队列/预计时长/SU；费率与qlimits按官方当前信息复核。单作业估算>30 SU、多节点或gpu-h100时按项目约定先询问，已有明确授权优先。
- 预算先由P1–P4实测：每个rollout耗时、每state候选数、Teacher forward耗时、Student backward、样本筛选率。不能从历史d-OPSD的训练时长直接推新方案成本。
- 预算表达式可用：rollout总耗时 + readout训练耗时 + sum(实际唯一Teacher候选数 × Teacher耗时 + Student训练耗时)，再加验证/保存与合理余量。MC样本数不等于去重后的Teacher调用数。
- 结束后用sacct顶层记录计算实际SU，至少15分钟计费，更新results/EXPERIMENTS.md；旧47.38 SU是截至2026-10-02的历史累计，不是余额。
- 新脚本避免默认邮件/外部通知、自动升级依赖或无提示删除输出目录。所有大实验需要明确运行配置和输出路径。

## 15. Claude 每个里程碑的交付说明

每次提交实现结果时附：

1. 修改/新增文件及其职责；当前完成到P几，哪些步骤尚未执行。
2. CPU测试与TACC真实模型测试分别列出命令、结果、版本和产物路径；不要用fake测试代替真实模型结论。
3. trainable参数列表、gate覆盖率、梯度流检查、数值parity与checkpoint恢复结果。
4. 若执行过集群作业，附job ID、实际耗时/SU、manifest和日志；未运行的命令明确标为示例。
5. 任何方法语义变化，例如hard消息、截断Student消息、多轮循环、更新主干、不同EOS规则，必须单独说明，不能作为工程优化悄悄混入首版。
6. 当真实数据表明early readout、teacher目标或收益不足时，保留失败样本与诊断，指出具体阻碍，不自动扩大训练或修改评测口径。

推荐从 **P0 → P1 → P2** 开始编码。先把“阈值轨迹 + 正确读出层 + 原输出一致性 + 单头可训练”做成可靠基础，再接入条件Teacher与on-policy消息训练。


