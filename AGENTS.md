# dOPSD 项目记忆

更新：2026-10-21。这是简短常驻上下文；研究文档已合并，细节按下面的来源核对，用户最新指令优先。

## 沟通偏好（2026-10-17）

- 用户在终端阅读回答，无法渲染 LaTeX。数学内容优先用中文解释、ASCII 纯文本公式或代码块；需要公式图片时提供已渲染的图片，不直接输出 LaTeX 源码。

## 研究目标与当前状态

- 本 repo 基于清华的 d-OPSD（Learning from the Self-future），研究用 on-policy self-distillation / privileged information（PI）改善 dLLM 步内并行 token 决策的协调性，争取更好的准确率—实际时延权衡。
- 已运行模型是 LLaDA-8B-Instruct；近期 Joint 主任务为 MATH，GSM8K 是历史实验与后续迁移任务。当前工作是迁移到 LLaDA2.0-mini（mini-CAP 为后续参照）；mini 的 baseline 尚未跑完。
- 核心缺口：双向 attention 已让位置共享上下文，但同一步 Receiver 的最终预测没有再条件于 Anchor 本次实际确定的 token identity。不要把问题表述为“token 之间没有 attention / 信息共享”。
- THU self-future PI 与 sibling PI 不同：前者提供后续 block 的部分最终 rollout；后者在相同真实去噪状态 S_t 中只揭开同批 Anchor A=a，其他位置不变。
- 当前主线是 Joint sibling PI：早层 soft draft 消息注入，冻结尾层传播，末层决定提交；不锁定硬 Anchor。Early Commitment / Partial Loop 是历史候选，尚无有效性结论，不自动恢复为主线。
- Early commitment 必须保证最终写入的 Anchor 与内部条件一致；允许末层改写的 early proposal 是不同语义。固定预算下还需明确 Anchor 是否占用提交额度。

## 已有实验与证据边界

- 固定预算协议：256 generated tokens / 128 steps / block length 32，平均每步提交 2 tokens。阈值协议：当前 block 中 confidence > 0.9 的位置同时提交，至少提交最高置信的 1 个，步数自适应。
- 2026-10-02 pipeline：3×A100、4-bit + LoRA r=128、BATCH_DIVIDE=8，训练 1344 steps；300 题 greedy checkpoint sweep 选出 step-448，base 78.3% → 82.7%。后期明显退化，step-1344 为 18.0%；不要把最终 checkpoint 当最佳模型。
- 同批 300 题、T=0.9：固定预算 base 75.0% → step-448 82.0%；阈值 0.9 为 79.3% → 79.7%。这些是历史结果，checkpoint 在同一 test 子集选出，不能当作独立 held-out 的最终效果。
- 历史 sibling response：固定预算 D_S 均值约 −0.130 nat、负向约 26%；阈值均值约 +0.017～+0.02、负向约 7%。future PI 的 sibling 特有缓解较小，也降低一般上下文敏感度；尚不能说原 d-OPSD 已学会了显式步内协调。
- 早期研究记录的受控 Reveal（128 个 Receiver）：正确 Anchor 使目标 log-prob +0.099，Top-1 100/128 → 102/128；错误 Anchor 使 log-prob −2.095，正确率降至 58/128。说明 identity 有影响，错误 Anchor 风险大；不是已证明大幅任务增益。
- 早期研究记录的原 LM Head 逐层读出：L28 55.47%、L31 97.66%；activation patch 恢复 Reveal 效应：L8 104%、L16 74%、L24 37%、L28 19%、L31 6%。不能证明早层没有 identity，也不能将 patch 比例理解为 accuracy 增益或可学习注入器的上限。
- Training-free L24 全位置 hard Draft feedback（alpha=0.1、conf>0.9、GSM8K 500 题）：30.8% → 26.6%，抽查 L24/L32 agreement 约 18%；alpha=0 parity 通过。该具体方案负收益；可靠性与 residual 注入对齐的影响尚未区分。
- 暂缓全位置 L24 Early Distillation；L32 外挂 Refiner 不作为当前主线。不要把候选训练、Partial Loop 或未验证架构写成已完成成果。
- 不同批次的样本、checkpoint、prompt、温度、长度、停止/答案抽取协议不同；尤其 30.8% 与历史 79.3% 不能直接比较。v3 新诊断数字来自研究记录，原 trace / 脚本仍需回查。
- D_S 是反事实 reveal 的 log-prob shift，不无条件等于严格 PMI。训练后 student 与 base teacher 的比较混入权重差异，不能全归因于 PI。
- Joint-v1：MATH-500 base 34.4% → 35.0%，NFE约−18.1%、实际时延约−11.0%；32修复/29损伤、净+3题，准确率提升未确证。100updates访问327题、178题贡献525states/2019位置，非训练6000题一遍。
- TACC续写审计：226条自然错轨迹中77条得到LOO参考，贡献29.4%监督位置、消耗73.8%续写NFE；这是每题4×8分支，不是单状态8次。主干冻结但injector训练，成功路径学习仍是收益的可能解释。

## 最新下一步（记录计划，不自动执行）

1. 先跑mini base MATH-500，**设置完全对齐DMax对LLaDA2.0-mini的评测**（2026-10-21用户定）：普通阈值解码（不是DMax自己的软并行解码）、block 32从位置0起算、T=0、置信度>=min(0.95, 最高-1e-5)、argmax为MASK的位置不参与、block末遇EOS才停、总长度向下取整到32（max_new_tokens=2048）、输出=去掉所有EOS后的全部token、DMax prompt与判分、NFE/TPF按DMax评测脚本口径。参照数字：DMax 75.8%/TPF 2.58（长度2048、dInfer、2×H200），CForce 84.2%/TPF 2.99（长度4096、SGLang）。kernel层面（融合MoE、TP、compile）不追求一致：只有bf16舍入噪声，用户决定不装dInfer/sglang。
2. baseline看过再决定后续；然后全层诊断决定readout层（不预设层号），64题验证＋64题一次确认，G绑定同一阈值，全部过阈位置入组。
3. root_success_v2：训练T=.9，每题从prompt最多8次、首次答对即停；成功轨迹均匀抽1个有组state，用自身最终tuple做LOO；失败跳过，无额外续写或重抽。6000训练池首遍不放回，按贡献题、SU与独立dev选checkpoint。
4. 首轮一条主线；后续需要机制归因时做同数据LOO-KL/参考CE及同伴消息干预。

## 当前代码与资料入口

- 仓库已精简为LLaDA2.0-mini项目（分支 `llada2-mini`）。`selective_draft/`：backend.py（块因果attention、可微MoE、前缀KV cache）、generation.py（DMax评测语义的解码器，文件头写了每条规则）、model.py（hook与组规则）、training.py / root_joint.py（readout与Joint训练）、evaluation.py（分片可续跑评测）、layer_profile.py（选层与readout选点）、scoring.py + vendor/dmax_val_math.py（DMax判分原样）、cli.py。CPU测试 `tests/test_{decoder,guidance,training,data}.py`（需transformers 4.57，用真实mini源码的小模型），集群作业 `tacc/llada2_pipeline.slurm`。**真实权重/GPU上只跑过baseline的前几题**（job 3519705，OOM后失败），后续阶段从未在GPU上运行。
- LLaDA-8B那一版（含KLASS适配、单source流程、same_state_v1 Joint、THU d-OPSD复现代码）在 tag `llada-joint-v1` / 分支 `joint`、`THUdOPSD`，需要时从那里取。
- `doc/Selective_Draft_OPSD.md`：研究概览；`doc/llada2-mini_handoff.md`：交接短版；`doc/llada2-mini_engineering_plan.md`：详细规格与官方链接（其中EOS/长度/KLASS相关条目已被DMax对齐决定取代，以代码头注释为准）；`doc/decoding_protocol_audit_zh.md`：跨仓库参数核对。
- `results/EXPERIMENTS.md`：TACC作业、实际耗时与SU账本；`results/joint/` 等：LLaDA-8B历史结果。
- `README.md`：用法；`requirements-llada2.txt` / `env-llada2-freeze.txt`：环境；`tacc/qstatus.sh <JOBID>`：排队位置与粗估开始时间。
- `CLAUDE.md`（不入库）：TACC 操作手册与工作约定；其 A 部分代码地图是旧背景，与当前 repo 不一致。

## 操作约定

- 本地 WSL 只编辑代码、读日志、整理资料，不启动训练或模型评测；训练/评测主要在 TACC Lonestar6。
- 延续 CLAUDE.md 的 TACC 约定：默认单节点 gpu-a100（3×A100-40GB），单卡评测可用 gpu-a100-small；先小规模验证再批作业。提交前报告预算，单作业估算 >50 SU 或多节点/gpu-h100 时先询问用户，已有明确授权优先。
- 费率是 2026-09-20 的历史值；使用前按官方文档/qlimits 复核。SU 按实际运行时间计算，至少计 15 分钟；结束后用 sacct 记账，47.38 SU 是截至 2026-10-02 的历史累计，不是当前余额。
- TACC 缓存/环境在 WORK，作业/输出在 SCRATCH，重要产物备份到 WORK；计算节点设置 HF_HUB_OFFLINE=1、WANDB_MODE=offline 和显式 HF_HOME。沿用 `tacc/env.sh`。
- 不在登录节点跑训练/推理；不随意升级依赖（见 requirements-llada2.txt）。脚本配置保持环境变量可覆盖的风格。
