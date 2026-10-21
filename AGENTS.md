# dOPSD 项目记忆

更新：2026-10-18。这是简短常驻上下文；研究文档已合并，细节按下面的来源核对，用户最新指令优先。

## 沟通偏好（2026-10-17）

- 用户在终端阅读回答，无法渲染 LaTeX。数学内容优先用中文解释、ASCII 纯文本公式或代码块；需要公式图片时提供已渲染的图片，不直接输出 LaTeX 源码。

## 研究目标与当前状态

- 本 repo 基于清华的 d-OPSD（Learning from the Self-future），研究用 on-policy self-distillation / privileged information（PI）改善 dLLM 步内并行 token 决策的协调性，争取更好的准确率—实际时延权衡。
- 已运行模型是 LLaDA-8B-Instruct；近期 Joint 主任务为 MATH，GSM8K 是历史实验与后续迁移任务。下一版计划迁移 LLaDA2.0-mini，mini-CAP 为后续参照；尚未运行 mini。
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

1. 先跑mini base MATH：block32、T0、confidence>.95、每block steps32。**2026-10-19用户定：max_new_tokens=2048（已试过不截断）**；官方MATH prompt/scorer未公开，暂用KLASS system prompt + KLASS判分（legacy判分并列核对），不再因官方recipe缺项阻塞后续阶段。
2. 全层诊断决定readout层，不预设L16/L17；64题验证＋64题一次确认。G绑定官方.95，全部过阈位置入组，无固定4人预算；真实mini记录消息路径与MoE尾层梯度。
3. 用户已接受root_success_v2：训练T=.9，每题从prompt最多8次、首次答对即停；成功轨迹均匀抽1个有组state，用自身最终tuple做LOO；失败跳过，无额外续写或重抽。6000训练池首遍不放回，按贡献题、SU与独立dev选checkpoint。
4. 首轮一条主线；后续需要机制归因时做同数据LOO-KL/参考CE及同伴消息干预。相关工作与历史否定结论见研究概览；Partial Loop等不自动加入实验。

## 当前代码与资料入口

- `selective_draft/`：已实现单源与Joint-v1流水线；用法见 `selective_draft/README.md`、`selective_draft/JOINT.md`，CPU测试在 `tests/test_selective_draft_*.py`。mini（`backend=llada2_mini`）与root_success_v2已实现并有CPU测试（`tests/test_selective_draft_llada2.py`，用真实mini源码的小模型），作业 `tacc/llada2_pipeline.slurm`；**尚未在真实权重/GPU上运行**，用法见 `selective_draft/README.md` 末节。
- `doc/Selective_Draft_OPSD.md`：合并后的研究概览，含方法演进、数学修正、实验与诊断；替代旧分散研究文档。
- `doc/llada2-mini_handoff.md`：开工/同事交接短版；`doc/llada2-mini_engineering_plan.md`：详细规格与官方链接；`doc/decoding_protocol_audit_zh.md`：跨仓库参数核对。
- `results/joint/` 保存Joint结果、逐题案例与续写审计；历史现象分析已并入研究概览，`doc/` 只保留上述四份文档。
- `results/pipeline_gsm_20261002/SUMMARY.md`、`metrics.csv`、`sweep.csv`：历史复现结果与指标；`results/EXPERIMENTS.md`：TACC 作业、实际耗时与 SU 账本。
- `d-opsd/d_opsd_train.py`、`d_opsd_trainer.py`、`d_opsd_config.py`：实际训练入口、trainer、配置；`d-opsd/run/*/opsd.sh`：任务启动脚本。
- `eval/generate.py`：固定预算/阈值生成；`analysis/llada_trace.py`、`pi_sibling_analysis.py` 等：真实轨迹与反事实分析。
- `tacc/`：环境与 Slurm 脚本；`env.yml`、`used-env.txt`、`tacc/setup_env.sh`：依赖/环境来源。
- `CLAUDE.md`：已完整阅读，TACC 操作手册与工作约定可参考；A 部分代码地图、Dream/旧 trajectory 入口、environment.yaml 版本等属于旧背景，与当前 repo 不一致。其 steps==tokens 的旧要求不作为本并行研究的不变量。

## 操作约定

- 本地 WSL 只编辑代码、读日志、整理资料，不启动训练或模型评测；训练/评测主要在 TACC Lonestar6。
- 延续 CLAUDE.md 的 TACC 约定：默认单节点 gpu-a100（3×A100-40GB），单卡评测可用 gpu-a100-small；先小规模验证再批作业。提交前报告预算，单作业估算 >30 SU 或多节点/gpu-h100 时先询问用户，已有明确授权优先。
- 费率是 2026-09-20 的历史值；使用前按官方文档/qlimits 复核。SU 按实际运行时间计算，至少计 15 分钟；结束后用 sacct 记账，47.38 SU 是截至 2026-10-02 的历史累计，不是当前余额。
- TACC 缓存/环境在 WORK，作业/输出在 SCRATCH，重要产物备份到 WORK；计算节点设置 HF_HUB_OFFLINE=1、WANDB_MODE=offline 和显式 HF_HOME。沿用 `tacc/env.sh`。
- 不在登录节点跑训练/推理；环境有 TRL 补丁和 TACC CUDA/GLIBC 适配，不随意升级依赖。脚本配置保持环境变量可覆盖的风格。
