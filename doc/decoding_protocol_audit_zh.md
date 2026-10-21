# LLaDA 系列评测配置核对：THU、KLASS、原始 LLaDA、iLLaDA 与本项目

核对日期：2026-10-17。逐项阅读官方仓库的**启动脚本、实际生成函数、prompt 和评分代码**；本项目依据保存的运行配置与 argv。外部来源固定到下方 commit，不把函数默认值、脚本设置、论文成绩混为一谈。本次没有运行模型或提交 TACC 作业。

## 0. 数据划分与本次纠正

**MATH-500 的 500 指测试题数，不是训练数据总量。** 我们训练源 `ankner/math-500` 的 train 有 7500 题；seed=42 打乱后分为 train 6000、dev 500、analysis 500，另外 500 题未使用。测试单独加载 `HuggingFaceH4/MATH-500` 全部 500 题，不是从这 7500 题中抽取。

本次 Joint 的 6000 是题池大小：100 次有效更新、9 次跳过、3 个 rank，共保存 **327 道不同训练题**的 rollout，全部属于 train 划分。复用的 readout 来自此前 80 道训练题的采集，其中 23 条答对轨迹用于 readout 训练；两阶段题数不能直接相加当成不同题数。最终 34.4% / 35.0% 是另外 500 道测试题的成绩。

核验不只依赖 split 名称：base 和 Joint 的实际 prompt hash 均与这 500 道测试题逐一匹配；另下载全部 7500 道训练源题，按题目正文（统一空白）比较，与测试题重合为 **0**。

| 仓库 / 设置 | 额外训练数据 | 测试数据与划分 |
| --- | --- | --- |
| 我们 Joint | 6000 题池，本次实际 327 道；另复用已有 readout | MATH-500 全部 500 题 |
| THU d-OPSD | `get_math_questions("train")` 加载 `ankner/math-500` 全部 7500 题；shuffle 后未另留 MATH dev | `HuggingFaceH4/MATH-500` 全部 500 题 |
| KLASS | 推理方法，无额外训练 | `data/math_test.json` 全部 500 题；此前已与 HF MATH-500 逐题对齐 |
| LLaDA 原始 Base 的 31.4 行 | 此处评测已有模型 | `minerva_math`，MATH 各学科 test 合计 5000 题，非 MATH-500 |
| iLLaDA 官方 MATH 脚本 | 此处评测已有模型 | `minerva_math`，各学科 test 合计 5000 题；CLI 设 0-shot |
| Fast-dLLM v1 | 推理方法，无额外训练 | 论文为 MATH 4-shot、lm-eval；公开 v1 脚本未提供 MATH 专用命令，无法据此确认论文实际测试题数，不能标成 MATH-500 |

THU 训练来源见 [data_utils.py 的 get_math_questions](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/data_utils.py#L135)；我们划分见 `outputs/joint_math_v1/meta/split_manifest.json`，实际采样见 `train/rollouts/*.pt`。原始 LLaDA 的 OpenCompass Instruct 路径另见 §3，不能与其 Base/lm-eval 路径混用。

**纠正 Fast-dLLM：**它确实是 NVIDIA 等机构的工作。此前引用 79.2% 是 GSM8K 5-shot，不是 MATH；只查看 README 而遗漏论文 MATH 表格是不完整的核对。论文 Table 1 的真实 MATH 数字补在下表。因此“所有 baseline 都在 31 左右”不成立；我们 34.4% 与 KLASS 的差距仍需对齐协议，不能据此判断方法更优。

**THU 涨点是否只是改 block？** [原论文](https://arxiv.org/abs/2606.18195) Table 1 的长度 256 MATH-500 是原始模型 **31.8% → d-OPSD 37.2%**；表注称 base 使用公开模型评测，Evaluation Details 统一声明 block=32、去噪步数为生成长度一半。**这是作者的声明，尚无公开逐题输出或运行日志证明 31.8% 的实际配置。** 仓库 `eval/eval.py` 支持空 `checkpoint_path` 评测原始模型；但 MATH shell 只给了 checkpoint 路径占位符，没有独立 base 启动脚本或 31.8% 的结果产物。因此不能把“论文声明同协议”升级成“已核验同协议”，也不能据此排除 block/其他配置混杂；目前同样没有证据证明作者直接复制了原始 LLaDA 成绩。[Block Diffusion（ICLR 2025）](https://arxiv.org/abs/2503.09573)研究段间自回归、段内扩散的建模与训练，支持 block 是质量—并行度的重要选择，但不能直接推出冻结 LLaDA 改 64→32 会提高几个点。LLaDA 官方自身也报告分块采样改善数学成绩，同时改变 EOS 设置，不能把其全部改善归因于 block 单一因素。

## 1. 先回答：别人有没有固定阈值解码成绩？

**有。KLASS 明确报告了 LLaDA 的 confidence >0.9 baseline；Fast-dLLM 也公开了阈值 0.9 解码及成绩。** 原始 LLaDA、THU 原脚本和下面的 iLLaDA MATH 脚本，则使用固定提交预算。

| 来源与方法 | 任务 / 模型 | 准确率 | 实际步数 | 证据与限制 |
| --- | --- | ---: | ---: | --- |
| KLASS：confidence-only，阈值 0.9 | MATH-500 / LLaDA-8B-Instruct | **31.6%** | **96.46** | 用户提供 PDF，第 6 页 Table 1；正文 §6.1 定义阈值。仓库没有独立复现该行的脚本，见 §5 |
| KLASS：KLASS，confidence 0.6 + KL 0.01 | 同上 | 33.8% | 128.62 | 论文 Table 1；官方 MATH 脚本 K1、解码 K2 |
| Fast-dLLM：LLaDA baseline | MATH，4-shot / LLaDA-8B-Instruct | **33.5%** | 论文表未列 | F4 Table 1，长度 256；不能认定为同一 MATH-500 协议 |
| Fast-dLLM：仅 Parallel，阈值 0.9 | 同上 | **33.4%** | 论文表未列 | F4 Table 1；默认 block 32、阈值 0.9 见 §4.1 |
| Fast-dLLM：Cache + Parallel | 同上 | **33.2%** | 论文表未列 | F4 Table 1；吞吐 51.7 tokens/s，不是步数 |
| 我们：base，阈值 0.9 | MATH-500 / LLaDA-8B-Instruct | 34.4% | 100.42 | 本地保存结果 O1；不是 THU 固定 128 步协议 |
| 我们：Joint，阈值 0.9 | 同上 | 35.0% | 82.23 | 同一评测协议，另有 readout/注入开销 |

Fast-dLLM v1 README 没有 MATH 行，但论文有：长度 512 时，baseline / Parallel / Cache+Parallel 分别为 37.2% / 36.8% / 36.0%。这些是 MATH 4-shot，不能直接和我们的 0-shot MATH-500 比。KLASS 中增加稳定性条件可以增加步数，所以其 128.62 步大于 confidence-only 的 96.46 步并不矛盾。

## 2. 生成配置对照表

下面的长度均为**新增 token 上限，不含 prompt**。原始 LLaDA 分成多个设置，避免把 31.4、42.2 和不同模型混在同一行。

| 项目 / 实验设置 | 模型 | 输出上限 | Block | 步数参数的实际含义 | 提交规则 | 温度 |
| --- | --- | ---: | ---: | --- | --- | ---: |
| THU d-OPSD：官方 MATH 评测脚本 T1 | LLaDA-8B-Instruct，按 checkpoint 加载适配器 | **256** | **32** | **整题 128 步** | 每步置信度最高的 **2 个**位置，无概率阈值 | 0 |
| KLASS：Top-1 baseline | LLaDA-8B-Instruct | 256 | 64 | 整题 256 步 | 每步 1 个位置 | 0 |
| KLASS：confidence-only（论文定义） | 同上 | 256 | 64 | 自适应，均值 96.46 | confidence >0.9；没有达标位置则 fallback | 0 |
| KLASS：官方 MATH 脚本 K1 | 同上 | 256 | 64 | `steps=256` 提供上限/每 block fallback 计划；实际自适应 | confidence >0.6，且最近 2 次 KL 都 <0.01；否则 fallback 1 个 | 0 |
| LLaDA 原始 Base：EVAL.md 的 31.4 行 L1 | **LLaDA-8B-Base** | **1024** | **1024** | 整题 1024 步 | 每步 1 个，无概率阈值 | 0 |
| LLaDA Instruct：纯扩散 L2 | LLaDA-8B-Instruct | **512** | **512** | 整题 512 步 | 每步 1 个；另有 EOS/EoT 特殊处理 | 0 |
| LLaDA Instruct：分块扩散 L3 | 同上 | **512** | **64** | 整题 512 步，每 block 64 步 | 每步 1 个，无概率阈值 | 0 |
| iLLaDA Instruct：官方 MATH 脚本 I1 | **iLLaDA-8B-Instruct** | **4096**，再受剩余上下文限制 | **32** | `var=True, steps=32`：**每个 block 32 步**，不是整题 32 步 | 每步 1 个，无概率阈值 | 0 |
| Fast-dLLM：Parallel F2 | LLaDA-8B-Instruct | 256 | 32 | `steps=8` 不是实际 NFE；阈值模式自适应 | confidence **>=0.9**；保证至少提交 1 个 | 0 |
| 我们：base / Joint O1/O2 | LLaDA-8B-Instruct | **256** | **32** | JSON 的 `steps=128` **不生效**；实际均值 100.42 / 82.23 | confidence **>0.9**；无达标位置则提交 1 个 | 评测 0；训练 rollout 0.9 |

**三个最容易误读的参数：**iLLaDA 的 32 是每 block 的步数；Fast-dLLM 阈值模式下的 8 不是整题只做 8 次前向；我们的 128 不是阈值模式的固定运行步数。

## 3. 数据、prompt、停止与评分对照

| 项目 | THU 官方 | KLASS 官方 | 原始 LLaDA Instruct 官方分块设置 | iLLaDA Instruct 官方 MATH 设置 | 我们实际运行 |
| --- | --- | --- | --- | --- | --- |
| 权重定位 | HF 模型名 + checkpoint 路径；未固定 HF revision | `./models/LLaDA-8B-Instruct` 本地目录，未给权重 hash | 模型配置指向作者本地目录，未给权重 hash | HF iLLaDA 模型名，未固定 revision | 固定 revision `08b83a6f...`，无 LoRA，base 对照不载入 Joint snapshot |
| 测试数据 | `HuggingFaceH4/MATH-500` | 仓库 `data/math_test.json`，实为 JSONL；已与同一 500 题逐题核对 | `opencompass/math`；脚本没有指定 MATH-500，不能拿它当同一 500 题 | lm-eval 的 `minerva_math`，指向 `EleutherAI/hendrycks_math` 的各学科 test；没有设置 500 题子集 | `HuggingFaceH4/MATH-500` 全部 500 题 |
| Few-shot | 0 | 0 | 0 | CLI 明确 0，覆盖 minerva task 默认的 4-shot | 0 |
| Prompt | user 内放要求；reasoning/answer 标签和 boxed；生成前追加 reasoning 标签 | 独立 system 要求逐步推理，以 `The final answer is` 作答 | user：题目 + `Please reason step by step, and put your final answer within ...`；使用 boxed | lm-eval task 的 prompt + 模型自己的 chat template；不使用 THU 标签格式 | **沿用 THU**，包括原模板末尾多余双引号 |
| 未来 MASK 窗口 | 一开始放完整 256 个 MASK | 一开始放完整 256 个 MASK | 一开始放完整 512 个 MASK | **逐 block 扩展**，没有未来 MASK block | 一开始放完整 256 个 MASK |
| EOS / 停止 | 完整解码；`stop_recording` 只影响统计，不退出主循环 | 完整处理 256 位置；没有 EOS 提前退出 | 完整处理 512 位置；解码后移除特殊 token | 每完成 block 检查 task stop words/EOS；停止后截断文本 | 每完成 block 检查 EOS；输出在首个 EOS 截断 |
| EOS 特殊设置 | 未启用 LLaDA 官方的 EOS/EoT confidence 抑制开关 | 普通预测 | 分块设置默认两开关均 false；纯扩散设置 confidence 抑制 true、logits 抑制 false | MASK=5；另对结束思考标记加 logit boost，值 100、power=3 | MASK=126336，EOS=126081；不抑制 EOS；禁止候选为 MASK |
| 评分 | 最后 boxed + `parser_helper.is_equiv` | 较丰富的抽取/归一化；分数可转小数；`compare_answers` 比较候选集合交集 | `math_postprocess_v2` + `MATHEvaluator(version=v2)`；不是 THU matcher | lm-eval task，含 exact_match / math_verify；具体报告指标需与原结果文件对齐 | 主要沿用 THU matcher，增加部分抽取回退；已知误判等值答案 |
| Softmax 精度 | `F.softmax(logits)`，未显式统一为 FP32/64 | LLaDA 解码显式 FP64 | 原生成函数未显式统一升精度 | 原生成函数未显式统一升精度 | 显式 FP32 |
| 测试 batch | 长度 256 时脚本为 8 | 1 | 1 | 1 | 1 |
| 推理 cache | THU 生成代码未提供 Fast-dLLM cache 路径 | 无额外 cache 方法 | 此处原始代码路径 | 此处原始代码路径 | 未使用 Fast-dLLM cache |

两个细节需要特别区分：

- **LLaDA 文件名中的 `confidence` 不表示概率阈值解码。** `diff_confidence_eos_eot_inf` 是 EOS/EoT 处理选项；实际仍由 `gen_steps` 决定提交预算。不能仅凭文件名归类。
- **停止与评分文本是两个问题。** 我们在 EOS 后既停止后续 block 计算，又截掉后续文本；KLASS 完成整个窗口后用 `skip_special_tokens=True` 解码。即使只想对齐步数，也不能忽略进入 matcher 的文本不同。

原始 LLaDA 的 OpenCompass 配置指向外部 `math.json`，仓库没有随该脚本固定测试文件 hash；本次不把其报告分数认定为已在相同 MATH-500 上验证。

训练配置也不是完全相同：

| 项目 | THU d-OPSD MATH 配置 T3 | 我们的 Joint |
| --- | --- | --- |
| 更新对象 | 4-bit 主模型加载 + LoRA，r=128、alpha=64、dropout=0.05 | 主干 BF16 冻结，仅 FP32 injector，rank=64；无量化、无 LoRA |
| 学习率 | 5e-6 | 1e-4，针对新模块 |
| 优化器共有项 | Adam betas=(0.9,0.99)、weight decay=0.1、clip=0.2、warmup ratio=0.0001 | 沿用这些值 |
| Pass@8 / 温度 | passk=8、T=0.9；不是测试 pass@8 | 每个抽中状态续写 8 次、T=0.9；不是整题初始状态采 8 次 |
| PI / 损失 | self-future，retain ratio=0.25；配置 top_k_loss=20、beta=1、jsd_token_clip=0.04 | 同组共同参考、留一 Teacher PI，组内全词表 forward KL |
| 长度 / 解码配置 | max_completion_length=256、block=32、diffusion_steps=128，原生成路径固定预算 | 同长度与 block，改为阈值解码，steps=128 不生效 |

该表核对的是 THU YAML 与 MATH shell 的最终覆盖值，不表示两种训练方法或训练预算相同。KLASS/Fast-dLLM 属于推理方法，此处没有额外训练；原始 LLaDA/iLLaDA 在表中作为已训练模型评测。

## 4. 哪些成绩能与我们比较

| 数字 | 对应什么 | 与我们是否同协议 |
| --- | --- | --- |
| LLaDA 官方 **31.4%** | **Base**，长度/步数/block 都为 1024，官方 EVAL.md 的 lm-eval 行 | **否：模型版本、长度、block、数据/任务和解码都不同** |
| KLASS 的 **31.4%** | **Instruct**，MATH-500，256/256/64，Top-1 | 否；只是恰好同为 31.4 |
| LLaDA 官方 **31.9% / 29.6%** | Instruct 纯扩散，512/512/512，分别为内部工具 / OpenCompass | 否 |
| LLaDA 官方 **42.2% / 42.7%** | Instruct 分块，512/512/64，分别为内部工具 / OpenCompass | 否，不能据此判断我们的 256-token 结果偏低或偏高 |
| iLLaDA 模型卡 **56.7%** | iLLaDA Instruct 的 MATH 报告；是另一个模型 | 否；模型卡不是本次重跑日志，不能保证当前脚本逐位复现它 |
| KLASS **31.6% / 96.46 步** | 同模型、同 MATH-500、confidence-only 0.9 | **最接近，但 prompt、block、EOS 和评分仍不同** |
| 我们 **34.4% → 35.0%** | 同一批 500 题、同 prompt/解码/评分，base → Joint | **可以作内部对照**；净多 3 题，尚未证明稳定准确率增益 |

用 KLASS 官方评分函数对已有输出重算，base 为 35.2%、Joint 为 36.6%。这说明**评分差异确实存在**；它没有统一 prompt、block 或生成规则，也不是新跑出的 KLASS 对照。新增判对样本主要涉及分数/小数、单位、逗号、区间和答案抽取，详见[重评分记录](../results/joint/klass_rescore.json)。

## 5. 代码核对发现的具体问题

1. **“按 THU 配置”这个说法范围过大。** 我们只沿用了 256、32、prompt 和部分优化器超参；THU 的固定 2-token 解码已改为阈值解码。这是明确的协议差异，不能要求复现其原分数。
2. **当前 matcher 有实际问题。** 例如 `1.2` 与 `6/5`、`8/18` 与 `4/9` 可被判成不同。应统一各方法的数学答案评分，再比较涨点；原始结果应保留，避免悄悄换分数。
3. **KLASS 的 confidence-only 行缺少直接复现入口。** K3 的 `alg` 只有 klass/default/random/topk_margin/entropy；K2 只有 klass 分支会启用阈值 ready_mask。`--alg default` 是固定 Top-k，不是 confidence-only；仓库现成 MATH 脚本只跑 confidence+KL 联合条件。论文 31.6 是明确报告的结果，但目前不能从一个公开 baseline 开关直接对应到那一行。
4. **iLLaDA 不等于把 MASK id 改成 5。** 官方 MATH 使用更长的输出预算、逐 block 扩展、结束思考 boost 与不同 task/scorer。我们仓库的 `illada_threshold.json` 已标为未运行占位方案，不能视作复现了 iLLaDA 官方评测。
5. **元数据要和实际 argv 一起读。** 我们 JSON 保留的 `steps=128`、`message_threshold=0.8` 不控制本次阈值步数或通信入组；旧 readout JSON 的 max_steps 与实际 argv/log 也不一致。实际Joint流程见[训练分析](../results/joint/math_v1_analysis.md)，运行配置与argv入口见下方O1，旧readout见[初试报告](../results/selective_draft/math_prelim_v3_REPORT.md)。

**目前没有找到“gen length 偷偷变大”或“阈值代码变成固定预算”这样的实际错误。已确认的是基准混用与评分问题。** 我们比 KLASS confidence-only 高出的分数，究竟有多少来自 prompt、block 或其他实现差异，静态读代码无法定量归因。

## 6. 最小对齐方式（计划，未执行）

先固定原始 LLaDA-8B-Instruct 和一个独立 dev 小集，把模型、题目、温度、停止、评分统一；分别使用 THU 的固定 128 步、阈值 0.9、KLASS 联合条件。先在 **同一 prompt、同一 block** 下比较，再只切换 THU/KLASS prompt 与 block 32/64，定位 baseline 差距。现有代码已支持前两种解码，KLASS 需按官方 K2 规则实现或调用，不能用 `alg default` 冒充。

这组检查无需训练。只有基线对齐后，再加入 Joint，才能判断是否超过 KLASS，而不是比较两套不同的评测系统。此处没有提交新评测。

## 7. 可回查的代码来源

仓库固定版本：THU `70d146c525c7`；KLASS `ad87ecbe9fde`；LLaDA/iLLaDA `9182493720ed`；Fast-dLLM `a9b81e4caa24`。完整commit和核对路径直接保留在下表链接中，不依赖单独的来源清单JSON。

| 标记 | 本次实际阅读的入口 |
| --- | --- |
| T1 | [THU MATH 启动脚本，第 10 行起](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/eval/run/math/opsd.sh#L10) |
| T2 | [THU 生成函数](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/eval/generate.py#L184)、[prompt](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/eval/math500.py)、[评分](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/eval/parse_and_get_acc.py#L55) |
| T3 | [THU 训练 YAML](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/opsd.yaml)、[MATH shell 覆盖值](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/run/math/opsd.sh) |
| K1 | [KLASS MATH 启动脚本](https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/scripts/llada_math.sh) |
| K2 | [KLASS ready-mask 与 fallback 实现](https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/src/model/llada_klass.py#L121) |
| K3 | [KLASS prompt / 评分 / CLI](https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/src/llada_evaluation.py)、[matcher](https://github.com/shkim0116/KLASS/blob/ad87ecbe9fdef4a6e93701afbc150cb9213fbcd7/src/utils.py#L412) |
| L1 | [LLaDA 官方 EVAL.md 的模型与成绩表](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/EVAL.md) |
| L2 | [LLaDA Instruct 纯扩散 MATH 配置](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/opencompass/examples/llada_instruct_gen_math_length512_block512_confidence.py#L14) |
| L3 | [LLaDA Instruct 分块 MATH 配置](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/opencompass/examples/llada_instruct_gen_math_length512_block64.py#L14)、[其数据与 prompt 配置](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/opencompass/opencompass/configs/datasets/math/math_0shot_gen_11c4b5.py) |
| I1 | [iLLaDA MATH 启动脚本](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/eval_illada.sh#L103) |
| I2 | [iLLaDA 逐 block 调用 generate](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py#L204)、[上下文上限与停止](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/eval_illada.py#L396)、[模型卡成绩](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) |
| F1 | [Fast-dLLM v1 官方结果表](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/README.md#L103) |
| F4 | [Fast-dLLM 论文](https://arxiv.org/abs/2505.22618)，Table 1 的 MATH 4-shot 成绩、§4.1 的 block/阈值/lm-eval 设置；本次补查 arXiv 当前 PDF |
| F2/F3 | [Fast-dLLM 阈值脚本](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/eval_gsm8k.sh)、[阈值提交实现](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py#L298) |
| O1 | [本次解析配置](../outputs/joint_math_v1/train/resolved_config.json)、[base 评测 argv](../outputs/joint_math_v1/eval/base/run_manifest.json)、[Joint 评测 argv](../outputs/joint_math_v1/eval/joint/run_manifest.json) |
| O2 | [本地实际解码](../selective_draft/generation.py)、[实际训练流程](../results/joint/math_v1_analysis.md)、[详细成绩](../results/joint/math_v1_detail.md) |

KLASS 论文来源为用户提供的 `E:\MSIS\DLLM\KLASS.pdf`，arXiv:2511.05664v2，Table 1 与 §6.1 / Appendix D.1；论文报告和公开仓库的复现入口已在上文分别注明。
