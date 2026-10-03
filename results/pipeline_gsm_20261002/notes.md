@@ overview
> 对应讨论文档：`dllm_opsd_parallel_reasoning_discussion_summary.md`（§13 的现象探针、§7/§8/§11 的问题）。本文件是唯一的结果文档；所有数字都由 `analysis/make_summary.py` 从原始 record 直接计算（同目录 `metrics.csv` 为全部数字），本节以下的“解读”为 Claude 的阅读。

### 一句话结论

在 THU d-OPSD（固定预算，每步并行提交 2 个 token）下：**同一步并行提交的 token 之间确实存在比“先后提交”更大的相互影响，而且以冲突为主**；THU 的 future PI **确实缩小了这一 gap（|D_T| < |D_S|，83% 的 token）**，但 PI 对任何额外上下文（匹配的非同步 token）的作用也差不多同样缩小，**sibling 特有的额外缓解很小**（+0.010 nat），且 PI 的变化方向更像“补充未来上下文”而不是“补充兄弟的取值”。PI 在冲突位置更倾向“暂缓 / 改向”、并和 student 的解码顺序有明显分歧，这与讨论文档 §8 一致。换成置信度阈值解码后，同一步 gap 小一个数量级，说明这一现象主要由**固定配额强制并行提交**引起。训练后 student 自身的 sibling gap 降低约 1/3，但对一般上下文的敏感度也同步下降。

### 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | LLaDA-8B-Instruct（4-bit + LoRA r=128），THU d-OPSD 训练（GSM8K train，3×A100，BATCH_DIVIDE=8，1344 步） |
| 选 checkpoint | 300 道 seeded GSM8K test 题，greedy + 固定预算 128 步（论文协议）→ step-448 |
| 分析的 4 组 | {base, step-448} × {固定预算 128 步（每步 2 token）, 阈值 0.9（每步揭开所有置信度 >0.9 的位置，至少 1 个）}；均为 T=0.9、生成 256 token、block 32，同一批 300 题 |
| teacher | **base 权重** + PI（与训练时的 fixed teacher 一致）；PI = 从下一个 block 起随机 25% 位置填入最终 rollout 的 token（PI 永远不覆盖当前 block，即不泄露 C_t 本身） |
| 重算精度 | fp16（与采样器一致） |
| 统计 | 排除 EOS 与答案结束之后的 token；兄弟 / 对照 / 协调类指标只用 \|C_t\| ≥ 2 的步；95% CI 为按 rollout 的集群 bootstrap |

**读表必看**：step-448 两列中 teacher（base+PI）与 student（训练后）权重不同，凡是含 p_T 的指标（g、D_T、coord、PI 作用分类、顺序、方向）都混入了“base 与训练后模型的差异”，**只有 base 两列反映纯 PI 效应**。只含 p_S 的指标（D_S、|D_S|、JS_sib、D_S,ctrl 等）在所有列都可比。

@@ accuracy
- greedy 扫描：step-448 最好（+4.4 pt），与论文“第 425 次更新最佳”一致；超过约 600 步后准确率崩溃（step-1344 只剩 18%），很可能是论文只评到第 500 步的原因。
- T=0.9：固定预算下训练提升 +7.0 pt（75.0 → 82.0）；阈值 0.9 下几乎无提升（79.3 → 79.7），且阈值解码本身就让 base 从 75.0 升到 79.3。

@@ pi_effect
**解读**（看 base 两列）：PI 对 y 主要是“强化”（固定预算 85%、阈值 92%），平均 g 很小（+0.02 ～ +0.05 nat），并使分布更尖（H 降 0.18 ～ 0.21）。“student 支持一般、PI 明显拉高”的 token 只占 1.4% ～ 4%，且 wrong 组反而更高——说明 g 本身不能当作“PI 有用”的指标（呼应讨论文档 §8）。

**step-448 两列**：g < 0、“暂缓”约 47% ～ 63%，主要反映训练后 student 比 base teacher 更自信，**不是** PI 的作用。

@@ sibling_gap
**解读**：
- **固定预算下 gap 很大且以冲突为主**：base 的 |D_S| = 0.211（wrong 0.264），带符号均值为负（−0.130），约 26% 的 token 被兄弟“否定”（D_S < 0）。答错的 rollout gap 更大。
- 冲突在解码早期最强（block 0–1：D_S = −0.181），后期减弱（block 5–6：−0.077）。
- **训练后 student 自身的 gap 变小**：|D_S| 0.211 → 0.140（−34%），wrong 0.264 → 0.173；JS_sib 0.020 → 0.011。这一行只涉及 student，结论有效；但需结合第 4 节（对一般上下文的敏感度也同步下降）。
- **阈值解码下 gap 小一个数量级**：|D_S| ≈ 0.02，且 D_S > 0（兄弟以支持为主），D_S < 0 的比例仅约 7%。

@@ teacher_gap
**解读**（这是讨论文档 13.4 的核心检验，看 base 两列）：
- **|D_T| < |D_S| 成立**：固定预算下 |D_S| − |D_T| = +0.068（约缩小 32%），83% 的 token 满足 |D_T| < |D_S|；wrong 组同样（+0.080，80%）。阈值解码下也成立（+0.014，92%）。
- 原定义的带符号 coord = D_S − D_T 为负（−0.034），因为 D_S 本身为负（冲突）；冲突情形下应看绝对值。
- **对错差异不明显**：PI 对 correct 与 wrong 的缓解幅度相近，未见“成功 future 特有”的效应。
- step-448 两列含权重差异，不作解读。

@@ control
**解读**（13.6：排除“多给任何 context 都会改变分布”）：
- **同一步兄弟确实特殊**：base 固定预算下，揭开兄弟使 y 的 log-prob 平均下降，而揭开一个匹配的、之后才提交的 token 使其上升（D_S − D_S,ctrl = −0.117，wrong −0.153）；按绝对值，兄弟引起的改变也更大（|D_S| − |D_S,ctrl| = +0.059）。注意：对照 token 是在 y_i 确定之后才解码的，所以这一对比本质上是“同时提交 vs 先后提交”。
- 有意思的细节：按整个分布（JS）看，兄弟引起的变化反而略小于对照（JS_sib − JS_ctrl = −0.006）——兄弟主要是**针对 y_i 本身**降低其概率，而非整体改写分布。
- **PI 对兄弟 gap 的“额外”缓解很小**：PI 也把对照 gap 缩小了 0.045（约 38%），扣除后 sibling 特有的额外缓解只有 +0.010（wrong +0.018），虽显著但小；按相对比例，兄弟（约 32%）反而不比对照（约 38%）多。
- **阈值解码下兄弟不再特殊**：|D_S| − |D_S,ctrl| = −0.002。但阈值解码下对照匹配质量明显变差（只有约 55% 的 token 能配到对照；距离差 4.4、信心差 0.54，固定预算下为 1.8、0.13），这一行结论较弱。

@@ cogain
**解读**：在所有 4 组、correct 和 wrong 两组中，同一步 token 对“都被 PI 拉高”的比例都**显著高于距离匹配的非同步 token 对**（base 固定预算 +2.9 pt，阈值 +7.8 pt；p = 0.002 为 500 次置换的下限）。即 PI 的作用确实带有 step-level 的集体结构，不只是独立的 token sharpening。阈值解码下 correct 组（+7.8）高于 wrong 组（+4.7），但 CI 有重叠。

@@ decisions
**解读**（讨论文档 §8：好的 PI 也可能让模型“先别确定”）：
- **PI 在兄弟冲突时更倾向暂缓 / 改向**：base 固定预算下，D_S < 0 时暂缓或改向占 19.6%，D_S ≥ 0 时为 13.3%（wrong：24.8% vs 16.8%）；阈值下 11.8% vs 6.1%。
- **teacher 与 student 的解码顺序有明显分歧**：teacher 最想揭开的位置只有约 54%（阈值约 51%）与 student 实际揭开的一致；约 20%（阈值约 27%）的已提交 token 在 teacher 看来是“过早提交”，冲突时更高（24% vs 18%）。block 内置信度排序的 Spearman 只有 0.37（阈值 0.22）。
- 这支持“PI 的一部分作用体现在**决定哪些位置该先定**”，与讨论文档 §8、§9 的“parallel decision quality”一致。

@@ direction
**解读**：base 固定预算下，PI 引起的分布变化与“看到兄弟”引起的变化有 67% 同方向，但与“看到对照 token”的同方向比例更高（74%），配对 cos 差为 −0.13。即 **PI 的作用方向更像“补充未来上下文”，而不是“补充同步兄弟的取值”**。这对“successful future 隐式恢复了 intra-step coordination”的假设是一个反面证据：PI 缓解兄弟 gap，更可能是因为它提供了让兄弟信息变得不那么关键的未来上下文，而不是专门编码了兄弟之间的依赖。

@@ threshold
**解读**：
- base 在阈值解码下，|D_S| 约 0.020～0.024，**不随 |C_t| 增大**（2 个与 9 个以上几乎相同）——高置信度 token 一起提交时彼此基本协调，符合阈值解码的前提。（更正：此前的快速汇总称“D_S 随 |C_t| 增大”，那是把 |C_t|=1 的零值混入平均造成的稀释。）
- step-448 在阈值解码下一步揭开的 token 更多（训练后更自信），|C_t| 为 3–4 时 |D_S| 升到约 0.06，D_S 转为负——训练后的模型在阈值下会把一些并不协调的 token 一起提交，这可能是阈值解码下训练没有带来准确率提升的原因之一（推测，未验证）。

@@ discussion
## 9. 对讨论文档各条判断的影响

| 讨论文档 | 本次数据 | 判断 |
|---|---|---|
| §4 同一步 marginal 各自合理但组合不协调 | 固定预算下同步兄弟使 y 的概率显著下降，且比先后提交的 token 影响更大；答错的 rollout 更严重 | **支持**（以冲突形式出现） |
| 13.4 PI 已隐式补了一部分 coordination（\|D_T\| < \|D_S\|） | base 固定预算下 \|D_T\| 比 \|D_S\| 小约 32%，83% 的 token 成立 | **支持**，但特异性弱：PI 对一般上下文的作用也缩小约 38%，方向更像未来上下文 |
| 13.5 PI 有 step-level 集体结构 | 距离匹配后 co-gain 仍显著为正（+3 ～ +8 pt） | **支持** |
| §8 PI 不只是提高概率，可能让模型暂缓 | 冲突处暂缓 / 改向更多；teacher 与 student 顺序重合仅约 54% | **支持** |
| “successful future 特有”（correct > wrong） | 大部分指标 correct 与 wrong 相近 | **未见证据** |
| §7 / §11 PI 能否在 inference 被内化 | 训练后 student 自身 \|D_S\| 下降 34%，但对对照的敏感度也下降 42%，且阈值解码下无准确率提升 | **部分支持，混有整体 sharpening**；需 `--teacher self` 和更细的对照才能分开 |
| 并行冲突的来源 | 阈值解码下 gap 小 10 倍且不随 \|C_t\| 增长 | 冲突主要来自**固定配额强制提交** |

对研究主线（§9、§15）的含义：
1. “parallel decision quality ≠ token correctness”在固定配额解码下有实证支撑：冲突集中在被配额强行同时提交的 token 上。
2. THU 的 future PI 能缓解冲突、能把冲突 token 往后推，但它更像“给更多上下文”，而不是学到“同一步该如何协调”；这给“专门针对 parallel decision 的 PI / supervision”留出了空间。
3. 如果以阈值解码为默认推理方式，同一步 gap 已经很小，coordination 的收益空间可能有限；问题更突出于固定步数 / 高并行度（少步数）设置。

@@ caveats
## 10. 局限与注意事项

1. **teacher 权重**：step-448 两列 teacher = base + PI，含权重差异；要测训练后模型上的纯 PI 效应，需用 `--teacher self` 重跑（每组约 6.5 SU）。
2. **对照的含义**：匹配对照是“之后才提交”的 token，它在生成时已经看到了 y_i，所以 sibling vs control 测的是“同时提交 vs 先后提交”，不等于“任意多一个 token 的上下文”。
3. 阈值解码下对照匹配率低（38%～59%）、匹配质量较差，相关结论较弱。
4. 单一随机种子、300 题、仅 GSM8K、T = 0.9；PI 每步只采样 1 次（25% 随机位置）。
5. “correct” 是整条 rollout 答对，单个 token 不一定最优；wrong 样本 54～75 条。
6. 生成长度 256 截断，与论文一致。

@@ files
## 11. 文件

本目录（进 git）：
- `SUMMARY.md`：本文件
- `metrics.csv`：本文所有数字（section、metric_id、label、run、group、n_rollouts、n_values、value、ci_lo、ci_hi、note）
- `rollouts.csv`：每条 rollout 一行（4 组）：对错、题目、标准答案、预测、完整生成文本、各指标均值
- `sweep.csv`：checkpoint 扫描
- `figs/<run>/*.png`：每组的图（run = fixed_base / fixed_best / thr_base / thr_best）
- `notes.md`：本文“解读”部分的源文本

`outputs/pipeline_gsm_20261002/tables/`（体积大，不进 git）：
- `tokens.csv.gz`：每个揭开的 token 一行（约 27 万行），全部逐 token 指标与过滤标记
- `steps.csv.gz`：每个解码步一行
- `topk.csv.gz`：每个 token 的 p_S、p_T、p_sib 的 top-20（已解码为文字，JSON 格式 `[[token, prob], ...]`），以及 y 在三种分布下的概率

主要列含义见 `tokens.csv.gz` 表头；指标定义见各节开头的“定义 / 读法”表。
