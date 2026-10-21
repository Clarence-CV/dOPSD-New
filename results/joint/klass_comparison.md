# Joint 与 KLASS：数字和评测口径

2026-10-17。阅读用户提供的 `E:\MSIS\DLLM\KLASS.pdf`（arXiv:2511.05664v2），并核对[官方代码](https://github.com/shkim0116/KLASS)。本次仅读取论文、代码和本地输出，并做 CPU 答案重评分，未运行模型。

实际训练和rollout见[Joint分析](math_v1_analysis.md)，评测及上游对照见[解码参数审计](../../doc/decoding_protocol_audit_zh.md)。注意：LLaDA 官方 EVAL.md 中 31.4% 对应 **Base、长度/步数/block 均为 1024**；下面 KLASS 的 31.4% 对应 **Instruct、256/256/64**。两者只是数字相同，不能视为同一实验。

## 1. 论文数字

论文第 6 页 Table 1，均为 LLaDA-8B-Instruct、MATH-500：

| 方法 | 准确率 | 平均步数 |
| --- | ---: | ---: |
| 论文 Top-1 | 31.4% | 256 |
| 论文 confidence > 0.9 | 31.6% | 96.46 |
| 论文 KLASS | 33.8% | 128.62 |
| 我们的 base，原评分 | 34.4% | 100.42 |
| 我们的 Joint，原评分 | 35.0% | 82.23 |
| 我们的 base，换 KLASS 评分 | 35.2% | 100.42 |
| 我们的 Joint，换 KLASS 评分 | 36.6% | 82.23 |

重评分只改变答案抽取和匹配，不改变输出、步数，也不替换原报告。base 新增判对 4 题，Joint 新增 8 题；没有原判对改判错。配对细目与评分源码版本见 [klass_rescore.json](klass_rescore.json)。

## 2. 已确认的协议差异

- **题目相同。** 官方仓库 `data/math_test.json` 实为 JSONL，与 HuggingFaceH4/MATH-500 的 500 道题逐题、同序一致；本地记录的 gold 也全部与该数据集 answer 字段一致。
- **block 不同。** KLASS 使用 64，我们使用 32。这会改变每一步的状态、提交候选和后续预测，准确率与步数都会受影响。
- **prompt 不同。** KLASS 使用独立 system 消息，要求以 `The final answer is` 作答；我们在 user 消息中要求 reasoning/answer 标签与 boxed 答案，并在生成前追加 reasoning 标签。
- **评分不同。** 我们主要使用 boxed 抽取和规范化字符串匹配；KLASS 使用更多抽取、归一化及分数转小数处理，能接受部分我们误判的等值答案。只统一评分后，我们 base 仍有 35.2%，高于其 confidence baseline 的 31.6%；因此差异不只来自评分。
- **停止不同。** KLASS 官方解码器处理完整的 256 位置，没有 EOS 提前停止；我们在完成 block 后检查 EOS，可提前退出。我们的有效长度约 244 token，单靠这个数不能量化提前停止贡献了多少步。
- **解码策略不同。** Top-1 的 256 步是固定每步填一个；confidence 的步数自适应。KLASS 又要求跨步分布稳定：MATH 使用 confidence > 0.6、KL < 0.01、历史长度 2，所以它比 confidence-only 的 96.46 步更多并不矛盾。
- **硬件不同。** KLASS 附录 D.1 使用 RTX A5000，我们使用 A100；绝对秒数不能直接作方法速度比较。Joint 还增加了 readout/注入开销，步数下降不等于同比例时延下降。

这些是已查到的差异，尚未通过逐项控制实验确定各自贡献。

## 3. 当前能得出的判断

**我们的现有准确率和步数在数字上优于论文里的 KLASS，但尚不能认定方法已经胜过 KLASS。** 我们未经 Joint 训练的 base 也已高于论文数字，说明直接横比混入了协议影响。

自身对照仍有正向信号：原评分为 +0.6 个百分点；KLASS 评分下为 +1.4 个百分点（176/500 → 183/500）；输出不变，所以实际耗时下降 11% 的原有测量不受重评分影响。KLASS 在其协议下相对 confidence-only 增加 2.2 个百分点，同时增加约 33% 步数；这些增量也不能跨协议直接定优劣。

最直接的下一步是在我们相同 prompt、block、停止、评分和硬件下，加一个冻结原模型的 KLASS 解码对照。无需再训练，先用 dev 小集核对实现和参数，再做统一评测。尚未执行。
