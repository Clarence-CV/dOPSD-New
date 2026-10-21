# LLaDA2-mini 下一版：实施交接简版

更新：2026-10-18。供同事 / Claude 开工前阅读。以下是已确定的实施方向，**mini适配、训练和官方MATH复现尚未完成**。

目标：先把mini基线跑对，再用更简单的采样训练，最后判断准确率、实际时延和训练成本。当前LLaDA Joint试验的主要信号是时延约减少11%、步数约减少18%；准确率34.4%→35.0%仅净增3/500题，尚不能认定稳定涨点。

## 0. 2026-10-19 实施状态（代码已写，未上 GPU）

- 用户决定：`max_new_tokens=2048`（已试过不截断）；官方MATH prompt/scorer缺项不再阻塞，暂用KLASS system prompt + KLASS判分，legacy判分并列保存。下文“不默认2048”“先完成官方复现再训练”两条按此修订。
- 已实现：mini backend（block-causal mask、MoE可微等价路径）、绝对block解码器（与官方HF generate在argmax修正下逐token一致）、`prefix_complete_eos`、阈值组规则、子划分、分片续跑评测、全层诊断与层规则、readout门槛/选点、root_success_v2训练；CPU测试与本地全流程模拟通过。
- 入口：`configs/selective_draft/llada2_mini_math.json`、`tacc/setup_llada2_env.sh`（登录节点一次）、`tacc/llada2_pipeline.slurm`（`STAGES`/`SMOKE`）、`tacc/sync_and_submit_llada2.sh`。说明见 `selective_draft/README.md` 末节。
- 未实现：prefix KV cache（W9）、真实mini消息路径干预审计（§11.3）、Math-Verify、mini-CAP/THU对照。

## 1. 下一版要解决什么

| 关心的问题 | 具体实现 |
| --- | --- |
| baseline是否对齐 | 固定**LLaDA2.0-mini**权重版本，先复现官方MATH，再加readout/injector。模型、题集、prompt、长度、停止、评分全部记录。 |
| block、阈值、steps | **block=32、confidence >0.95、评测T=0**；所有达标位置同时提交，没有则提交最高置信的一个。`steps=32`是每block计划参数，整题实际NFE另计。原LLaDA官方MATH的block64不能直接搬来。 |
| EOS、MASK、长度 | 只认生成区域已提交的EOS，且其左侧没有MASK才停；统一返回completion，排除prompt。长度先查官方配方，**不默认2048**；停止逻辑修补与原实现逐步对照。 |
| prompt、题集、评分 | 不沿用旧THU prompt冒充mini官方设置，不混淆MATH与MATH-500。核对分数/小数等价判分，保留原始评分与交叉审计，不选分高的scorer。 |
| 通信组为什么是4个 | **取消固定4个。** 早层draft中confidence>.95的位置全部进入G，少于2个不通信。G与最终提交集合阈值相同，但读出层不同，集合不保证相同。 |
| 选哪层、head训到哪里 | 不预设L16/L17。先测全层分布变尖、token身份稳定和高置信覆盖，按预定规则选层；head拟合末层分布，固定**64题验证＋64题独立确认**，按checkpoint交付。 |
| 消息有没有真正起作用 | 真实mini运行记录注入变化、下一层输入、后续Receiver logits、消息干预响应和梯度；特别修查**MoE eval分支的no_grad**，不能只看injector.grad非零。 |
| 训多少题、选哪个checkpoint | 6000题训练池先不放回，每25次有效更新保存；dev128粗筛、dev500选点。按贡献监督的独特题数、实际成本和验证曲线续训，**不把100步当收敛标准**。 |

## 2. 只保留一条主训练流程

```text
取一个新训练题
→ 当前Student从prompt完整生成，训练T=0.9
→ 最多8次（含首次），第一次答案正确即停；全错则换题
→ 从成功轨迹中均匀抽1个有同伴组的中间状态S
→ 取这条轨迹自身最终token在G上的完整组合，作为唯一参考pi
→ 检查与S已提交token、EOS和长度相容
→ LOO全员通过才训练；失败换题，不续写、不重抽、不搜索其他tuple
```

LOO检查：预测组中位置i时，只给冻结Teacher揭开其他组员的参考token，i自己仍为MASK；要求每个位置Teacher argmax都赞同pi_i。它检验局部一致性，答案正确性由独立verifier判断。

完整rollout用于验证答案并获得相容参考；回看保存状态不需要重算前缀。与旧方案相比，取消“每题抽4状态、每状态额外续写8次”和候选距离排序。新方案有成功筛选偏差，需明确记录；省下的预算优先覆盖未访问的新题，实际能多覆盖多少由pilot测量，不能直接承诺固定倍数。

## 3. 两阶段分工

- **阶段一只训练readout**：中间hidden → 小适配器 → 原final norm和共享LM head；拟合同状态原模型末层分布q0，不做LOO。关闭注入时，最终输出应与base一致。
- **阶段二只训练injector**：冻结主干与readout，用自然draft的完整概率分布形成soft embedding，注入后交给原生尾层。Teacher看参考同伴token；Student只看S和自己的draft，不能获得hard reference。用Teacher完整分布做forward KL。
- **主干冻结不等于模型不能学习**：injector在更新，所以“采样找到正确路径，再学进去”完全可能。首轮不增加复杂分支；保存数据，后续需要归因时，再比较同数据的LOO-KL与直接参考token CE，并做同伴消息干预。

## 4. 开工顺序与当前缺项

1. **官方基线**：核对mini的T0采样问题、原生block-causal attention、EOS/返回范围；先做无指导parity与完整MATH复现。
2. **工程与head**：验证冻结尾层梯度、真实消息路径，再做全层诊断、readout训练及独立确认。
3. **小规模采样pilot**：固定.95与root最大8，测答案成功率、LOO接受率、贡献题/GPU小时及显存。
4. **一条训练链**：逐段50→100→200→400作为待评审上限，不强制跑满；固定dev选点后才做最终测试，保存实际成本和覆盖曲线。

**当前前置缺项：官方MATH 93.22%对应的完整prompt、生成长度、scorer、题集细节和实际NFE参考仍未找齐。** 已知block32/.95/T0，不等于已经具备完整复现条件；未知字段保持未解析，不能用自选配方代替后宣布复现。

首轮不同时做阈值网格、多层候选训练、Partial Loop、Receiver新损失或额外续写。基线与Joint必须匹配prompt、长度、停止、评分、精度和推理backend；实际时延包含readout、消息与注入成本。

## 5. 详细资料

- [实施规格与官方来源](llada2-mini_engineering_plan.md)：遇到具体配置、接口和验收问题时查阅。
- [跨仓库解码参数审计](decoding_protocol_audit_zh.md)：THU、KLASS、LLaDA/iLLaDA、Fast-dLLM差异。
- [研究概览](Selective_Draft_OPSD.md)：方法演进、证据与数学边界。
- [官方mini模型页](https://huggingface.co/inclusionAI/LLaDA2.0-mini)；[模型论文](https://arxiv.org/abs/2512.15745)；[THU root重试源码](https://github.com/xingzhejun/d-opsd-code/blob/70d146c525c79c4c6d056bcd69853f1365ce5d8c/d-opsd/d_opsd_trainer.py#L479)。
