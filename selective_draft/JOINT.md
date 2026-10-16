# Joint-reference OPSD

分支 `joint`。在原有 `selective_draft` 流水线上增加共同参考训练；真实 LLaDA 训练/评测在 TACC 计算节点执行。

## 模型与流程

- Teacher：冻结的原始 LLaDA-8B-Instruct，不启用消息模块。
- Student：同一份冻结主干 + 冻结的已有 L28 readout + 可训练的共享消息模块。
- 每步在当前 block 的 MASK 中按 L28 最大概率选最多 4 个位置，平局按位置从小到大。至少两个位置才通信，**不使用旧单源 0.8 消息门槛**。
- 每个位置把自己的完整 soft draft 转成 embedding 加权和，经过共享 injector，同时加到该位置的 L28 residual。L29–32 原有双向 attention 负责交换信息；组员不提前锁定。
- 最终仍在当前 block 的所有 MASK 位置上解码：confidence 严格大于 0.9 的候选提交；没有达标位置则提交最高置信的一个。推理时 greedy，生成长度 256、block 32、按 EOS 所在 block 停止。

训练每次更新在每个进程上取 1 道训练题，用当前 Student、T=0.9 生成自然轨迹，从中随机取最多 4 个可通信状态。原轨迹无论最终答对与否，都可提供状态。

对每个状态，从**完全相同的已提交 token 和 MASK 状态**重新续写 8 次，温度 0.9。只保留答案验证通过、保持已知 token、已完整解码到停止位置的 continuation。取它们在通信组上的整体 tuple，去重后按与当前 soft-message embedding 的平方距离排序，平局按 tuple 字典序。

依次认证候选：预测位置 i 时，只给 Teacher 揭开同组其他位置的候选 token，i 自身和组外 MASK 保持不变。只有所有 Teacher 首选都与候选一致，才选作共同参考；找到首个通过者即停止认证。所有候选都失败则跳过该状态。

Student 始终只看原始 MASK 输入和自己的自然 soft draft，训练目标是组内平均的完整词表 `KL(q_teacher || p_student)`。硬参考只用于构造 Teacher PI，不输入 Student。每次更新刷新自然状态；候选库在该状态的一次训练中固定，失败候选与认证记录也保存。

这里的 **8 次预算用于每个选中状态的候选 continuation**，不是先从题目重试 8 次再对每条结果继续做 8 次。与旧流水线“首次答对即停止”不同，这里保留全部 8 次结果，以便在已发现的多个整体模式中选择参考。

## 配置

默认配置：[`configs/selective_draft/llada_joint_math.json`](../configs/selective_draft/llada_joint_math.json)。沿用前次 MATH 的数据源、split seed、prompt 和答案评分；训练集 6000，dev/analysis 各 500，独立 MATH-500 test 来源保持不变。

| 参数 | 默认值 | 来源 |
| --- | --- | --- |
| 候选次数、温度 | 8、0.9 | THU / 现有 pass@8 设置 |
| 生成长度、block | 256、32 | THU / 现有设置 |
| 解码阈值、评测温度 | 0.9、0 | 本项目当前阈值协议 |
| 组大小、读出层 | 4、L28 | 本次方案 |
| 模块学习率 | 1e-4 | 新模块初试；比上一版 3e-4 保守 |
| Adam betas | (0.9, 0.99) | `d-opsd/opsd.yaml` |
| weight decay、grad clip | 0.1、0.2 | `d-opsd/opsd.yaml` |
| 调度 | constant_with_warmup，ratio=0.0001 | `d-opsd/opsd.yaml`；100 steps 时为 1 个 warmup step |
| 每进程题数 / 每题状态数 | 1 / 最多 4 | 每卡一题，复用当前状态抽样规模 |
| 初试更新数、保存间隔 | 100、25 | 小规模 pilot / THU 保存间隔 |

完整词表 forward KL、冻结主干、不使用 LoRA、sibling PI 都是本次方法定义，因而不照搬 THU 的 top-k loss、reverse KL 或 future retain ratio。`microbatch_states`、`filter_correct_only` 属于旧训练路径；joint 路径固定逐状态反传、只用验证通过的候选作目标。

## 运行

复用原来的 **stage-readout snapshot**，不要用旧单 Anchor 的 stage-guidance snapshot。模型 revision 必须与 readout 一致。已有 manifest 可直接复用，保持题目划分相同。

```bash
# 在 TACC 计算节点，先 source tacc/env.sh。以下路径按实际产物设置。
python -m selective_draft.cli train-joint \
  --config configs/selective_draft/llada_joint_math.json \
  --set data.split_manifest=/path/to/split_manifest.json \
  --readout-snapshot /path/to/readout/snapshot \
  --max-steps 100 --save-every 25 --output-dir /path/to/joint/train

# 精确恢复：相同数据、配置和进程数；不再同时传入 readout snapshot。
python -m selective_draft.cli train-joint \
  --config configs/selective_draft/llada_joint_math.json \
  --set data.split_manifest=/path/to/split_manifest.json \
  --resume /path/to/joint/train/checkpoint-step-25 \
  --max-steps 100 --output-dir /path/to/joint/train
```

批作业入口：[`tacc/joint.slurm`](../tacc/joint.slurm)。默认一个 gpu-a100 节点、3 个进程、6.5 小时 wall limit；提交前复核费率并报告预算。脚本按 readout 的 manifest 设置模型 revision，训练后在 MATH-500 test 全部 500 题上并行（每卡一个）评测：原始 LLaDA（`off`）、`group_soft`、只读出不注入的 `group_readout`。

初次只做少量更新，示例：

```bash
mkdir -p logs
READOUT_SNAPSHOT=/path/to/readout/snapshot \
SPLIT_MANIFEST=/path/to/split_manifest.json \
RUN=joint_smoke STEPS=3 N_TRAIN=8 N_EVAL=4 NPROC=1 \
sbatch --partition=gpu-a100-small --time=00:30:00 tacc/joint.slurm
```

修改 `STEPS`、`NPROC`、`LR`、`N_TRAIN`、`N_EVAL`、`EVAL_SPLIT` 可覆盖脚本设置；断点恢复用 `RESUME`（已有 `summary.json` 的评测会跳过）。

## 输出与验证

- `train/log_rank*.jsonl`：loss、完整组 KL 总和、认证成功数 / 状态数、候选数量、Teacher 查询数、参考一致率、高置信偏离、注入幅度；G 外待填位置相对原模型的 KL 与 0.9 提交决定翻转数（`outside_kl` / `outside_commit_flips`，无 loss 覆盖，只作监测）。
- `train/rollouts/`：自然轨迹，含每步通信组，保留正确和错误结果。
- `train/references/`：每个状态的 8 条完整候选、soft 消息、draft top-k、认证检查和选中的整体参考。参考状态编号与 RNG 一同恢复。
- `train/checkpoint-step-*/`：模块权重、优化器、每个进程的采样顺序及 RNG；`train/snapshot/` 用于推理。
- `eval/*/summary.json`：准确率、实际耗时、步数、通信组提交比例与 fallback 数量。

本地只运行不加载预训练权重的 CPU 测试：

```bash
OMP_NUM_THREADS=1 python tests/test_selective_draft_joint.py
```

覆盖：零注入 parity、同伴消息传播与梯度、真实状态续写、目标位置不泄漏、完整 tuple 选择、8 次候选预算、冻结参数、无参考时跳过、checkpoint 精确恢复。
