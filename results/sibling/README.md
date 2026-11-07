# Sibling OPSD — 实验索引

分支 `sibling-opsd`，LLaDA2.0-mini，Vista GH200。全局账本（含每个作业的 SU）见 `../EXPERIMENTS.md`；本文件只列这条线的作业和对应的结果文档。

方法文档（`doc/`）：
- `doc/plan_a_loo_defer.md`：Sibling OPSD 原始设计（LOO teacher + 截顶目标）。§3/§5/§6 仍写的是 τ=0.7、α=0.4，已过时；以 `main_t05_v1.md` 第 1 节的设置为准。
- `doc/plan_path_opsd.md`：下一版 Path OPSD 方案草稿（未实现）。

| 日期 | 作业 (Vista) | 内容 | SU | 结果 |
|---|---|---|---|---|
| 2026-11-03 | 1118187 / 1118193 / 1118195 / 1118882 | dev 调试（流水线、每 GPU 2 进程、τ=0.5 截顶目标） | 0.25 / 0.25 / 0.77 / 0.28 | `../EXPERIMENTS.md` 对应行 |
| 2026-11-03 | 1118270 | **Phase 0**：冻结原始模型，64 题，贪心 τ=0.95/0.7/0.5，LOO 冲突统计 | 1.38 | [`p0_frozen_base.md`](p0_frozen_base.md) |
| 2026-11-04 | 1120791 | MASK sink 探针：LoRA 能否在 defer 行学会预测 MASK | 0.84 | 学不会，此路已放弃；见 `../EXPERIMENTS.md` 对应行 |
| 2026-11-05 | 1118947 | **主训练** sib_t05_v1：τ=0.5，截顶 LOO 目标，500 次更新 | 25.43 | [`main_t05_v1.md`](main_t05_v1.md) |
| 2026-11-06 | 1127239 | 阈值扫描：MATH-500 上原始模型和 step 375 在 τ=0.9/0.8/0.7 | 估 9–10，排队中 | 完成后补到 `main_t05_v1.md` 第 6 节 |

原始输出：Vista `/scratch/11277/yilunai0430/dopsd/sibling/<RUN>/`；已拉回本地的在 `outputs/<RUN>/`（不含 checkpoint）。
