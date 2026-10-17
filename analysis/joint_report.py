"""Detailed tables for a joint-reference OPSD run: training trajectory, rollout/continuation cost, certification,
and paired evaluation. Usage: python analysis/joint_report.py RUN_DIR [LEVELS_JSON] > report.md
LEVELS_JSON (optional): {"test-0": level, ...} for MATH-500 difficulty breakdowns."""

import collections
import glob
import json
import math
import os
import statistics as st
import sys

import torch

HIDDEN_NORM_L28 = 265.0  # mean L28 hidden norm at the source position, measured in math_prelim_v3 (same model/layer)


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.mean(v) if v else float("nan")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n) if n else 1.0


def main(run, levels_path=None):
    tr = os.path.join(run, "train")
    logs = [json.loads(l) for r in sorted(glob.glob(f"{tr}/log_rank*.jsonl")) for l in open(r)]
    rolls = [x for f in sorted(glob.glob(f"{tr}/rollouts/*.pt"))
             for x in torch.load(f, weights_only=False)["rollouts"]]
    refs = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(f"{tr}/references/*.pt"))]
    B = 10
    bins = range(0, 100, B)

    # ---------------------------------------------------------------- 1. training trajectory
    print("## 1. 训练过程（每 10 次更新一段；3 卡合计）\n")
    print("rollout 为当前 Student 以 T=0.9 生成的 on-policy 轨迹（每段约 30 道不同训练题，噪声大，只看趋势）。"
          "s/update 为 rank0 日志相邻记录的时间差，含被跳过的更新。\n")
    r0 = sorted([r for r in logs if r["rank"] == 0], key=lambda r: r["time"])
    dt = {}
    for a, b in zip(r0, r0[1:]):
        dt.setdefault(b["optimizer_step"] if "skipped" in b else b["optimizer_step"] - 1, []).append(b["time"] - a["time"])
    rows = []
    for lo in bins:
        inb = lambda s: lo <= s < lo + B
        L = [r for r in logs if inb(r["optimizer_step"] - (0 if "skipped" in r else 1))]
        U = [r for r in L if r.get("local_states", 0) > 0]
        R = [x for x in rolls if inb(x["meta"]["policy_version"])]
        skipped = sum("skipped" in r for r in L if r["rank"] == 0)
        secs = [t for s, v in dt.items() if inb(s) for t in v]
        dn = mean([r.get("delta_norm") for r in U])
        rows.append([f"{lo}-{lo + B - 1}", skipped, f"{mean(secs):.0f}",
                     f"{sum(x['is_correct'] for x in R)}/{len(R)}", mean([x["stats"]["nfe"] for x in R]),
                     mean([r["loss"] for r in U]), mean([r.get("kl_sum") for r in U]),
                     mean([r.get("reference_agreement") for r in U]), mean([r.get("high_conf_count") for r in U]),
                     mean([r.get("high_conf_wrong") for r in U]),
                     f"{dn:.1f} ({dn / HIDDEN_NORM_L28:.0%})", mean([r.get("outside_kl") for r in U]),
                     mean([r.get("outside_commit_flips") for r in U]), mean([r.get("grad_norm") for r in U])])
    print(table(["更新", "跳过", "s/update", "rollout 答对", "rollout 步数", "loss(组均KL)", "组KL和", "参考一致率",
                 "高置信数/组", "高置信偏离/组", "注入幅度(占hidden)", "G外KL", "G外翻转/状态", "grad_norm"], rows))
    print("loss = 组内 4 个位置 KL(q_teacher‖p_student) 的平均；参考一致率 = Student argmax 等于共同参考的比例；"
          "高置信 = 概率 >0.9；注入幅度占比按 L28 hidden 范数约 265（v3 实测）估算。grad_norm 为裁剪前（clip=0.2）。\n")

    # ---------------------------------------------------------------- 2. coverage / certification over training
    print("## 2. 参考覆盖与认证（按状态所在的策略版本分段）\n")
    rows = []
    for lo in bins:
        D = [d for d in refs if lo <= d["policy_version"] < lo + B]
        C = [c for d in D for c in d["candidates"]]
        anyc = [d for d in D if d["stats"]["correct_candidates"] > 0]
        cert = [d for d in D if d["selected_tokens"] is not None]
        rows.append([f"{lo}-{lo + B - 1}", len(D), sum(c["is_correct"] for c in C) / max(1, len(C)),
                     len(anyc) / max(1, len(D)), len(cert) / max(1, len(D)), len(cert) / max(1, len(anyc)),
                     mean([d["stats"]["unique_candidates"] for d in anyc]),
                     mean([c["stats"]["nfe"] for c in C]), mean([d["stats"]["teacher_forwards"] for d in D])])
    print(table(["更新", "状态数", "续写答对率", "≥1 条答对", "认证通过/全部", "认证通过/有答对",
                 "去重 tuple 数", "续写步数", "Teacher 前向/状态"], rows))

    # by position of the state in the trajectory
    print("按状态在轨迹中的位置（所在 block）：越靠后，剩余要生成的越少、续写越容易答对。\n")
    rows = []
    for blk in range(8):
        D = [d for d in refs if d["state"]["block_id"] == blk]
        if not D:
            continue
        C = [c for d in D for c in d["candidates"]]
        rows.append([blk, len(D), sum(c["is_correct"] for c in C) / max(1, len(C)),
                     sum(d["stats"]["correct_candidates"] > 0 for d in D) / len(D),
                     sum(d["selected_tokens"] is not None for d in D) / len(D), mean([c["stats"]["nfe"] for c in C])])
    print(table(["block", "状态数", "续写答对率", "≥1 条答对", "认证通过", "续写步数"], rows))

    # per-problem: is coverage a property of the problem?
    by_prob = collections.defaultdict(list)
    for d in refs:
        by_prob[(d["meta"]["example_id"], d["policy_version"])].append(d)
    per = [sum(x["selected_tokens"] is not None for x in v) / len(v) for v in by_prob.values()]
    hist = collections.Counter(round(p * 4) / 4 for p in per)
    print(f"每道训练题（{len(per)} 题次）内认证通过比例的分布：" +
          "，".join(f"{k:.2f}: {hist[k]}" for k in sorted(hist)) + "。两头集中说明覆盖主要由题目难度决定。\n")

    # certification by rank inside G
    print("组内按 L28 置信度排序的位置（0 = 最自信）：认证检查结果。\n")
    rank_tot, rank_fail, rank_prob = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for d in refs:
        for a in d["certification"]:
            for k, (t, tok, p) in enumerate(zip(a["top1"], a["tokens"], a["reference_probs"])):
                rank_tot[k] += 1
                rank_fail[k] += t != tok
                rank_prob[k].append(p)
    print(table(["组内位次", "检查次数", "Teacher top1≠候选", "候选 token 的 Teacher 概率中位数", "概率 >0.9 的比例"],
                [[k, rank_tot[k], rank_fail[k] / rank_tot[k], st.median(rank_prob[k]),
                  sum(p > 0.9 for p in rank_prob[k]) / len(rank_prob[k])] for k in sorted(rank_tot)]))

    # ---------------------------------------------------------------- 3. compute budget
    print("## 3. 算力去向（训练阶段，按前向次数估算）\n")
    nfe_roll = sum(x["stats"]["nfe"] for x in rolls)
    nfe_cont = sum(c["stats"]["nfe"] for d in refs for c in d["candidates"])
    nfe_cont_waste = sum(c["stats"]["nfe"] for d in refs if d["selected_tokens"] is None for c in d["candidates"])
    nfe_teacher = sum(d["stats"]["teacher_forwards"] for d in refs)
    nfe_student = sum(1 for d in refs if d["selected_tokens"] is not None)
    tot = nfe_roll + nfe_cont + nfe_teacher + 2 * nfe_student
    print(table(["部分", "前向次数", "占比"], [
        ["on-policy rollout", nfe_roll, nfe_roll / tot],
        ["同状态续写（8 条/状态）", nfe_cont, nfe_cont / tot],
        ["  其中：最终没拿到参考的状态", nfe_cont_waste, nfe_cont_waste / tot],
        ["Teacher 认证", nfe_teacher, nfe_teacher / tot],
        ["Student 前向+反向（按 2 次计）", 2 * nfe_student, 2 * nfe_student / tot]]))

    # ---------------------------------------------------------------- 4. evaluation
    ev = os.path.join(run, "eval")
    R = {m: {r["example_id"]: r for r in map(json.loads, open(f"{ev}/{m}/rows_thr0.9.jsonl"))}
         for m in ("base", "joint", "readout_only") if os.path.exists(f"{ev}/{m}/rows_thr0.9.jsonl")}
    b, j = R["base"], R["joint"]
    ids = list(b)
    print("## 4. 评测（MATH-500 test，greedy，阈值 0.9，batch 1）\n")
    rows = []
    for m, X in R.items():
        V = list(X.values())
        rows.append([m, mean([r["is_correct"] for r in V]), mean([r["row_steps"] for r in V]),
                     mean([r["effective_tokens"] / r["row_steps"] for r in V]), mean([r["latency_s"] for r in V]),
                     q([r["latency_s"] for r in V], 0.5), q([r["latency_s"] for r in V], 0.9),
                     mean([r["fallback_rows"] for r in V]), mean([r["effective_tokens"] for r in V]),
                     sum(r["stop_reason"] != "eos" for r in V)])
    print(table(["模型", "准确率", "步数", "token/步", "平均延迟 s", "中位", "P90", "fallback 步/题",
                 "有效 token", "未停(截断)"], rows))
    if "readout_only" in R:
        same = sum(R["readout_only"][i]["generation"] == b[i]["generation"] for i in ids)
        print(f"readout_only 与 base 逐字相同：{same}/{len(ids)}；其延迟相对 base "
              f"{mean([R['readout_only'][i]['latency_s'] for i in ids]) / mean([b[i]['latency_s'] for i in ids]) - 1:+.1%}"
              "（= 读出与选组的纯开销）。\n")

    print("逐题比值 joint / base 的分布（同一题配对）：\n")
    sr = [j[i]["row_steps"] / b[i]["row_steps"] for i in ids]
    lr = [j[i]["latency_s"] / b[i]["latency_s"] for i in ids]
    print(table(["指标", "P10", "P25", "中位", "P75", "P90", "joint 更少的题比例"],
                [["步数比", *[q(sr, p) for p in (.1, .25, .5, .75, .9)], sum(x < 1 for x in sr) / len(sr)],
                 ["延迟比", *[q(lr, p) for p in (.1, .25, .5, .75, .9)], sum(x < 1 for x in lr) / len(lr)]]))

    V = list(j.values())
    gt = sum(r["group_tokens"] for r in V)
    print(f"joint 推理中的通信组：每题平均 {mean([r['gate_open_steps'] for r in V]):.1f} 步开组"
          f"（占步数 {sum(r['gate_open_steps'] for r in V) / sum(r['row_steps'] for r in V):.0%}），"
          f"组内位置当步提交 {sum(r['group_committed'] for r in V) / gt:.1%}，"
          f"最终候选与 L28 draft argmax 不同 {sum(r['group_identity_changed'] for r in V) / gt:.1%}。\n")

    groups = [("全部", ids), ("前 200", ids[:200]), ("后 300", ids[200:])]
    if levels_path:
        lv = json.load(open(levels_path))
        groups += [(f"level {k}", [i for i in ids if lv.get(i) == k]) for k in range(1, 6)]
    rows = []
    for name, sub in groups:
        up = sum(not b[i]["is_correct"] and j[i]["is_correct"] for i in sub)
        dn = sum(b[i]["is_correct"] and not j[i]["is_correct"] for i in sub)
        f = lambda X, k: mean([X[i][k] for i in sub])
        rows.append([name, len(sub), f(b, "is_correct"), f(j, "is_correct"), up, dn, sign_p(up, dn),
                     f(b, "row_steps"), f(j, "row_steps"), f(j, "row_steps") / f(b, "row_steps") - 1,
                     f(b, "latency_s"), f(j, "latency_s"), f(j, "latency_s") / f(b, "latency_s") - 1])
    print(table(["子集", "n", "base 准确率", "joint 准确率", "错→对", "对→错", "符号检验 p", "base 步数",
                 "joint 步数", "步数变化", "base 延迟", "joint 延迟", "延迟变化"], rows))

    print("步数减少幅度与答题结果变化的关系（按逐题步数比分四档）：\n")
    order = sorted(ids, key=lambda i: j[i]["row_steps"] / b[i]["row_steps"])
    rows = []
    for k in range(4):
        sub = order[k * len(order) // 4:(k + 1) * len(order) // 4]
        rows.append([f"Q{k + 1}", mean([j[i]["row_steps"] / b[i]["row_steps"] for i in sub]),
                     sum(not b[i]["is_correct"] and j[i]["is_correct"] for i in sub),
                     sum(b[i]["is_correct"] and not j[i]["is_correct"] for i in sub),
                     sum(b[i]["generation"] == j[i]["generation"] for i in sub)])
    print(table(["档（步数比从小到大）", "平均步数比", "错→对", "对→错", "输出逐字相同"], rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
