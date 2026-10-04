"""One consolidated report for several pi_sibling_analysis runs, plus every table.

    python analysis/make_summary.py --out RESULTS_DIR --tables_out BIG_TABLE_DIR \\
        --run fixed_base=DIR:"base · fixed" --run fixed_best=DIR:"step-448 · fixed" ... \\
        [--sweep sweep.csv] [--vocab llada_vocab.json] [--notes notes.md] [--figs figs]

Writes
  RESULTS_DIR/SUMMARY.md      setup, accuracy, one section per research question with a run x group
                              table (mean [95% CI]), definitions, figure links and notes
  RESULTS_DIR/metrics.csv     every number in SUMMARY.md (long format)
  RESULTS_DIR/rollouts.csv    one row per rollout (all runs)
  BIG_TABLE_DIR/tokens.csv.gz one row per revealed token (all runs, all scalar fields)
  BIG_TABLE_DIR/steps.csv.gz  one row per decoding step
  BIG_TABLE_DIR/topk.csv.gz   top-20 of p_S, p_T, p_sib per token, decoded with --vocab

Statistics: special tokens and tokens after the answer end are excluded. Sibling, control and
coordination metrics use only tokens whose step revealed >= 2 tokens (|C_t| = 1 has no sibling and
would contribute exact zeros). CIs: 95% cluster bootstrap over rollouts.
--notes: a markdown file with "@@ <section_id>" blocks (overview, accuracy, thu_train_data, one per
section id, cogain, threshold, discussion, caveats, files); each block is inserted at that place.
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pi_sibling import cluster_boot, cogain_stats, pi_type, step_level  # noqa: E402
from report_pi_sibling import export_tables  # noqa: E402

FIELDS = ("g", "D_S", "D_T", "coord", "D_S_ctrl", "D_T_ctrl", "coord_ctrl", "ctrl_ok", "js_sib", "js_ctrl",
          "js_pi", "cos_pi_sib", "cos_pi_ctrl", "logp_S", "ent_S", "ent_T", "top1_T", "token", "step", "block",
          "pos", "n_parallel", "rank_T_norm", "order_overlap", "order_spearman", "ctrl_dist_gap",
          "ctrl_conf_gap", "ctrl_delay")


def load(d):
    rolls = []
    for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
        r = torch.load(f, weights_only=False)
        if r["g"].numel() == 0:
            continue
        keep = ~r["is_special"].bool() & ~r["after_answer"].bool()
        x = {k: r[k][keep].float().numpy() for k in FIELDS}
        x["correct"] = bool(r["meta"]["correct"])
        x["n_all_tokens"] = int(r["g"].numel())
        if x["g"].size:
            rolls.append(x)
    return rolls


# ---- metric definitions: (id, label, fn(rollout) -> values, definition) -------------------------
multi = lambda r: r["n_parallel"] > 1
paired = lambda r: multi(r) & (r["ctrl_ok"] > 0)
A = np.abs

SECTIONS = [
    ("pi_effect", "1. PI 如何改变 student 的分布（讨论文档 13.2、§8）", [
        ("g", "g = log p_T(y) − log p_S(y)", lambda r: r["g"], "PI 对该 rollout 自己 token 的 log-prob 改变；只有 correct 组可称“增益”"),
        ("frac_g_pos", "P(g > 0)", lambda r: (r["g"] > 0).astype(float), "PI 拉高 y 的 token 比例"),
        ("frac_lift", "P(p_S(y)<0.5 且 g>0.5)", lambda r: ((np.exp(r["logp_S"]) < 0.5) & (r["g"] > 0.5)).astype(float),
         "student 原本支持一般、PI 明显拉高的比例"),
        ("js_pi", "JS(p_S, p_T)", lambda r: r["js_pi"], "PI 引起的整体分布变化（nats）"),
        ("d_ent", "H(p_T) − H(p_S)", lambda r: r["ent_T"] - r["ent_S"], "PI 让分布更尖（<0）还是更平（>0）"),
        ("sharpen", "强化 sharpen", lambda r: (pi_type(r) == 0).astype(float), "teacher top-1 = y 且 g > 0"),
        ("defer", "暂缓 defer", lambda r: (pi_type(r) == 1).astype(float), "teacher top-1 = y 但 g ≤ 0（同方向但更不确定）"),
        ("redirect", "改向 redirect", lambda r: (pi_type(r) == 2).astype(float), "teacher top-1 ≠ y"),
    ]),
    ("sibling_gap", "2. 同一步 sibling gap（13.3）", [
        ("D_S", "D_S（带符号）", lambda r: r["D_S"][multi(r)], "log p_S(y_i | 兄弟) − log p_S(y_i)；>0 兄弟支持，<0 兄弟否定"),
        ("absD_S", "|D_S|", lambda r: A(r["D_S"][multi(r)]), "不预设符号的 gap 大小"),
        ("frac_DS_neg", "P(D_S < 0)", lambda r: (r["D_S"][multi(r)] < 0).astype(float), "兄弟否定当前 token 的比例"),
        ("js_sib", "JS(p_S, p_sib)", lambda r: r["js_sib"][multi(r)], "看到兄弟后整个分布的变化"),
        ("D_S_early", "D_S，block 0–1", lambda r: r["D_S"][multi(r) & (r["block"] <= 1)], "解码早期"),
        ("D_S_mid", "D_S，block 2–4", lambda r: r["D_S"][multi(r) & (r["block"] >= 2) & (r["block"] <= 4)], "解码中期"),
        ("D_S_late", "D_S，block 5–6", lambda r: r["D_S"][multi(r) & (r["block"] >= 5)], "解码后期"),
    ]),
    ("teacher_gap", "3. 加入 PI 后的 coordination gap（13.4）", [
        ("D_T", "D_T（带符号）", lambda r: r["D_T"][multi(r)], "log p_T(y_i | 兄弟, PI) − log p_T(y_i | PI)"),
        ("absD_T", "|D_T|", lambda r: A(r["D_T"][multi(r)]), ""),
        ("abs_reduction", "|D_S| − |D_T|（配对）", lambda r: (A(r["D_S"]) - A(r["D_T"]))[multi(r)],
         ">0：有 PI 后兄弟带来的改变变小，即 PI 已提供部分协调信息"),
        ("frac_abs_smaller", "P(|D_T| < |D_S|)", lambda r: (A(r["D_T"]) < A(r["D_S"]))[multi(r)].astype(float), ""),
        ("coord", "coord = D_S − D_T", lambda r: r["coord"][multi(r)], "原始定义（带符号）"),
    ]),
    ("control", "4. matched non-sibling control（13.6）", [
        ("ctrl_ok", "有匹配对照的比例", lambda r: (r["ctrl_ok"] > 0)[multi(r)].astype(float), "block 最后一步无候选"),
        ("D_S_ctrl", "D_S,ctrl（带符号）", lambda r: r["D_S_ctrl"][paired(r)], "改填一个匹配的、之后才揭开的 token"),
        ("absD_S_ctrl", "|D_S,ctrl|", lambda r: A(r["D_S_ctrl"][paired(r)]), ""),
        ("sib_minus_ctrl", "D_S − D_S,ctrl（配对）", lambda r: (r["D_S"] - r["D_S_ctrl"])[paired(r)],
         "<0：同一步兄弟比先后提交的 token 更不支持 y_i"),
        ("abs_sib_minus_ctrl", "|D_S| − |D_S,ctrl|（配对）", lambda r: (A(r["D_S"]) - A(r["D_S_ctrl"]))[paired(r)],
         ">0：兄弟引起的改变比对照大"),
        ("js_sib_minus_ctrl", "JS_sib − JS_ctrl（配对）", lambda r: (r["js_sib"] - r["js_ctrl"])[paired(r)], ""),
        ("abs_reduction_ctrl", "|D_S,ctrl| − |D_T,ctrl|（配对）", lambda r: (A(r["D_S_ctrl"]) - A(r["D_T_ctrl"]))[paired(r)],
         "PI 对“对照 token 的作用”的替代程度"),
        ("reduction_sib_minus_ctrl", "(|D_S|−|D_T|) − (|D_S,ctrl|−|D_T,ctrl|)",
         lambda r: ((A(r["D_S"]) - A(r["D_T"])) - (A(r["D_S_ctrl"]) - A(r["D_T_ctrl"])))[paired(r)],
         ">0：PI 缓解兄弟 gap 的幅度超过缓解对照 gap 的幅度"),
        ("ctrl_dist_gap", "匹配质量：距离差", lambda r: r["ctrl_dist_gap"][paired(r)], "|j−i| 与 |k−i| 之差（位置）"),
        ("ctrl_conf_gap", "匹配质量：信心差", lambda r: r["ctrl_conf_gap"][paired(r)], "|log p_S(y_j) − log p_S(y_k)|"),
        ("ctrl_delay", "匹配质量：揭开延迟", lambda r: r["ctrl_delay"][paired(r)], "对照比兄弟晚几步揭开"),
    ]),
    ("decisions", "6. PI 与解码决策：暂缓与顺序（§8）", [
        ("defer_redirect_conflict", "暂缓或改向 | D_S < 0", lambda r: (pi_type(r) > 0)[multi(r) & (r["D_S"] < 0)].astype(float), "兄弟冲突时"),
        ("defer_redirect_noconflict", "暂缓或改向 | D_S ≥ 0", lambda r: (pi_type(r) > 0)[multi(r) & (r["D_S"] >= 0)].astype(float), ""),
        ("overlap", "顺序重合度", lambda r: step_level(r, "order_overlap"), "|C_t ∩ teacher top-|C_t|| / |C_t|（逐步）"),
        ("premature", "过早提交率", lambda r: (r["rank_T_norm"] > 0.5).astype(float), "teacher 排名在 block 后一半"),
        ("premature_conflict", "过早提交率 | D_S < 0", lambda r: (r["rank_T_norm"] > 0.5)[multi(r) & (r["D_S"] < 0)].astype(float), ""),
        ("premature_noconflict", "过早提交率 | D_S ≥ 0", lambda r: (r["rank_T_norm"] > 0.5)[multi(r) & (r["D_S"] >= 0)].astype(float), ""),
        ("spearman", "顺序相关 Spearman", lambda r: step_level(r, "order_spearman"), "conf_S 与 conf_T 在 block 内的秩相关（逐步）"),
    ]),
    ("direction", "7. PI 的变化方向是否和兄弟一致", [
        ("cos_sib", "cos(Δ_PI, Δ_sib)", lambda r: r["cos_pi_sib"][multi(r)], "Δ_PI = p_T − p_S，Δ_sib = p_sib − p_S"),
        ("frac_cos_sib_pos", "P(cos(Δ_PI, Δ_sib) > 0)", lambda r: (r["cos_pi_sib"] > 0)[multi(r)].astype(float), ""),
        ("frac_cos_ctrl_pos", "P(cos(Δ_PI, Δ_ctrl) > 0)", lambda r: (r["cos_pi_ctrl"] > 0)[paired(r)].astype(float), "对照基线"),
        ("cos_sib_minus_ctrl", "cos_sib − cos_ctrl（配对）", lambda r: (r["cos_pi_sib"] - r["cos_pi_ctrl"])[paired(r)],
         ">0：PI 的方向更像兄弟带来的方向，而非对照"),
    ]),
]
THR_BINS = [(2, 2, "2"), (3, 4, "3–4"), (5, 8, "5–8"), (9, 10**6, "9+")]


def esc(x):
    """Markdown table cell: '|' would split the cell (labels like |D_S|, |C_t|)."""
    return str(x).replace("|", "\\|")


def fmt(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:+.3f}"


def table(runs, data, rows_spec, section_id, metrics_rows):
    """rows_spec: [(id, label, fn, definition)] -> markdown lines; also appends to metrics_rows."""
    head = "| 指标 | 组 | " + " | ".join(esc(lab) for _, lab in runs) + " |"
    L = [head, "|---|---|" + "---|" * len(runs)]
    for mid, label, fn, _ in rows_spec:
        for g in ("correct", "wrong"):
            cells = []
            for run, _ in runs:
                rs = [r for r in data[run] if r["correct"] == (g == "correct")]
                vals = [v for v in (fn(r) for r in rs) if np.asarray(v).size]
                m, lo, hi = cluster_boot(vals, np.mean) if vals else (np.nan,) * 3
                n_tok = int(sum(np.asarray(v).size for v in vals))
                metrics_rows.append([section_id, mid, label, run, g, len(vals), n_tok, m, lo, hi, ""])
                cells.append(f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]" if np.isfinite(m) else "–")
            L.append(f"| {esc(label)} | {g} | " + " | ".join(cells) + " |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='id=records_dir:"label"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables_out", required=True)
    ap.add_argument("--sweep", default="")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--figs", default="figs", help="figure dir relative to --out, one subdir per run id")
    ap.add_argument("--title", default="Results")
    ap.add_argument("--preamble", default="", help="markdown file inserted after the title")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tables_out, exist_ok=True)

    runs, dirs, data = [], {}, {}
    for spec in args.run:
        rid, rest = spec.split("=", 1)
        d, label = rest.split(":", 1)
        runs.append((rid, label)), dirs.update({rid: d})
        data[rid] = load(d)
        print(f"[summary] {rid}: {len(data[rid])} rollouts")
    notes = {}
    if args.notes:
        for block in re.split(r"(?m)^@@ ", open(args.notes).read())[1:]:
            key, _, body = block.partition("\n")
            notes[key.strip()] = body.strip()

    metrics = []
    L = [f"# {args.title}", ""]
    if args.preamble:
        L += [open(args.preamble).read().strip(), ""]
    if "overview" in notes:
        L += [notes["overview"], ""]

    # accuracy
    L += ["## 0. 准确率与样本量", "", "| 组 | rollouts | correct | wrong | 准确率 | 纳入分析的 token |", "|---|---|---|---|---|---|"]
    for rid, lab in runs:
        rs = data[rid]
        ok = sum(r["correct"] for r in rs)
        L.append(f"| {lab} | {len(rs)} | {ok} | {len(rs) - ok} | {ok / len(rs):.1%} | {sum(r['g'].size for r in rs)} |")
        metrics.append(["accuracy", "accuracy", "accuracy", rid, "all", len(rs), "", ok / len(rs), "", "", ""])
    if args.sweep:
        L += ["", "checkpoint 扫描（greedy，固定预算 128 步，300 题）：", "", "| checkpoint | 准确率 |", "|---|---|"]
        for r in csv.DictReader(open(args.sweep)):
            L.append(f"| {r['name']} | {float(r['accuracy']):.1%} |")
            metrics.append(["sweep", r["name"], "greedy accuracy", r["name"], "all", 1, r["n"], float(r["accuracy"]), "", "", ""])
    L.append("")
    if "accuracy" in notes:
        L += [notes["accuracy"], ""]
    if "thu_train_data" in notes:  # standalone chapter, placed before the per-question sections
        L += [notes["thu_train_data"], ""]

    fig_links = {"pi_effect": ["pi_gain", "pi_effect_types"], "sibling_gap": ["sibling_gap", "sibling_vs_step"],
                 "teacher_gap": ["B_coordination_gap"], "control": ["C_sibling_vs_control"],
                 "decisions": ["decoding_order", "pi_effect_types"], "direction": ["pi_vs_sibling_direction"],
                 "cogain": ["A_cogain"], "threshold": ["threshold_sibling_vs_parallel"]}

    def figs_line(key):
        out = []
        for name in fig_links.get(key, []):
            links = [f"[{lab}]({args.figs}/{rid}/{name}.png)" for rid, lab in runs
                     if os.path.exists(os.path.join(args.out, args.figs, rid, f"{name}.png"))]
            if links:
                out.append(f"图 `{name}`：" + " · ".join(links))
        return ["", *out, ""] if out else [""]

    for sid, title, spec in SECTIONS:
        L += [f"## {title}", ""]
        L += ["| 指标 | 定义 / 读法 |", "|---|---|"] + [f"| {esc(lab)} | {esc(d)} |" for _, lab, _, d in spec] + [""]
        L += table(runs, data, spec, sid, metrics)
        L += figs_line(sid)
        if sid in notes:
            L += [notes[sid], ""]
        if sid == "control":  # A (CoGain) follows the control section
            L += ["## 5. CoGain：PI 是否让同一步 token 成组受益（13.5）", "",
                  "原定义 CoGain_t = |C_t|⁻¹ Σ 1[g_i>0] 在每步 |C_t| 相同时，均值不受打乱影响；因此改为成对检验："
                  "同一步的 token 对“都被 PI 拉高”的比例，对比**同一 rollout、同一 block、距离相同但不同步提交**的 token 对"
                  "（排除位置相邻效应）。p = 单侧置换检验，差值 CI = rollout bootstrap。", "",
                  "| 组 | 模型·解码 | 同一步对数 | P(都拉高) 真实 | 距离匹配期望 | 差 [95% CI] | 置换 p |", "|---|---|---|---|---|---|---|"]
            rng = np.random.default_rng(0)
            for g in ("correct", "wrong"):
                for rid, lab in runs:
                    rs = [r for r in data[rid] if r["correct"] == (g == "correct")]
                    c = cogain_stats(rs, rng)
                    if c is None:
                        continue
                    L.append(f"| {g} | {lab} | {c['n_pairs']} | {c['real']:.3f} | {c['expect']:.3f} | "
                             f"{c['diff']:+.3f} [{c['diff_lo']:+.3f}, {c['diff_hi']:+.3f}] | {c['p']:.3f} |")
                    metrics.append(["cogain", "pair_cogain_diff", "co-gain real − distance-matched expectation", rid, g,
                                    len(rs), c["n_pairs"], c["diff"], c["diff_lo"], c["diff_hi"], f"p={c['p']:.4f}"])
                    metrics.append(["cogain", "pair_cogain_real", "P(both g>0) co-decoded pairs", rid, g,
                                    len(rs), c["n_pairs"], c["real"], "", "", ""])
                    metrics.append(["cogain", "pair_cogain_expect", "distance-matched expectation", rid, g,
                                    len(rs), c["n_pairs"], c["expect"], "", "", ""])
            L += figs_line("cogain")
            if "cogain" in notes:
                L += [notes["cogain"], ""]

    # threshold: by natural |C_t|
    thr_runs = [(rid, lab) for rid, lab in runs if rid.startswith("thr")]
    if thr_runs:
        L += ["## 8. 阈值解码：gap 随同一步揭开数 |C_t| 的变化", "",
              "| 指标 | 组 | \\|C_t\\| | " + " | ".join(esc(lab) for _, lab in thr_runs) + " |", "|---|---|---|" + "---|" * len(thr_runs)]
        for key, lab, f in (("absD_S", "|D_S|", lambda r, s: A(r["D_S"][s])), ("D_S", "D_S", lambda r, s: r["D_S"][s]),
                            ("js_sib", "JS_sib", lambda r, s: r["js_sib"][s]), ("absD_T", "|D_T|", lambda r, s: A(r["D_T"][s]))):
            for g in ("correct", "wrong"):
                for lo_n, hi_n, blab in THR_BINS:
                    cells = []
                    for rid, _ in thr_runs:
                        rs = [r for r in data[rid] if r["correct"] == (g == "correct")]
                        vals = [v for v in (f(r, (r["n_parallel"] >= lo_n) & (r["n_parallel"] <= hi_n)) for r in rs) if v.size]
                        m, l_, h_ = cluster_boot(vals, np.mean) if vals else (np.nan,) * 3
                        metrics.append(["threshold", f"{key}|C_t|={blab}", lab, rid, g, len(vals),
                                        int(sum(v.size for v in vals)), m, l_, h_, ""])
                        cells.append(f"{fmt(m)} [{fmt(l_)}, {fmt(h_)}]" if np.isfinite(m) else "–")
                    L.append(f"| {esc(lab)} | {g} | {blab} | " + " | ".join(cells) + " |")
        L += figs_line("threshold")
        if "threshold" in notes:
            L += [notes["threshold"], ""]

    for key in ("discussion", "caveats", "files"):
        if key in notes:
            L += [notes[key], ""]
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric_id", "label", "run", "group", "n_rollouts", "n_values", "value", "ci_lo", "ci_hi", "note"])
        w.writerows(metrics)
    print(f"[summary] SUMMARY.md + metrics.csv ({len(metrics)} rows) -> {args.out}")

    # tables: tokens / steps / rollouts (all runs in one file each), top-k
    export_tables(dirs, args.tables_out)
    os.replace(os.path.join(args.tables_out, "rollouts.csv"), os.path.join(args.out, "rollouts.csv"))
    if args.vocab:
        export_topk(dirs, json.load(open(args.vocab)), os.path.join(args.tables_out, "topk.csv.gz"))
    print(f"[summary] tables -> {args.tables_out} (rollouts.csv -> {args.out})")


def export_topk(dirs, vocab, path):
    """One row per analysed token: y and the top-20 of p_S, p_T, p_sib as JSON [[token, prob], ...]."""
    dec = lambda ids, ps: json.dumps([[vocab[i], round(float(p), 4)] for i, p in zip(ids, ps)], ensure_ascii=False)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "rollout", "correct", "step", "pos", "n_parallel", "is_special", "after_answer",
                    "y_id", "y", "p_S_y", "p_T_y", "p_sib_y", "top20_S", "top20_T", "top20_sib"])
        for run, d in dirs.items():
            for fpath in sorted(glob.glob(os.path.join(d, "*.pt"))):
                r = torch.load(fpath, weights_only=False)
                rid = os.path.splitext(os.path.basename(fpath))[0]
                ok = int(bool(r["meta"]["correct"]))
                cols = {k: r[k].numpy() for k in ("step", "pos", "n_parallel", "is_special", "after_answer", "token",
                                                    "logp_S", "logp_T", "logp_sib", "topk_S_ids", "topk_S_p",
                                                    "topk_T_ids", "topk_T_p", "topk_sib_ids", "topk_sib_p")}
                for i in range(r["g"].numel()):
                    y = int(cols["token"][i])
                    w.writerow([run, rid, ok, int(cols["step"][i]), int(cols["pos"][i]), int(cols["n_parallel"][i]),
                                int(cols["is_special"][i]), int(cols["after_answer"][i]), y, vocab[y],
                                f"{np.exp(cols['logp_S'][i]):.4f}", f"{np.exp(cols['logp_T'][i]):.4f}",
                                f"{np.exp(cols['logp_sib'][i]):.4f}",
                                dec(cols["topk_S_ids"][i], cols["topk_S_p"][i]),
                                dec(cols["topk_T_ids"][i], cols["topk_T_p"][i]),
                                dec(cols["topk_sib_ids"][i], cols["topk_sib_p"][i])])


if __name__ == "__main__":
    main()
