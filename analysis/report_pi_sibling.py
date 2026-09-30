"""Machine-readable results for pi_sibling_analysis runs (for humans and for AI-assisted analysis).

Called by plot_pi_sibling.py; writes into the figure directory:
  report.md        run metadata, metric definitions, every summary number as markdown tables
  tokens.csv.gz    one row per revealed token (all scalar per-token fields, incl. excluded ones)
  steps.csv.gz     one row per (run, rollout, step)
  rollouts.csv     one row per rollout: correctness, question, gold, prediction, completion, means
"""

import csv
import datetime
import glob
import gzip
import os
import subprocess

import numpy as np
import torch

SKIP = ("topk_S_ids", "topk_S_p", "topk_T_ids", "topk_T_p", "topk_sib_ids", "topk_sib_p")
STEP_KEYS = ("block", "step_in_block", "n_parallel", "n_block_masked", "order_overlap", "order_spearman")
MEAN_KEYS = ("g", "D_S", "D_T", "coord", "D_S_ctrl", "D_T_ctrl", "coord_ctrl", "js_sib", "js_pi",
             "cos_pi_sib", "rank_T_norm", "logp_S")

DEFINITIONS = """\
All quantities are computed per revealed token i at the step t where it was committed (C_t = tokens
revealed at step t, y_i = its value in the final rollout). Tokens of the last block (no future, so no
PI) are not analysed. Unless stated otherwise, summaries exclude special tokens (EOS/EOT) and tokens
after the answer end. CIs are 95% cluster bootstrap over rollouts.

| name | definition | reading |
|---|---|---|
| g | log p_T(y_i \\| S_t, PI) - log p_S(y_i \\| S_t) | PI alignment with the rollout's own token; a gain only on correct rollouts |
| D_S | log p_S(y_i \\| S_t, y[C_t\\\\i]) - log p_S(y_i \\| S_t) | what the co-decoded siblings add for the student |
| D_T | log p_T(y_i \\| S_t, PI, y[C_t\\\\i]) - log p_T(y_i \\| S_t, PI) | what siblings still add once PI is known |
| coord | D_S - D_T | part of the sibling information PI already provides (hypothesis: > 0 on correct rollouts) |
| *_ctrl | same with a matched non-sibling control (same block, revealed later, matched on distance, student confidence and reveal delay) | sibling - control > 0 means same-step siblings are special |
| js_sib / js_pi | JS(p_S, p_sib) / JS(p_S, p_T), full vocabulary, nats | size of the distribution shift |
| cos_pi_sib | cos(p_T - p_S, p_sib - p_S) over the vocabulary | do the PI shift and the sibling shift agree |
| pair co-gain | P(g_i > 0 and g_j > 0) for co-decoded pairs vs distance-matched pairs revealed at different steps | PI lifts co-decoded tokens together beyond position effects |
| sharpen / defer / redirect | teacher top-1 = y_i and g > 0 / teacher top-1 = y_i and g <= 0 / teacher top-1 != y_i | what PI does to the token; defer can be useful (hold the slot back) |
| rank_S / rank_T | rank of i among all masked slots of the block by top-1 prob (1 = most confident) | decoding-order preference |
| rank_T_norm | (rank_T - 1) / (#masked in block - 1) | > 0.5: teacher would commit it late (premature commit) |
| order_overlap | \\|C_t & teacher's top-\\|C_t\\|\\| / \\|C_t\\| | would the teacher reveal the same slots now |
| order_spearman | Spearman(conf_S, conf_T) over the block's masked slots | overall order agreement |
"""


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "unknown"


def _num(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.6g}"
    return v


def export_tables(sets, out):
    """sets: {run_name: records_dir}. Returns per-run metadata rows for the report."""
    runs = []
    tok_f = gzip.open(os.path.join(out, "tokens.csv.gz"), "wt", newline="")
    step_f = gzip.open(os.path.join(out, "steps.csv.gz"), "wt", newline="")
    roll_f = open(os.path.join(out, "rollouts.csv"), "w", newline="")
    tok_w = step_w = roll_w = None
    for run, d in sets.items():
        files = sorted(glob.glob(os.path.join(d, "*.pt")))
        n_ok = 0
        meta0 = None
        for f in files:
            r = torch.load(f, weights_only=False)
            m, src = r["meta"], r["meta"].get("source", {})
            meta0 = meta0 or m
            rid = os.path.splitext(os.path.basename(f))[0]
            n_ok += bool(m["correct"])
            keys = [k for k in r if k != "meta" and k not in SKIP and torch.is_tensor(r[k]) and r[k].dim() == 1]
            n = r["g"].numel()
            if tok_w is None:
                tok_keys = keys
                tok_w = csv.writer(tok_f)
                tok_w.writerow(["run", "rollout", "correct"] + tok_keys)
            cols = {k: r[k].float().numpy() if r[k].dtype.is_floating_point
                    else r[k].int().numpy() for k in tok_keys if k in r}  # bools -> 0/1
            for i in range(n):
                tok_w.writerow([run, rid, int(m["correct"])] + [_num(cols[k][i]) if k in cols else "" for k in tok_keys])
            # step level
            if step_w is None:
                step_w = csv.writer(step_f)
                step_w.writerow(["run", "rollout", "correct", "step"] + list(STEP_KEYS)
                                + ["mean_g", "frac_g_pos", "all_g_pos", "n_special"])
            steps = r["step"].numpy()
            for t in np.unique(steps):
                sel = steps == t
                first = np.argmax(sel)
                g = r["g"].numpy()[sel]
                step_w.writerow([run, rid, int(m["correct"]), int(t)]
                                + [_num(float(r[k].numpy()[first])) if k in r else "" for k in STEP_KEYS]
                                + [_num(float(g.mean())), _num(float((g > 0).mean())), int((g > 0).all()),
                                   int(r["is_special"].numpy()[sel].sum())])
            # rollout level (means over analysed, non-special, pre-answer-end tokens)
            keep = (~r["is_special"].bool() & ~r["after_answer"].bool()).numpy()
            if roll_w is None:
                roll_w = csv.writer(roll_f)
                roll_w.writerow(["run", "rollout", "correct", "n_tokens_analysed", "gold", "pred", "question",
                                 "completion"] + [f"mean_{k}" for k in MEAN_KEYS])
            means = []
            for k in MEAN_KEYS:
                v = r[k].float().numpy()[keep] if k in r else np.array([])
                v = v[~np.isnan(v)]
                means.append(_num(float(v.mean())) if v.size else "")
            roll_w.writerow([run, rid, int(m["correct"]), int(keep.sum()), src.get("ground_truth", src.get("gold", "")),
                             src.get("parsed_answer", src.get("pred", "")), src.get("question", ""),
                             src.get("generations", src.get("completion", ""))] + means)
        src0 = (meta0 or {}).get("source", {})
        runs.append({
            "run": run, "records": d, "rollouts": len(files),
            "accuracy": f"{n_ok / len(files):.3f}" if files else "",
            "steps": src0.get("steps", ""), "threshold": src0.get("threshold") or "none (fixed budget)",
            "temperature": src0.get("temperature", ""), "gen_length": (meta0 or {}).get("gen_length", ""),
            "block_length": (meta0 or {}).get("block_length", ""),
            "adapter": (meta0 or {}).get("adapter", "") or "none (base model)",
            "teacher": (meta0 or {}).get("teacher", ""), "pi_samples": (meta0 or {}).get("pi_samples", ""),
            "teacher_retain_ratio": (meta0 or {}).get("teacher_retain_ratio", ""),
        })
    tok_f.close(), step_f.close(), roll_f.close()
    return runs


def write_report(out, runs, rows, primary, note=""):
    L = [f"# PI / intra-step coordination analysis report", "",
         f"generated {datetime.datetime.now().isoformat(timespec='seconds')}, code {_git_rev()}", ""]
    if note:
        L += [note, ""]
    L += ["## Runs", "", "| " + " | ".join(runs[0].keys()) + " |", "|" + "---|" * len(runs[0])]
    L += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in runs]
    L += ["", f"Figures 1-3 and 5-8, 10, 11 use the primary run `{primary}`; figure 4 uses every fixed-budget run; "
          "figure 9 uses the threshold run.", "", "## Definitions", "", DEFINITIONS, "## Results", ""]
    titles = {"fig1": "PI gain", "fig2": "Sibling gap", "fig3": "Sibling gap vs step", "fig4": "Sibling gap vs |C_t| (fixed budgets)",
              "fig5": "PI shift vs sibling shift", "fig6": "B. Coordination gap", "fig7": "C. Sibling vs matched control (paired)",
              "fig8": "A. Pairwise co-gain (distance-matched)", "fig9": "Sibling gap vs natural |C_t| (threshold)",
              "fig10": "PI effect types", "fig11": "Decoding order"}
    for fig in titles:
        sub = [r for r in rows if r[0] == fig]
        if not sub:
            continue
        L += [f"### {titles[fig]} ({fig})", "", "| quantity | group | n_rollouts | note | stat | value | 95% CI |",
              "|---|---|---|---|---|---|---|"]
        for _, q, g, n, note_, stat, v, lo, hi in sub:
            # one table row per entry: no newlines, and escape "|" (a markdown column separator)
            q, note_ = (str(x).replace("\n", " ").replace("|", "\\|") for x in (q, note_))
            ci = f"[{_num(lo)}, {_num(hi)}]" if _num(lo) != "" else ""
            L.append(f"| {q} | {g} | {n} | {note_} | {stat} | {_num(v)} | {ci} |")
        L.append("")
    L += ["## Files", "", "- `summary.csv`: every number above", "- `tokens.csv.gz`: one row per revealed token (all runs)",
          "- `steps.csv.gz`: one row per decoding step", "- `rollouts.csv`: one row per rollout with question/gold/prediction/completion",
          "- `fig*.png`: figures", ""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L))
