"""Raw-table export for pi_sibling_analysis runs (used by make_summary.py).

  tokens.csv.gz    one row per revealed token (all scalar per-token fields, incl. excluded ones)
  steps.csv.gz     one row per (run, rollout, step)
  rollouts.csv     one row per rollout: correctness, question, gold, prediction, completion, means
"""

import csv
import glob
import gzip
import os

import numpy as np
import torch

SKIP = ("topk_S_ids", "topk_S_p", "topk_T_ids", "topk_T_p", "topk_sib_ids", "topk_sib_p")
STEP_KEYS = ("block", "step_in_block", "n_parallel", "n_block_masked", "order_overlap", "order_spearman")
MEAN_KEYS = ("g", "D_S", "D_T", "coord", "D_S_ctrl", "D_T_ctrl", "coord_ctrl", "js_sib", "js_pi",
             "cos_pi_sib", "rank_T_norm", "logp_S")

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
