"""Assemble one REPORT.md from a pipeline run directory (tacc/selective_draft_math.slurm layout).
Works on partial runs: every section reports what exists and says what is missing.

    python -m selective_draft.report RUN_DIR
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional


def _load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl(path: str) -> List[dict]:
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except OSError:
        return []


def _f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def _table(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_meta(run: str) -> str:
    lines = ["## Run metadata", ""]
    for name in ("job.txt", "git.txt", "model_revision.txt", "env.txt"):
        p = os.path.join(run, "meta", name)
        if os.path.exists(p):
            txt = open(p).read().strip()
            lines += [f"**{name}**", "", "```", txt[:3000], "```", ""]
    stages = os.path.join(run, "stages.tsv")
    if os.path.exists(stages):
        rows = list(csv.reader(open(stages), delimiter="\t"))
        if rows:
            lines += ["**Stages** (wall time per stage; `skip` = output already on disk)", "",
                      _table(rows[0], rows[1:]), ""]
    return "\n".join(lines)


def section_smoke(run: str) -> str:
    base = os.path.join(run, "smoke")
    off = _jsonl(os.path.join(base, "parity_off", "rows_thr0.9.jsonl"))
    if not off:
        return "## Parity smoke\n\nnot run.\n"
    rows = []
    for k in ("parity_readout_only", "parity_zero_inject"):
        other = _jsonl(os.path.join(base, k, "rows_thr0.9.jsonl"))
        same = sum(a["generation"] == b["generation"] and a["row_steps"] == b["row_steps"] for a, b in zip(off, other))
        lat_off = (_load(os.path.join(base, "parity_off", "summary.json")) or {}).get("summaries", {}).get("thr0.9", {})
        lat_k = (_load(os.path.join(base, k, "summary.json")) or {}).get("summaries", {}).get("thr0.9", {})
        rows.append([k, f"{same}/{len(other)}", lat_off.get("latency_median_s"), lat_k.get("latency_median_s")])
    return ("## Parity smoke (guidance off vs readout_only vs zero-initialised injector, threshold 0.9)\n\n"
            + _table(["mode", "identical generations + steps", "median latency off (s)", "median latency mode (s)"], rows)
            + "\n\nIdentical outputs are required; the latency gap is the early-readout overhead.\n")


def section_states(run: str) -> str:
    rows = []
    for name in ("train", "analysis_base", "analysis_guided"):
        s = _load(os.path.join(run, "states", name, "summary.json"))
        if s:
            rows.append([name, s.get("rollouts"), s.get("accuracy"), s.get("mean_steps"), s.get("mean_commit"),
                         s.get("gate_open_frac_all"), s.get("gate_open_frac_receiver_steps")])
    if not rows:
        return "## Collected states\n\nnone.\n"
    return ("## Collected states (batched collection; accuracy here is a by-product, not the evaluation)\n\n"
            + _table(["set", "rollouts", "accuracy", "steps/rollout", "commits/step", "gate open (all steps)",
                      "gate open (steps with receivers)"], rows) + "\n")


def section_readout(run: str) -> str:
    keys = ["states", "positions", "kl_mean", "top1_agree", "maxr_mean", "source_agree", "source_maxr",
            "gate_open_0.8", "source_agree_given_gate_0.8", "gate_open_0.9", "source_agree_given_gate_0.9",
            "source_final_committable_0.9"]
    reps = {n: _load(os.path.join(run, "diag", n, "readout_report.json")) for n in ("readout_native", "readout_trained")}
    if not any(reps.values()):
        return "## Early readout\n\nnot run.\n"
    rows = [[k] + [(reps[n] or {}).get(k) for n in reps] for k in keys]
    txt = ("## Early readout at the boundary layer (analysis states, guidance off)\n\n"
           + _table(["metric", "native LM-head lens", "trained adapter"], rows)
           + "\n\n`top1_agree` / `source_agree` compare with the final layer's argmax, which is agreement, not correctness.\n")
    log = _jsonl(os.path.join(run, "readout", "log_rank0.jsonl"))
    if log:
        def avg(rs, k):
            v = [r[k] for r in rs if k in r]
            return sum(v) / len(v) if v else None
        n = max(1, len(log) // 10)
        txt += ("\nStage-1 training curve (mean of first / last 10% of updates): "
                f"loss {_f(avg(log[:n], 'loss'))} → {_f(avg(log[-n:], 'loss'))}, "
                f"top1_agree {_f(avg(log[:n], 'top1_agree'))} → {_f(avg(log[-n:], 'top1_agree'))}, "
                f"updates {len(log)}.\n")
    cal = (reps.get("readout_trained") or {}).get("calibration_buckets")
    if cal:
        txt += "\nTrained-readout calibration (max r bucket → agreement with final argmax):\n\n" + _table(
            ["bucket", "agree rate", "count"], [[k, v["agree_rate"], v["count"]] for k, v in cal.items()]) + "\n"
    return txt


def section_teacher(run: str) -> str:
    names = ["teacher_L28_trained", "teacher_L28_native", "teacher_L24_native"]
    reps = {n: _load(os.path.join(run, "diag", n, "teacher_report.json")) for n in names}
    if not any(reps.values()):
        return "## Teacher targets and patch probe\n\nnot run.\n"
    keys = ["states_with_receivers", "gated_states", "gate_coverage", "source_maxr", "vstar_eq_realized",
            "vstar_eq_final_argmax", "final_argmax_eq_realized", "kl_mc_vs_q0", "kl_topk_vs_q0", "topk_retained_mass",
            "kl_argmax_vs_q0", "kl_realized_vs_q0", "kl_realized_vs_mc", "oracle_kl_before", "oracle_kl_after_patch",
            "oracle_recovery_ratio_of_sums", "oracle_delta_norm"]
    rows = [[k] + [(reps[n] or {}).get(k) for n in names] for k in keys]
    return ("## Teacher targets (mixture vs argmax vs realized) and the activation-patch probe\n\n"
            + _table(["metric"] + names, rows)
            + "\n\nKL columns: how far each Receiver target moves from the unconditional q⁰ (mean over receivers, nats). "
              "The oracle rows patch only h_A at the boundary with its value under the realized reveal and measure how much of "
              "KL(q^realized‖q⁰) the native tail recovers; it is a mechanism probe, not an accuracy estimate.\n")


def section_consistency(run: str) -> str:
    names = ["consistency_base", "consistency_guided"]
    reps = {n: _load(os.path.join(run, "diag", n, "consistency_report.json")) for n in names}
    if not any(reps.values()):
        return "## Same-step consistency\n\nnot run.\n"
    keys = ["states_examined", "steps", "receiver_tokens", "committed_per_step", "psa", "psa_vs_unconditional",
            "pmi", "pmi_negative", "regret", "induced_commits", "induced_fidelity", "induced_baseline_wait_steps"]
    rows = [[k] + [(reps[n] or {}).get(k) for n in names] for k in keys]
    return ("## Same-step consistency (steps with ≥2 commits, reveal = highest-confidence committed token)\n\n"
            + _table(["metric", "base states", "guided states"], rows)
            + "\n\nReading: `psa` = agreement of the parallel commit with the frozen teacher's reveal-A prediction (both can be "
              "wrong); `pmi` = reveal-induced log-prob change (a conditional response, PMI only under joint consistency); "
              "`regret` = the token changes when re-predicted after the step (stability, not correctness); "
              "`induced_fidelity` = message-induced early commits that match the guidance-off continuation.\n")


def section_guidance(run: str) -> str:
    log = [r for r in _jsonl(os.path.join(run, "guidance", "log_rank0.jsonl"))]
    if not log:
        return "## Stage-2 training\n\nnot run.\n"
    upd = [r for r in log if "loss" in r]
    skipped = sum(1 for r in log if r.get("skipped"))
    def avg(rs, k):
        v = [r[k] for r in rs if k in r and r[k] == r[k]]
        return sum(v) / len(v) if v else None
    n = max(1, len(upd) // 5)
    keys = ["loss", "grad_norm", "delta_norm", "source_hidden_norm", "n_receivers", "source_conf_off", "source_conf_on",
            "source_argmax_changed", "readout_argmax_eq_realized", "final_argmax_eq_realized", "refresh_correct",
            "refresh_gate_open_steps", "refresh_rollout_steps", "refresh_buffer_states"]
    rows = [[k, avg(upd[:n], k), avg(upd[-n:], k)] for k in keys]
    mism = upd[-1].get("reselect_mismatch_total") if upd else None
    return ("## Stage-2 (message injector) training\n\n"
            f"updates with a loss: {len(upd)}, skipped updates: {skipped}, source re-selection mismatches: {_f(mism)}\n\n"
            + _table(["metric", "first 20% of updates", "last 20% of updates"], rows)
            + "\n\n`delta_norm / source_hidden_norm` is the relative size of the injected message.\n")


def section_eval(run: str) -> str:
    root = os.path.join(run, "eval")
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []
    summ = {n: _load(os.path.join(root, n, "summary.json")) for n in names}
    if not any(summ.values()):
        return "## Evaluation\n\nnot run.\n"
    rows = []
    for n, s in summ.items():
        for tag, v in ((s or {}).get("summaries") or {}).items():
            rows.append([n, tag, v.get("n"), v.get("accuracy"), v.get("accuracy_se"), v.get("mean_row_steps"),
                         v.get("tokens_per_step"), v.get("latency_median_s"), v.get("latency_p90_s"),
                         v.get("mean_effective_tokens"), v.get("truncation_rate"), v.get("gate_coverage_receiver_steps"),
                         v.get("gate_source_committed_frac")])
    txt = ("## Evaluation (MATH-500 test, batch 1, greedy, timed; guided = stage-2 snapshot)\n\n"
           + _table(["run", "protocol", "n", "acc", "acc SE", "steps/row", "tokens/step", "latency med (s)",
                     "latency p90 (s)", "eff. tokens", "truncated", "gate cov.", "source committed"], rows) + "\n")

    def paired(a_dir, b_dir):
        out = []
        for fn in sorted(os.listdir(os.path.join(root, b_dir))):
            if not fn.startswith("rows_") or not os.path.exists(os.path.join(root, a_dir, fn)):
                continue
            a = {r["example_id"]: r for r in _jsonl(os.path.join(root, a_dir, fn))}
            b = {r["example_id"]: r for r in _jsonl(os.path.join(root, b_dir, fn))}
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            both = sum(a[i]["is_correct"] and b[i]["is_correct"] for i in ids)
            ao = sum(a[i]["is_correct"] and not b[i]["is_correct"] for i in ids)
            bo = sum(b[i]["is_correct"] and not a[i]["is_correct"] for i in ids)
            same = sum(a[i]["generation"] == b[i]["generation"] for i in ids)
            dsteps = sum(b[i]["row_steps"] - a[i]["row_steps"] for i in ids) / len(ids)
            dlat = sum(b[i]["latency_s"] - a[i]["latency_s"] for i in ids) / len(ids)
            out.append([f"{b_dir} vs {a_dir}", fn[5:-6], len(ids), both, ao, bo, len(ids) - both - ao - bo, same,
                        dsteps, dlat])
        return out

    prow = []
    for n in names:
        if n.startswith("guided_") and ("base_" + n[len("guided_"):]) in names:
            prow += paired("base_" + n[len("guided_"):], n)
        if n.startswith("readout_only_") and ("base_" + n[len("readout_only_"):]) in names:
            prow += paired("base_" + n[len("readout_only_"):], n)
    if prow:
        txt += ("\nPaired comparison on the same problems (second run vs first; readout_only must match base exactly, "
                "its latency difference is the early-readout overhead):\n\n"
                + _table(["pair", "protocol", "n", "both right", "first only", "second only", "both wrong",
                          "identical text", "Δ steps/row", "Δ latency s"], prow) + "\n")
    return txt


def main(run: str) -> str:
    parts = [f"# Selective Draft OPSD — MATH preliminary run\n\nRun directory: `{run}`. Generated by "
             "`python -m selective_draft.report`; all numbers are read from the JSON/JSONL files listed in each section.\n",
             section_meta(run), section_smoke(run), section_states(run), section_readout(run), section_teacher(run),
             section_consistency(run), section_guidance(run), section_eval(run)]
    text = "\n\n".join(parts)
    with open(os.path.join(run, "REPORT.md"), "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(main(sys.argv[1]))
