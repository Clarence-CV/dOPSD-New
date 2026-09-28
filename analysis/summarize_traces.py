"""Headline stats over LLaDA trace files (*.pt from eval --trace_dir or trainer traces/).

    python analysis/summarize_traces.py <trace_dir> [<trace_dir> ...]

Per dir: #rollouts, accuracy, |C_t| histogram, and for correct vs wrong rollouts the
mean entropy over masked positions, mean prob of the revealed tokens, and the mean
position gap between tokens revealed in the same step (parallel-commit spread).
"""

import glob
import os
import sys
from collections import Counter

import torch


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def rollout_stats(tr):
    rv = tr["reveal"]
    gaps = []
    for t in rv["step"].unique():
        pos = rv["pos"][rv["step"] == t].long().sort().values
        if pos.numel() > 1:
            gaps.append(float((pos[1:] - pos[:-1]).float().mean()))
    return {
        "entropy": float(tr["masked"]["entropy"].float().mean()),
        "reveal_prob": float(rv["prob"].mean()),
        "gap": mean(gaps),
    }


def main(dirs):
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.pt")))
        if not files:
            print(f"{d}: no traces")
            continue
        c_hist, groups = Counter(), {True: [], False: []}
        for f in files:
            tr = torch.load(f, weights_only=False)
            ok = tr["meta"].get("is_correct")
            ok = bool(ok >= 1.0) if isinstance(ok, float) else bool(ok)  # sudoku stores cell accuracy
            c_hist.update(tr["n_revealed"].tolist())
            groups[ok].append(rollout_stats(tr))
        n = len(files)
        print(f"== {d}")
        print(f"   rollouts={n}  acc={len(groups[True]) / n:.4f}  "
              f"steps/rollout={sum(c_hist.values()) / n:.1f}  |C_t| histogram={dict(sorted(c_hist.items()))}")
        for name, key in (("correct", True), ("wrong", False)):
            g = groups[key]
            print(f"   {name:7s} n={len(g):4d}  masked entropy={mean([s['entropy'] for s in g]):.4f}  "
                  f"revealed prob={mean([s['reveal_prob'] for s in g]):.4f}  "
                  f"same-step gap={mean([s['gap'] for s in g]):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
