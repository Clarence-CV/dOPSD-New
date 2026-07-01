"""
Plot gsm8k Acc@1 vs training step for two dOPSD supervision variants:
  - random DECODE STEP  (on-policy-style: noise drawn from the decoding schedule)
  - random MASK TOKEN   (naive port: uniform re-mask of the rollout)

Same style/font as plot_gsm8k_acc.py and visualize.py (bold sans-serif / Lato,
light grid, full box, faint glow behind the line).

Usage:
    python plot_decode_vs_mask.py --out_dir results
Produces:
    results/decode_vs_mask.png  and  results/decode_vs_mask.pdf
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


STEPS = [200, 500, 800, 1100]

SERIES = [
    ("dOPSD random decode step", [82.56, 82.33, 82.18, 83.09], "#1f77b4"),
    ("dOPSD random mask token",  [77.94, 74.49, 72.68, 73.24], "#d62728"),
]


def pick_font():
    preferred = ["Lato", "Source Sans Pro", "Source Sans 3", "Helvetica",
                 "Arial", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def apply_style():
    font = pick_font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
    })
    return font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--title", default="eval/gsm8k")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    font = apply_style()
    print(f"[font] using '{font}'")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    x = np.asarray(STEPS)
    for label, vals, color in SERIES:
        y = np.asarray(vals)
        ax.plot(x, y, color=color, lw=6, alpha=0.15, solid_capstyle="round", zorder=2)
        ax.plot(x, y, color=color, lw=2.5, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)

    ax.set_title(args.title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Acc@1")
    ax.set_xticks(STEPS)
    ax.grid(True, color="0.85", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor("0.3")

    # both lines sit high -> open extra room at the bottom for the legend
    allv = np.concatenate([np.asarray(v) for _, v, _ in SERIES])
    rng = allv.max() - allv.min()
    ax.set_ylim(allv.min() - 0.35 * rng, allv.max() + 0.12 * rng)
    ax.set_xlim(min(STEPS) - 60, max(STEPS) + 60)

    leg = ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95,
                    borderpad=0.5, handlelength=1.6)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.out_dir, f"decode_vs_mask.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("[saved]", out)


if __name__ == "__main__":
    main()
