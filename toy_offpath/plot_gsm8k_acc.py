"""
Plot gsm8k Acc@1 vs training step for three OPSD variants, styled to match the
wandb-style eval curves used in the paper (bold sans-serif labels, light grid,
full box, faint "raw" line behind a bold "smoothed" line, legend top-left).

Usage:
    python plot_gsm8k_acc.py --out_dir results
Produces:
    results/gsm8k_acc.png  and  results/gsm8k_acc.pdf
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
STEPS = [200, 500, 800, 1100]

# label -> (values per step, color).  Palette mirrors the reference figure:
#   winner = blue, answer-only = red, full-solution = green.
SERIES = [
    ("dOPSD",              [82.56, 82.33, 82.18, 83.09], "#1f77b4"),
    ("OPSD answer only",   [78.46, 80.34, 79.42, 80.16], "#d62728"),
    ("OPSD full solution", [65.20, 65.47, 67.25, 64.38], "#2ca02c"),
]


# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #
def pick_font():
    """Pick the closest available font to the reference (wandb) bold sans-serif."""
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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
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
        # faint "raw" glow behind, then the bold "smoothed" line + markers
        ax.plot(x, y, color=color, lw=6, alpha=0.15, solid_capstyle="round", zorder=2)
        ax.plot(x, y, color=color, lw=2.5, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)

    # axes / box / grid styled like the reference
    ax.set_title(args.title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Acc@1")
    ax.set_xticks(STEPS)
    ax.grid(True, color="0.85", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor("0.3")

    # a little vertical padding around the data
    allv = np.concatenate([np.asarray(v) for _, v, _ in SERIES])
    pad = 0.08 * (allv.max() - allv.min())
    ax.set_ylim(allv.min() - pad, allv.max() + pad)
    ax.set_xlim(min(STEPS) - 60, max(STEPS) + 60)

    # place the legend in the empty middle band (between the answer-only and
    # full-solution curves) so it never overlaps a line
    leg = ax.legend(loc="center left", fontsize=10, frameon=True, framealpha=0.95,
                    borderpad=0.5, handlelength=1.6)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.out_dir, f"gsm8k_acc.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("[saved]", out)


if __name__ == "__main__":
    main()
