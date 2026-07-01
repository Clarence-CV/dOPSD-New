"""
Visualize the toy "off the decoding path" experiment -- off-path fraction only.

Panel (c): fraction of uniform-masked tokens that are actually "easy" -- i.e.
above the on-policy frontier (commit-confidence higher than the most-confident
token the on-policy schedule would still have masked at level p). On-policy is 0
by construction. A large uniform fraction means the naive random mask routinely
hides tokens the model would have committed early -- partial states the dLLM
never visits.

This is computed purely from the logged commit-confidences, so it is the
robust, confound-free confirmation of the obstacle.

Styled to match plot_gsm8k_acc.py (bold sans-serif / Lato, light grid, full box,
faint glow behind the line) so the paper figures share one look.

Usage:
    python visualize.py --out_dir results
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# --------------------------------------------------------------------------- #
# styling (shared look with plot_gsm8k_acc.py)
# --------------------------------------------------------------------------- #
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
    args = ap.parse_args()

    font = apply_style()
    print(f"[font] using '{font}'")

    with open(os.path.join(args.out_dir, "results.json")) as f:
        R = json.load(f)

    p_levels = np.asarray(R["p_levels"])
    offp = np.asarray(R["offpath_frac"])        # [P, 2] (mean, se)

    ONP, UNI = "#1f77b4", "#d62728"             # on-policy blue, uniform red
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    m, se = offp[:, 0], offp[:, 1]
    # faint glow + bold line + markers (matches the eval-curve style)
    ax.plot(p_levels, m, color=UNI, lw=6, alpha=0.15, solid_capstyle="round", zorder=2)
    ax.plot(p_levels, m, color=UNI, lw=2.5, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=0.8,
            label="uniform mask", zorder=3)
    ax.fill_between(p_levels, m - se, m + se, color=UNI, alpha=0.2, zorder=1)
    ax.axhline(0, color=ONP, ls="--", lw=2, label="on-policy mask (0 by construction)",
               zorder=3)

    ax.set_title("Fraction of masked tokens that are 'easy'")
    ax.set_xlabel("masked fraction p")
    ax.set_ylabel("frac. masked above on-policy frontier")
    ax.set_ylim(-0.04, 1.04)

    ax.grid(True, color="0.85", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor("0.3")

    leg = ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95,
                    borderpad=0.5, handlelength=1.6)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(args.out_dir, f"offpath_fraction.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("[saved]", out)


if __name__ == "__main__":
    main()
