"""Figures + summary table for pi_sibling_analysis records.

    python analysis/plot_pi_sibling.py --out FIG_DIR \\
        --records s128=/path/records_steps128 [--records s64=... --records s32=...]

The FIRST --records set is the primary run (figures 1-3, 5); figure 4 pools every set
(one per steps setting -> one |C_t| value each). Tokens that are special (EOS/EOT)
or after the answer end are excluded unless --include_special.

Confidence intervals are 95% cluster-bootstrap over ROLLOUTS (tokens within a rollout
are correlated, so a token-level CI would be overconfident).

  fig1_pi_gain.png            g = log p_T(y) - log p_S(y), correct vs wrong (ECDF)
  fig2_sibling_gap.png        delta = log p_sib(y) - log p_S(y) and JS(p_S, p_sib), correct vs wrong
  fig3_sibling_vs_step.png    sibling gap vs denoising step
  fig4_sibling_vs_parallel.png sibling gap vs |C_t|
  fig5_cos_pi_sib.png         cos(p_T - p_S, p_sib - p_S), correct vs wrong
  summary.csv                 every plotted number (table view)
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical slots 1-2 (validated: CVD dE 24.7, normal dE 33.6 on #fcfcfb).
COLORS = {"correct": "#2a78d6", "wrong": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#1f1f1e", "#5c5b55", "#e4e3dd", "#fcfcfb"
FIELDS = ("g", "delta", "js_sib", "js_pi", "cos_pi_sib", "logp_S", "step", "n_parallel")
N_BOOT = 2000


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def load(rec_dir, include_special):
    """-> list of per-rollout dicts of numpy arrays (filtered tokens) + 'correct'."""
    rolls = []
    for f in sorted(glob.glob(os.path.join(rec_dir, "*.pt"))):
        r = torch.load(f, weights_only=False)
        if r["g"].numel() == 0:
            continue
        keep = torch.ones_like(r["g"], dtype=torch.bool)
        if not include_special:
            keep &= ~r["is_special"].bool() & ~r["after_answer"].bool()
        d = {k: r[k][keep].float().numpy() for k in FIELDS}
        d["correct"] = r["meta"]["correct"]
        if d["g"].size:
            rolls.append(d)
    return rolls


def cluster_boot(rolls, fn, seed=0):
    """Token-level statistic fn(values) with a 95% CI from resampling rollouts."""
    if not rolls:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = fn(np.concatenate(rolls))
    boots = [fn(np.concatenate([rolls[i] for i in rng.integers(0, len(rolls), len(rolls))]))
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def split(rolls):
    return {"correct": [r for r in rolls if r["correct"]], "wrong": [r for r in rolls if not r["correct"]]}


def ecdf_panel(ax, groups, key, xlabel, rows, tag):
    for name, rs in groups.items():
        if not rs:
            continue
        v = np.sort(np.concatenate([r[key] for r in rs]))
        y = np.arange(1, v.size + 1) / v.size
        ax.plot(v, y, color=COLORS[name], linewidth=2, label=f"{name} ({len(rs)} rollouts)")
        m, lo, hi = cluster_boot([r[key] for r in rs], np.mean)
        ax.axvline(m, color=COLORS[name], linewidth=1, linestyle=(0, (3, 3)))
        rows.append([tag, key, name, len(rs), sum(r[key].size for r in rs), "mean", m, lo, hi])
    lo_, hi_ = np.percentile(np.concatenate([r[key] for rs in groups.values() for r in rs]), [0.5, 99.5])
    ax.set_xlim(lo_, hi_)
    style(ax, xlabel, "cumulative fraction of tokens")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)


def fig1(groups, out, rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ecdf_panel(ax, groups, "g", r"PI gain  $g_i=\log p_T(y_i)-\log p_S(y_i)$  (nats)", rows, "fig1")
    ax.set_title("PI gain on the rollout's own tokens (dashed = mean)", color=INK, fontsize=11, loc="left")
    # Hypothesis metric: student lukewarm on y (p_S < 0.5) but PI lifts it (g > 0.5 nat).
    for name, rs in groups.items():
        m, lo, hi = cluster_boot([np.stack([r["logp_S"], r["g"]], 1) for r in rs],
                                 lambda a: float(np.mean((np.exp(a[:, 0]) < 0.5) & (a[:, 1] > 0.5))))
        rows.append(["fig1", "frac(p_S<0.5 & g>0.5)", name, len(rs), "", "fraction", m, lo, hi])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_pi_gain.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig2(groups, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ecdf_panel(axes[0], groups, "delta", r"sibling gain  $\Delta_i=\log p_{sib}(y_i)-\log p_S(y_i)$", rows, "fig2")
    ecdf_panel(axes[1], groups, "js_sib", r"$D_{JS}(p_S, p_{sib})$  (nats)", rows, "fig2")
    axes[0].set_title("Same-step sibling gap (dashed = mean)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_sibling_gap.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def band_panel(ax, groups, key, xkey, bins, xlabel, ylabel, rows, tag):
    centers = [(a + b - 1) / 2 for a, b in zip(bins[:-1], bins[1:])]
    for name, rs in groups.items():
        m_, lo_, hi_ = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sub = [r[key][(r[xkey] >= a) & (r[xkey] < b)] for r in rs]
            sub = [s for s in sub if s.size]
            m, lo, hi = cluster_boot(sub, np.mean) if sub else (np.nan,) * 3
            m_.append(m), lo_.append(lo), hi_.append(hi)
            rows.append([tag, f"{key} | {xkey} in [{a},{b})", name, len(sub), "", "mean", m, lo, hi])
        ax.fill_between(centers, lo_, hi_, color=COLORS[name], alpha=0.18, linewidth=0)
        ax.plot(centers, m_, color=COLORS[name], linewidth=2, marker="o", markersize=4, label=name)
    style(ax, xlabel, ylabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)


def fig3(groups, out, rows, width=8):
    max_step = int(max(r["step"].max() for rs in groups.values() for r in rs)) + 1
    bins = list(range(0, max_step + width, width))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    band_panel(axes[0], groups, "js_sib", "step", bins, "denoising step t", r"mean $D_{JS}(p_S, p_{sib})$", rows, "fig3")
    band_panel(axes[1], groups, "delta", "step", bins, "denoising step t", r"mean $\Delta_i$", rows, "fig3")
    axes[0].set_title(f"Sibling gap vs step ({width}-step bins, band = 95% CI)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_sibling_vs_step.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig4(sets, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel in ((axes[0], "js_sib", r"mean $D_{JS}(p_S, p_{sib})$"), (axes[1], "delta", r"mean $\Delta_i$")):
        for k, name in enumerate(("correct", "wrong")):
            xs, ms, err = [], [], []
            for tag, rolls in sets.items():
                rs = split(rolls)[name]
                if not rs:
                    continue
                npar = int(np.median(np.concatenate([r["n_parallel"] for r in rs])))
                m, lo, hi = cluster_boot([r[key] for r in rs], np.mean)
                xs.append(npar), ms.append(m), err.append((m - lo, hi - m))
                rows.append(["fig4", key, name, len(rs), f"{tag} |C_t|={npar}", "mean", m, lo, hi])
            order = np.argsort(xs)
            x = np.log2(np.array(xs)[order]) + (k - 0.5) * 0.08
            ax.errorbar(x, np.array(ms)[order], yerr=np.array(err)[order].T, color=COLORS[name], linewidth=2,
                        marker="o", markersize=6, capsize=3, label=name)
        ticks = sorted({int(np.median(np.concatenate([r["n_parallel"] for r in rl]))) for rl in sets.values()})
        ax.set_xticks(np.log2(ticks), [str(t) for t in ticks])
        style(ax, r"tokens revealed per step  $|C_t|$", ylabel)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_title("Sibling gap vs parallel reveal count (95% CI)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_sibling_vs_parallel.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig5(groups, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ecdf_panel(axes[0], groups, "cos_pi_sib", r"$\cos(p_T-p_S,\ p_{sib}-p_S)$", rows, "fig5")
    axes[0].axvline(0, color=INK_2, linewidth=0.8)
    names = [n for n in ("correct", "wrong") if groups[n]]
    for k, name in enumerate(names):
        m, lo, hi = cluster_boot([r["cos_pi_sib"] for r in groups[name]], lambda a: float(np.mean(a > 0)))
        rows.append(["fig5", "frac(cos>0)", name, len(groups[name]), "", "fraction", m, lo, hi])
        axes[1].bar(k, m, width=0.5, color=COLORS[name])
        axes[1].errorbar(k, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, linewidth=1)
        axes[1].text(k, hi + 0.01, f"{m:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    axes[1].axhline(0.5, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[1].set_xticks(range(len(names)), names)
    style(axes[1], "", "fraction of tokens with cos > 0")
    axes[0].set_title("Do the PI shift and the sibling shift agree?", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_cos_pi_sib.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", action="append", required=True, help="name=records_dir (first = primary)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include_special", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sets = {}
    for spec in args.records:
        name, d = spec.split("=", 1)
        sets[name] = load(d, args.include_special)
        print(f"[plot] {name}: {len(sets[name])} rollouts from {d}")
    primary = split(next(iter(sets.values())))
    rows = []
    fig1(primary, args.out, rows)
    fig2(primary, args.out, rows)
    fig3(primary, args.out, rows)
    fig4(sets, args.out, rows)
    fig5(primary, args.out, rows)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "quantity", "group", "n_rollouts", "n_tokens/note", "stat", "value", "ci_lo", "ci_hi"])
        w.writerows(rows)
    print(f"[plot] wrote 5 figures + summary.csv to {args.out}")


if __name__ == "__main__":
    main()
