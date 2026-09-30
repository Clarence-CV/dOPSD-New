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
  fig6_coord_gap.png          B: D_S, D_T, coord = D_S - D_T (sibling and matched control), correct vs wrong
  fig7_sibling_vs_control.png C: sibling vs matched non-sibling control, paired (ctrl_ok tokens only)
  fig8_cogain.png             A: pairwise co-gain P(g_i>0 and g_j>0) for co-decoded pairs vs DISTANCE-
                              MATCHED pairs (same rollout and block, same |i-j|, different steps);
                              the within-block shuffle of the all-up rate is kept in the table only
                              (it ignores adjacency, so it is confounded by position)
  fig10_pi_effect_types.png   PI effect per token: sharpen (top1_T=y, g>0), defer (top1_T=y, g<=0),
                              redirect (top1_T!=y); and defer/redirect rate given sibling conflict (D_S<0)
  fig11_decoding_order.png    teacher-vs-student decoding order: top-|C_t| overlap, premature-commit
                              rate (rank_T_norm > 0.5), Spearman of block confidences
  fig9_threshold_parallel.png sibling gap vs |C_t| WITHIN a threshold-decoded run (--thr_records),
                              where |C_t| varies naturally; |C_t|=1 has no siblings and is left out
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
FIELDS = ("g", "delta", "js_sib", "js_pi", "cos_pi_sib", "logp_S", "step", "block", "n_parallel",
          "D_S", "D_T", "coord", "D_S_ctrl", "D_T_ctrl", "coord_ctrl", "ctrl_ok",
          "pos", "token", "top1_T", "rank_S", "rank_T", "rank_T_norm", "order_overlap", "order_spearman")
N_PERM = 500
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


def dot_panel(ax, groups, keys, labels, rows, tag, note="ctrl_ok tokens"):
    """Mean +/- 95% CI per quantity (x) and group (colour), on the ctrl_ok paired subset."""
    for k, name in enumerate(("correct", "wrong")):
        rs = groups[name]
        if not rs:
            continue
        ms, err = [], []
        for key in keys:
            vals = [r[key][r["ctrl_ok"] > 0] for r in rs]
            m, lo, hi = cluster_boot([v for v in vals if v.size], np.mean)
            ms.append(m), err.append((m - lo, hi - m))
            rows.append([tag, key, name, len(rs), note, "mean", m, lo, hi])
        x = np.arange(len(keys)) + (k - 0.5) * 0.18
        ax.errorbar(x, ms, yerr=np.array(err).T, fmt="o", color=COLORS[name], markersize=7, capsize=4,
                    linewidth=2, label=name)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(keys)), labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)


def fig6(groups, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    dot_panel(axes[0], groups, ["D_S", "D_T", "D_S_ctrl", "D_T_ctrl"],
              ["D_S\nsiblings", "D_T\nsiblings", "D_S\ncontrol", "D_T\ncontrol"], rows, "fig6")
    style(axes[0], "", "mean change in log p(y_i)  (nats)")
    axes[0].set_title("What revealed tokens add: student (D_S) vs PI teacher (D_T)", color=INK, fontsize=10, loc="left")
    dot_panel(axes[1], groups, ["coord", "coord_ctrl"], ["siblings", "control"], rows, "fig6")
    style(axes[1], "", "coord = D_S - D_T  (nats)")
    axes[1].set_title("Gap PI already fills", color=INK, fontsize=10, loc="left")
    fig.suptitle("B. Coordination gap (paired subset with a matched control)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_coord_gap.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig7(groups, out, rows):
    pairs = [("D_S", "D_S_ctrl", "D_S: sibling - control"), ("coord", "coord_ctrl", "coord: sibling - control")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (sib, ctrl, label) in zip(axes, pairs):
        names = [n for n in ("correct", "wrong") if groups[n]]
        for k, name in enumerate(names):
            diffs = [(r[sib] - r[ctrl])[r["ctrl_ok"] > 0] for r in groups[name]]
            m, lo, hi = cluster_boot([d for d in diffs if d.size], np.mean)
            rows.append(["fig7", f"{sib} - {ctrl} (paired)", name, len(groups[name]), "ctrl_ok tokens", "mean", m, lo, hi])
            ax.bar(k, m, width=0.5, color=COLORS[name])
            ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, linewidth=1)
            ax.annotate(f"{m:+.4f}", (k, hi), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=9)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xticks(range(len(names)), names)
        ax.margins(y=0.15)
        style(ax, "", f"{label}  (nats)")
    fig.suptitle("C. Same-step siblings vs matched non-sibling future tokens, paired  (> 0: siblings special)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_sibling_vs_control.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def all_up_rate(rolls, shuffle_rng=None):
    """P(every token of a co-decoded group has g > 0); optionally regroup within (rollout, block)."""
    hits = total = 0
    for r in rolls:
        up = r["g"] > 0
        for b in np.unique(r["block"]):
            sel = np.where(r["block"] == b)[0]
            vals = up[sel]
            if shuffle_rng is not None:
                vals = vals[shuffle_rng.permutation(vals.size)]
            steps = r["step"][sel]  # group sizes are preserved; membership is shuffled
            for t in np.unique(steps):
                grp = vals[steps == t]
                if grp.size > 1:
                    hits += bool(grp.all())
                    total += 1
    return hits / total if total else float("nan")


def pair_table(r):
    """Co-decoded pairs of a rollout with their distance-matched candidates.

    Returns a list of (both_up_real, both_up_candidates[np.array]) for every pair (i, j) revealed
    in the same step; candidates are pairs in the same block at the same |pos_i - pos_j| whose
    tokens were revealed in DIFFERENT steps. Pairs without any candidate are dropped.
    """
    up, pos, step, blk = r["g"] > 0, r["pos"].astype(int), r["step"].astype(int), r["block"].astype(int)
    out = []
    for b in np.unique(blk):
        I = np.where(blk == b)[0]
        P, S, U = pos[I], step[I], up[I]
        a, c = np.triu_indices(I.size, k=1)
        d, same, both = np.abs(P[a] - P[c]), S[a] == S[c], U[a] & U[c]
        for k in np.where(same)[0]:
            cand = both[(d == d[k]) & ~same]
            if cand.size:
                out.append((bool(both[k]), cand))
    return out


def fig8(groups, out, rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [n for n in ("correct", "wrong") if groups[n]]
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        rs = groups[name]
        tabs = [pair_table(r) for r in rs]
        pairs = [p for t in tabs for p in t]
        if not pairs:
            continue
        real = float(np.mean([p[0] for p in pairs]))
        expect = float(np.mean([p[1].mean() for p in pairs]))
        null = np.array([np.mean([p[1][rng.integers(p[1].size)] for p in pairs]) for _ in range(N_PERM)])
        pval = (1 + np.sum(null >= real)) / (1 + null.size)
        diffs = []
        for _ in range(N_BOOT // 4):
            sample = [p for i in rng.integers(0, len(tabs), len(tabs)) for p in tabs[i]]
            if sample:
                diffs.append(np.mean([p[0] for p in sample]) - np.mean([p[1].mean() for p in sample]))
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        nlo, nhi = np.percentile(null, [2.5, 97.5])
        note = f"{len(pairs)} co-decoded pairs with a distance match"
        rows.append(["fig8", "pair co-gain P(both g>0) co-decoded", name, len(rs), note, "fraction", real, np.nan, np.nan])
        rows.append(["fig8", "pair co-gain distance-matched null", name, len(rs), f"perm p={pval:.4f}", "fraction",
                     float(null.mean()), nlo, nhi])
        rows.append(["fig8", "pair co-gain real - matched expectation", name, len(rs), "rollout bootstrap CI",
                     "difference", real - expect, dlo, dhi])
        # Reference only: the earlier within-block shuffle of the all-up rate (not distance-matched).
        ref_null = np.array([all_up_rate(rs, np.random.default_rng(sd)) for sd in range(100)])
        rows.append(["fig8", "REF all-up rate (within-block shuffle, adjacency-confounded)", name, len(rs),
                     f"real={all_up_rate(rs):.4f}", "fraction", float(ref_null.mean()), *np.percentile(ref_null, [2.5, 97.5])])
        ax.bar(k - 0.14, real, width=0.26, color=COLORS[name], label=f"{name}: co-decoded pairs")
        ax.bar(k + 0.14, null.mean(), width=0.26, color=COLORS[name], alpha=0.35, label=f"{name}: distance-matched")
        ax.errorbar(k + 0.14, null.mean(), yerr=[[null.mean() - nlo], [nhi - null.mean()]], color=INK,
                    capsize=4, linewidth=1)
        ax.annotate(f"diff {real - expect:+.3f} [{dlo:+.3f}, {dhi:+.3f}]\nperm p={pval:.3f}", (k, max(real, nhi)),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(range(len(names)), names)
    style(ax, "", "P(both tokens of a pair have g > 0)")
    ax.set_title("A. Pairwise co-gain: co-decoded vs distance-matched pairs", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_cogain.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pi_type(r):
    """0 = sharpen (teacher top-1 is y, g>0), 1 = defer (top-1 is y, g<=0), 2 = redirect (top-1 != y)."""
    redirect = r["top1_T"] != r["token"]
    return np.where(redirect, 2, np.where(r["g"] > 0, 0, 1))


def bar_groups(ax, groups, series, rows, tag, ylabel):
    """series: [(label, fn(rollout) -> indicator/value array)]; bars per series x group with CI."""
    names = [n for n in ("correct", "wrong") if groups[n]]
    w = 0.8 / max(len(names), 1)
    top = 0.0
    for k, name in enumerate(names):
        for s_i, (label, fn) in enumerate(series):
            vals = [fn(r) for r in groups[name]]
            vals = [v for v in vals if v.size]
            m, lo, hi = cluster_boot(vals, np.mean) if vals else (np.nan,) * 3
            rows.append([tag, label, name, len(vals), "", "mean", m, lo, hi])
            x = s_i + (k - (len(names) - 1) / 2) * w
            ax.bar(x, m, width=w * 0.9, color=COLORS[name], label=name if s_i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, linewidth=1)
            top = max(top, hi if np.isfinite(hi) else m) if np.isfinite(m) else top
    ax.set_xticks(range(len(series)), [lab for lab, _ in series], fontsize=8)
    style(ax, "", ylabel)
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)  # headroom so the legend never covers a bar
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2, loc="upper center")


def fig10(groups, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    bar_groups(axes[0], groups, [
        ("sharpen", lambda r: (pi_type(r) == 0).astype(float)),
        ("defer", lambda r: (pi_type(r) == 1).astype(float)),
        ("redirect", lambda r: (pi_type(r) == 2).astype(float)),
    ], rows, "fig10", "fraction of tokens")
    axes[0].set_title("What PI does to each revealed token", color=INK, fontsize=10, loc="left")
    multi = lambda r: r["n_parallel"] > 1
    bar_groups(axes[1], groups, [
        ("defer|redirect\nD_S<0 (conflict)", lambda r: (pi_type(r) > 0)[multi(r) & (r["D_S"] < 0)].astype(float)),
        ("defer|redirect\nD_S>=0", lambda r: (pi_type(r) > 0)[multi(r) & (r["D_S"] >= 0)].astype(float)),
    ], rows, "fig10", "fraction of tokens")
    axes[1].set_title("Does PI hold back tokens whose siblings conflict?", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig10_pi_effect_types.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def step_level(r, key):
    """One value per step (step-level quantities are repeated on every token of the step)."""
    _, first = np.unique(r["step"], return_index=True)
    v = r[key][first]
    return v[~np.isnan(v)]


def fig11(groups, out, rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    bar_groups(axes[0], groups, [("overlap", lambda r: step_level(r, "order_overlap"))], rows, "fig11",
               "|C_t & teacher top-|C_t|| / |C_t|")
    axes[0].set_title("Would the teacher reveal the same slots?", color=INK, fontsize=10, loc="left")
    bar_groups(axes[1], groups, [
        ("all", lambda r: (r["rank_T_norm"] > 0.5).astype(float)),
        ("D_S<0", lambda r: (r["rank_T_norm"] > 0.5)[(r["n_parallel"] > 1) & (r["D_S"] < 0)].astype(float)),
        ("D_S>=0", lambda r: (r["rank_T_norm"] > 0.5)[(r["n_parallel"] > 1) & (r["D_S"] >= 0)].astype(float)),
    ], rows, "fig11", "premature commits (teacher rank in lower half)")
    axes[1].set_title("Premature commits", color=INK, fontsize=10, loc="left")
    bar_groups(axes[2], groups, [("spearman", lambda r: step_level(r, "order_spearman"))], rows, "fig11",
               "Spearman(conf_S, conf_T) over block")
    axes[2].set_title("Order agreement", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig11_decoding_order.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


THR_BINS = [(2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 10**6, "9+")]


def fig9(tag, groups, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel in ((axes[0], "js_sib", r"mean $D_{JS}(p_S, p_{sib})$"), (axes[1], "D_S", r"mean $D_S$")):
        for k, name in enumerate(("correct", "wrong")):
            xs, ms, err = [], [], []
            for b, (lo_n, hi_n, lab) in enumerate(THR_BINS):
                sub = [r[key][(r["n_parallel"] >= lo_n) & (r["n_parallel"] <= hi_n)] for r in groups[name]]
                sub = [v for v in sub if v.size]
                if not sub:
                    continue
                m, lo, hi = cluster_boot(sub, np.mean)
                xs.append(b), ms.append(m), err.append((m - lo, hi - m))
                rows.append(["fig9", f"{key} | |C_t| in {lab}", name, len(sub), tag, "mean", m, lo, hi])
            if xs:
                ax.errorbar(np.array(xs) + (k - 0.5) * 0.08, ms, yerr=np.array(err).T, color=COLORS[name],
                            linewidth=2, marker="o", markersize=6, capsize=3, label=name)
        ax.set_xticks(range(len(THR_BINS)), [lab for _, _, lab in THR_BINS])
        style(ax, r"tokens revealed in the same step  $|C_t|$", ylabel)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_title(f"Threshold decoding ({tag}): sibling gap vs natural |C_t| (95% CI)", color=INK,
                      fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig9_threshold_parallel.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", action="append", required=True, help="name=records_dir (first = primary)")
    ap.add_argument("--thr_records", default="", help="name=records_dir of a threshold-decoded run (fig9)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include_special", action="store_true")
    ap.add_argument("--note", default="", help="One-line context written at the top of report.md.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sets, dirs = {}, {}
    for spec in args.records:
        name, d = spec.split("=", 1)
        dirs[name] = d
        sets[name] = load(d, args.include_special)
        print(f"[plot] {name}: {len(sets[name])} rollouts from {d}")
    primary = split(next(iter(sets.values())))
    rows = []
    fig1(primary, args.out, rows)
    fig2(primary, args.out, rows)
    fig3(primary, args.out, rows)
    fig4(sets, args.out, rows)
    fig5(primary, args.out, rows)
    fig6(primary, args.out, rows)
    fig7(primary, args.out, rows)
    fig8(primary, args.out, rows)
    fig10(primary, args.out, rows)
    fig11(primary, args.out, rows)
    if args.thr_records:
        name, d = args.thr_records.split("=", 1)
        fig9(name, split(load(d, args.include_special)), args.out, rows)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "quantity", "group", "n_rollouts", "n_tokens/note", "stat", "value", "ci_lo", "ci_hi"])
        w.writerows([[str(c).replace("\n", " ") if isinstance(c, str) else c for c in r] for r in rows])
    from report_pi_sibling import export_tables, write_report

    if args.thr_records:
        tname, td = args.thr_records.split("=", 1)
        dirs[tname] = td
    runs = export_tables(dirs, args.out)
    write_report(args.out, runs, rows, next(iter(dirs)), note=args.note)
    n_fig = len(glob.glob(os.path.join(args.out, "fig*.png")))
    print(f"[plot] wrote {n_fig} figures, summary.csv, report.md, tokens/steps/rollouts tables to {args.out}")


if __name__ == "__main__":
    main()
