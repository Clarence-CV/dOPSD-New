"""
Visualize the toy "off the decoding path" experiment.

Produces offpath_figure.png with four panels:
  (a) Frontier cartoon  -- one rollout's positions sorted by commit-confidence.
        On-policy mask = a clean threshold cut (the low-confidence tail).
        Uniform mask    = scattered across the whole confidence range
                          (it hides easy/early tokens and reveals hard/late ones).
  (b) Off-path penalty  -- matched-position delta-NLL vs p. For positions masked
        under BOTH schemes (same token, same position, only context differs),
        how much worse does the model predict the masked token from the uniform
        context than from the on-policy context. > 0 == off-path hurts.
  (c) Off-path fraction -- fraction of uniform-masked tokens that are actually
        "easy" (above the on-policy frontier), vs on-policy (0 by construction).
  (d) Masked-position confidence at p -- on-policy masks the genuinely-hard
        tokens (low confidence); uniform leaves easy/high-confidence tokens
        masked -- a configuration the model never generates.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, "results.json")) as f:
        R = json.load(f)
    m2 = np.load(os.path.join(args.out_dir, "m2_confidence.npz"))

    p_levels = np.asarray(R["p_levels"])
    delta = np.asarray(R["delta_nll"])          # [P, 2] (mean, se)
    offp = np.asarray(R["offpath_frac"])        # [P, 2]
    ex = R["example"]
    m2_p = R["m2_p"]

    ONP, UNI = "#1f77b4", "#d62728"             # on-policy blue, uniform red
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---------------- (a) frontier cartoon ----------------
    ax = axes[0, 0]
    cc = np.asarray(ex["commit_conf"])[:ex["content_len"]]
    order = np.argsort(-cc)                      # easy (high conf) -> hard (low conf)
    ranks = np.arange(len(cc))
    cc_sorted = cc[order]
    pos_to_rank = {int(p): r for r, p in enumerate(order)}

    ax.plot(ranks, cc_sorted, color="0.6", lw=1.5, zorder=1,
            label="commit confidence (neg-entropy)")

    onp_masked = set(ex["onpolicy_masked"])
    uni_masked = set(ex["uniform_masked"])
    k = len(onp_masked)
    # on-policy masks exactly the hardest tail -> a clean cut at rank len-k
    ax.axvspan(len(cc) - k - 0.5, len(cc) - 0.5, color=ONP, alpha=0.12, zorder=0,
               label="on-policy mask (threshold cut)")
    # uniform-masked positions scattered by their confidence rank
    uni_ranks = [pos_to_rank[p] for p in uni_masked]
    ax.scatter(uni_ranks, cc[[order[r] for r in uni_ranks]], marker="x",
               color=UNI, s=55, lw=2, zorder=3, label="uniform mask (scattered)")

    ax.set_title(f"(a) On-policy = threshold cut, uniform = scatter  (p={m2_p})")
    ax.set_xlabel("position rank  (easy / early  →  hard / late)")
    ax.set_ylabel("commit confidence")
    ax.legend(fontsize=8, loc="lower left")

    # ---------------- (b) matched-position delta-NLL ----------------
    ax = axes[0, 1]
    m, se = delta[:, 0], delta[:, 1]
    ax.plot(p_levels, m, "-o", color=UNI, label="uniform − on-policy")
    ax.fill_between(p_levels, m - se, m + se, color=UNI, alpha=0.2)
    ax.axhline(0, color=ONP, ls="--", lw=1, label="on-policy reference")
    ax.set_title("(b) Off-path penalty: ΔNLL at shared masked positions")
    ax.set_xlabel("masked fraction p")
    ax.set_ylabel("NLL(uniform ctx) − NLL(on-policy ctx)")
    ax.legend(fontsize=8)

    # ---------------- (c) off-path fraction ----------------
    ax = axes[1, 0]
    m, se = offp[:, 0], offp[:, 1]
    ax.plot(p_levels, m, "-o", color=UNI, label="uniform mask")
    ax.fill_between(p_levels, m - se, m + se, color=UNI, alpha=0.2)
    ax.axhline(0, color=ONP, ls="--", lw=1, label="on-policy mask (0 by constr.)")
    ax.set_title("(c) Fraction of masked tokens that are 'easy' (off-path)")
    ax.set_xlabel("masked fraction p")
    ax.set_ylabel("frac. masked above on-policy frontier")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)

    # ---------------- (d) masked-position confidence ----------------
    ax = axes[1, 1]
    onp_c = m2["onpolicy"]
    uni_c = m2["uniform"]
    bins = np.linspace(0, 1, 30)
    ax.hist(onp_c, bins=bins, density=True, alpha=0.55, color=ONP,
            label=f"on-policy (median {np.median(onp_c):.2f})")
    ax.hist(uni_c, bins=bins, density=True, alpha=0.55, color=UNI,
            label=f"uniform (median {np.median(uni_c):.2f})")
    ax.set_title(f"(d) Model confidence at masked positions  (p={m2_p})")
    ax.set_xlabel("max-prob confidence at masked position")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    fig.suptitle("The noise is off the decoding path: uniform re-masking visits "
                 "partial states the dLLM never generates", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "offpath_figure.png")
    fig.savefig(out, dpi=150)
    print("[saved]", out)

    # echo the qualitative example
    qe = os.path.join(args.out_dir, "qualitative_example.txt")
    if os.path.exists(qe):
        print("\n================ qualitative example ================")
        print(open(qe).read())


if __name__ == "__main__":
    main()
