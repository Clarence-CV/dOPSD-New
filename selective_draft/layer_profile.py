"""Stage-1 diagnostics on a fixed state bank (plan §6.5, §7.1): which layer to read out, and which readout
checkpoint to hand to Joint training. Both decisions follow pre-registered rules; no task accuracy is used.

State bank: one recorded denoising state per rollout, drawn uniformly with a seed derived from the example id,
so every checkpoint and every layer is measured on exactly the same token states. Positions = the current
block's MASK positions of that state. Questions (= rollouts) are the bootstrap unit.

layer_profile: native logit lens r_l = lm_head(norm(h_l)) at every layer, q0 = final distribution.
  stable crossing of a final-confident position i (q0 candidate confidence > tau): first layer l such that for
  every k >= l the layer-k candidate equals the q0 candidate with layer-k probability > tau.
  C_l = per question, share of final-confident positions with crossing <= l (questions with none are reported
  separately); l_best = argmax_{l<L} C_l; l_star = earliest l < L with C_l >= C_best - SE_boot(C_best).
readout_metrics / select_readout: §7.1.1 table and the §7.1.2 gates and one-standard-error choice.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .decoding import choose_tokens
from .model import GuidedModel
from .training import state_tensors


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def pick_states(rollouts: List[dict], tag: str = "state-bank-v1") -> List[Tuple[dict, dict]]:
    """One state per rollout with at least one eligible position, uniform, seeded by the example id."""
    picks = []
    for roll in rollouts:
        steps = [s for s in roll["steps"] if len(s["eligible_positions"]) > 0]
        if not steps:
            continue
        g = torch.Generator().manual_seed(_seed(tag, roll["meta"].get("example_id", "")))
        picks.append((roll, steps[int(torch.randint(len(steps), (1,), generator=g))]))
    return picks


def _bootstrap_se(values: List[float], n_boot: int = 2000, seed: int = 0) -> float:
    if len(values) < 2:
        return float("nan")
    v = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(values), (n_boot, len(values)), generator=g)
    return float(v[idx].mean(1).std())


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _capture_all_layers(gm: GuidedModel, x: torch.Tensor, attn: torch.Tensor, pos: torch.Tensor):
    hidden = {}
    handles = []
    for li, blk in enumerate(gm.backend.blocks, start=1):
        def hook(module, inputs, output, li=li):
            hidden[li] = gm.backend.block_hidden(output)[0, pos].detach()
        handles.append(blk.register_forward_hook(hook))
    try:
        logits = gm.base_forward(x[None], attn[None])
    finally:
        for h in handles:
            h.remove()
    return logits[0, pos], hidden


@torch.no_grad()
def layer_profile(gm: GuidedModel, picks: List[Tuple[dict, dict]], cfg: Config, tau: Optional[float] = None):
    be = gm.backend
    L, MASK, EOS = be.n_layers, be.mask_token_id, be.eos_token_id
    role_end = getattr(be, "role_end_token_id", None)
    tau = float(cfg.generation.commit_threshold if tau is None else tau)
    logV = math.log(be.vocab_size)
    per_layer = {l: {"entropy": [], "maxprob": [], "above_tau": [], "kl": [], "agree": [], "group2": [],
                     "precision": [], "recall": [], "mask_pred": [], "eos_pred": [], "role_end_pred": []}
                 for l in range(1, L + 1)}
    questions = []
    for roll, step in picks:
        x, attn, _, pos = state_tensors(roll, step, gm.device)
        logits, hidden = _capture_all_layers(gm, x, attn, pos)
        log_q0 = F.log_softmax(logits.float(), -1)
        cand0, conf0 = choose_tokens(log_q0, 0.0, None, MASK)
        final_hi = conf0 > tau
        ok = torch.zeros(L + 1, len(pos), dtype=torch.bool, device=gm.device)
        for l in range(1, L + 1):
            log_r = F.log_softmax(be.project(hidden[l]).float(), -1)
            r = log_r.exp()
            cand, conf = choose_tokens(log_r, 0.0, None, MASK)
            hi = conf > tau
            raw_arg = log_r.argmax(-1)
            d = per_layer[l]
            d["entropy"].append(float((-(r * log_r).sum(-1) / logV).mean()))
            d["maxprob"].append(float(conf.mean()))
            d["above_tau"].append(float(hi.float().mean()))
            d["kl"].append(float((log_q0.exp() * (log_q0 - log_r)).sum(-1).mean()))
            d["agree"].append(float((cand == cand0).float().mean()))
            d["group2"].append(float(int(hi.sum()) >= 2))
            if bool(hi.any()):
                d["precision"].append(float((final_hi[hi] & (cand[hi] == cand0[hi])).float().mean()))
            if bool(final_hi.any()):
                d["recall"].append(float((hi[final_hi] & (cand[final_hi] == cand0[final_hi])).float().mean()))
            d["mask_pred"].append(float((raw_arg == MASK).float().mean()))
            d["eos_pred"].append(float((raw_arg == EOS).float().mean()))
            if role_end is not None and role_end < be.vocab_size:
                d["role_end_pred"].append(float((raw_arg == role_end).float().mean()))
            ok[l] = (cand == cand0) & (r.gather(-1, cand0[:, None]).squeeze(-1) > tau)
        # stable crossing: first l with ok[k] for every k >= l (suffix-and), L if only the last layer holds
        suffix = torch.ones(len(pos), dtype=torch.bool, device=gm.device)
        crossing = torch.full((len(pos),), L + 1, dtype=torch.long, device=gm.device)
        for l in range(L, 0, -1):
            suffix &= ok[l]
            crossing = torch.where(suffix, torch.full_like(crossing, l), crossing)
        q = {"example_id": roll["meta"].get("example_id"), "step_id": int(step["step_id"]),
             "block_progress": float(1 - len(pos) / max(1, int(step["block_end"]) - int(step["block_start"]))),
             "n_positions": len(pos), "n_final_confident": int(final_hi.sum()),
             "is_correct": roll.get("is_correct")}
        if bool(final_hi.any()):
            cr = crossing[final_hi]
            q["C"] = {l: float((cr <= l).float().mean()) for l in range(1, L + 1)}
        questions.append(q)
    with_f = [q for q in questions if "C" in q]
    curve = {}
    for l in range(1, L + 1):
        vals = [q["C"][l] for q in with_f]
        curve[l] = {"C": _mean(vals), "C_se": _bootstrap_se(vals, seed=l),
                    **{k: _mean(v) for k, v in per_layer[l].items()}}
    early = [l for l in range(1, L)]
    decision = {"rule": "l_star = earliest l < L with C_l >= max_{l<L} C_l - SE_boot(C_best); SE over questions",
                "tau": tau, "n_states": len(questions), "n_with_final_confident": len(with_f),
                "n_layers": L, "early_layer": None, "status": "no_usable_early_layer"}
    if with_f and early:
        l_best = max(early, key=lambda l: (curve[l]["C"], -l))
        c_best, se = curve[l_best]["C"], curve[l_best]["C_se"]
        se = 0.0 if math.isnan(se) else se
        l_star = min(l for l in early if curve[l]["C"] >= c_best - se)
        decision.update(l_best=l_best, C_best=c_best, C_best_se=se, l_star=l_star, C_star=curve[l_star]["C"],
                        group2_coverage_at_l_star=curve[l_star]["group2"], tail_layers=L - l_star)
        if c_best > 0 and curve[l_star]["group2"] > 0:
            decision.update(early_layer=l_star, status="selected_on_val")
    return {"curve": curve, "questions": questions, "decision": decision}


@torch.no_grad()
def readout_metrics(gm: GuidedModel, picks: List[Tuple[dict, dict]], cfg: Config,
                    tau: Optional[float] = None) -> Dict[str, object]:
    """§7.1.1 for the readout currently loaded in gm, on fixed states. Per-question values kept for paired CIs."""
    be = gm.backend
    MASK, EOS = be.mask_token_id, be.eos_token_id
    tau = float(cfg.generation.commit_threshold if tau is None else tau)
    rows = []
    for roll, step in picks:
        x, attn, ctx, pos = state_tensors(roll, step, gm.device)
        out = gm(x[None], attn[None], context=ctx, mode="off", capture_hidden=True)
        log_q0 = F.log_softmax(out.logits[0, pos].float(), -1)
        log_r = gm.readout_log_probs(out.hidden_early[0, pos])
        cand0, _ = choose_tokens(log_q0, 0.0, None, MASK)
        cand, conf = choose_tokens(log_r, 0.0, None, MASK)
        sel = conf > tau
        kl = (log_q0.exp() * (log_q0 - log_r)).sum(-1)
        q0_at = log_q0.exp().gather(-1, cand[:, None]).squeeze(-1)
        row = {"example_id": roll["meta"].get("example_id"), "is_correct": roll.get("is_correct"),
               "n_positions": len(pos), "kl_all": float(kl.mean()), "agreement_all": float((cand == cand0).float().mean()),
               "n_selected": int(sel.sum()), "has_receivers": len(pos) >= 2, "group_open": int(sel.sum()) >= 2,
               "special_pred": float(((log_r.argmax(-1) == MASK) | (log_r.argmax(-1) == EOS)).float().mean())}
        if bool(sel.any()):
            row.update(kl_selected=float(kl[sel].mean()), agreement_selected=float((cand[sel] == cand0[sel]).float().mean()),
                       fidelity_abs=(conf[sel] - q0_at[sel]).abs().tolist(), fidelity_signed=(conf[sel] - q0_at[sel]).tolist())
        rows.append(row)
    sel_rows = [r for r in rows if r["n_selected"]]
    recv = [r for r in rows if r["has_receivers"]]
    fid = [v for r in sel_rows for v in r["fidelity_abs"]]
    summary = {
        "n_states": len(rows), "tau": tau,
        "KL_all": _mean(r["kl_all"] for r in rows), "KL_all_se": _bootstrap_se([r["kl_all"] for r in rows]),
        "agreement_all": _mean(r["agreement_all"] for r in rows),
        "KL_selected": _mean(r["kl_selected"] for r in sel_rows),
        "agreement_selected": _mean(r["agreement_selected"] for r in sel_rows),
        "group_coverage": _mean(float(r["group_open"]) for r in recv), "group_coverage_denominator": len(recv),
        "selected_positions": sum(r["n_selected"] for r in rows),
        "fidelity_abs_mean": _mean(fid), "fidelity_abs_p90": sorted(fid)[int(0.9 * (len(fid) - 1))] if fid else float("nan"),
        "fidelity_signed_mean": _mean(v for r in sel_rows for v in r["fidelity_signed"]),
        "special_pred": _mean(r["special_pred"] for r in rows),
    }
    return {"summary": summary, "rows": [{k: v for k, v in r.items() if not k.startswith("fidelity")} for r in rows]}


GATES = {"kl_improvement": 0.20, "agreement_selected": 0.90, "group_coverage": 0.50, "selected_positions": 128,
         "fidelity_abs_mean": 0.05}


def _paired_improvement(step0_rows: List[dict], rows: List[dict], n_boot: int = 2000) -> Tuple[float, float]:
    a = {r["example_id"]: r["kl_all"] for r in step0_rows}
    d = [a[r["example_id"]] - r["kl_all"] for r in rows if r["example_id"] in a]
    if not d:
        return float("nan"), float("nan")
    v = torch.tensor(d, dtype=torch.float64)
    idx = torch.randint(len(d), (n_boot, len(d)), generator=torch.Generator().manual_seed(1))
    return float(v.mean()), float(v[idx].mean(1).quantile(0.025))


def select_readout(results: Dict[str, dict], steps: Dict[str, int]) -> Dict[str, object]:
    """results: name -> readout_metrics output (must include 'step0'); steps: name -> optimizer step.
    Gate every checkpoint (§7.1.2), keep the passing ones, take min KL_all, then the earliest checkpoint within one
    bootstrap SE of it; ties -> earlier. step0 may be chosen when it passes every gate except the KL improvement."""
    base = results["step0"]
    k0 = base["summary"]["KL_all"]
    table = {}
    for name, res in results.items():
        s = res["summary"]
        mean_gain, lo = _paired_improvement(base["rows"], res["rows"])
        checks = {"agreement_selected": s["agreement_selected"] >= GATES["agreement_selected"],
                  "group_coverage": s["group_coverage"] >= GATES["group_coverage"]
                  and s["selected_positions"] >= GATES["selected_positions"],
                  "fidelity": s["fidelity_abs_mean"] <= GATES["fidelity_abs_mean"]}
        if name != "step0":
            checks["kl_improvement"] = (k0 - s["KL_all"]) >= GATES["kl_improvement"] * k0 and lo > 0
        table[name] = {"step": steps[name], "KL_all": s["KL_all"], "KL_all_se": s["KL_all_se"],
                       "kl_gain_mean": mean_gain, "kl_gain_ci_low": lo, "checks": checks,
                       "passed": all(bool(v) for v in checks.values())}
    passing = [n for n, t in table.items() if t["passed"]]
    out = {"gates": GATES, "table": table, "selected": None, "status": "no_checkpoint_passed"}
    if passing:
        best = min(passing, key=lambda n: (table[n]["KL_all"], table[n]["step"]))
        se = table[best]["KL_all_se"]
        se = 0.0 if math.isnan(se) else se
        chosen = min((n for n in passing if table[n]["KL_all"] <= table[best]["KL_all"] + se),
                     key=lambda n: table[n]["step"])
        out.update(selected=chosen, best_by_kl=best, status="selected_on_val")
    return out
