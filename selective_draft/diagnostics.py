"""Research diagnostics on recorded states. All run offline on rollout shards and need no training:
  readout      early-readout reliability (KL to q^0, top-1 agreement, gate coverage, calibration buckets)
  teacher      target signal: KL of {mixture, realized, argmax} targets to q^0, candidate agreement rates,
               and the activation-patch oracle ceiling at the early layer
  consistency  same-step coordination: parallel–sequential agreement, sibling PMI, one-step regret,
               message-induced-commit fidelity
See doc/Selective_Draft_OPSD_Review.md §3–§4 for definitions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import Config
from .generation import _autocast, baseline_block_commitments, choose_tokens, select_commits
from .losses import kl_between_logprobs
from .model import GuidedModel
from .teacher import log_probs_at, oracle_patch_log_probs, reveal_view, teacher_targets
from .trace import iter_states
from .training import state_tensors


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


@torch.no_grad()
def readout_report(gm: GuidedModel, rollouts: List[dict], cfg: Config, max_states: Optional[int] = None,
                   gates=(0.8, 0.9)) -> Dict[str, object]:
    device = gm.device
    per_pos = defaultdict(list)
    per_state = defaultdict(list)
    buckets = defaultdict(lambda: [0, 0])  # maxr bucket -> [agree, count]
    n = 0
    for _, _, roll, step in iter_states(rollouts):
        if max_states is not None and n >= max_states:
            break
        x, attn, ctx, pos = state_tensors(roll, step, device)
        out = gm(x[None], attn[None], context=ctx, mode="off", capture_hidden=True)
        log_q0 = F.log_softmax(out.logits[0, pos].float(), -1)
        log_r = gm.readout_log_probs(out.hidden_early[0, pos])
        kl = kl_between_logprobs(log_q0, log_r)
        maxr, maxq = log_r.exp().max(-1).values, log_q0.exp().max(-1).values
        agree = (log_r.argmax(-1) == log_q0.argmax(-1))
        per_pos["kl"].extend(kl.tolist()); per_pos["agree"].extend(agree.float().tolist())
        per_pos["maxr"].extend(maxr.tolist()); per_pos["maxq"].extend(maxq.tolist())
        for m, a in zip(maxr.tolist(), agree.tolist()):
            b = min(9, int(m * 10)) / 10
            buckets[b][0] += int(a); buckets[b][1] += 1
        src = int(maxr.argmax())
        per_state["source_agree"].append(float(agree[src]))
        per_state["source_maxr"].append(float(maxr[src]))
        per_state["source_final_committable_0.9"].append(float(maxq[src] > 0.9))
        has_recv = pos.numel() >= 2
        for g in gates:
            per_state[f"gate_open_{g}"].append(float(has_recv and maxr[src] >= g))
            per_state[f"gate_open_{g}_and_source_agree"].append(float(has_recv and maxr[src] >= g and agree[src]))
        n += 1
    rep = {"states": n, "positions": len(per_pos["kl"]),
           "kl_mean": _mean(per_pos["kl"]), "top1_agree": _mean(per_pos["agree"]),
           "maxr_mean": _mean(per_pos["maxr"]), "maxq_mean": _mean(per_pos["maxq"])}
    rep.update({k: _mean(v) for k, v in per_state.items()})
    rep["calibration_buckets"] = {str(k): {"agree_rate": v[0] / v[1], "count": v[1]} for k, v in sorted(buckets.items())}
    for g in gates:
        opened = [a for a, o in zip(per_state["source_agree"], per_state[f"gate_open_{g}"]) if o]
        rep[f"source_agree_given_gate_{g}"] = _mean(opened)
    return rep


@torch.no_grad()
def teacher_report(gm: GuidedModel, rollouts: List[dict], cfg: Config, max_states: Optional[int] = None,
                   oracle: bool = True, modes=("realized", "argmax", "mc", "topk")) -> Dict[str, object]:
    device = gm.device
    rng = torch.Generator().manual_seed(cfg.teacher.seed)
    acc = defaultdict(list)
    n, n_gated = 0, 0
    for _, _, roll, step in iter_states(rollouts):
        if len(step["eligible_positions"]) < 2:
            continue
        if max_states is not None and n >= max_states:
            break
        n += 1
        x, attn, ctx, pos = state_tensors(roll, step, device)
        P = roll["meta"]["prompt_len"]
        cap = gm(x[None], attn[None], context=ctx, mode="readout_only")
        gi = cap.guidance
        A = int(gi.source_pos[0])
        gated = bool(gi.gate_open[0])
        if not gated:
            continue  # only gated states produce a message; report coverage separately
        n_gated += 1
        r_A = gi.source_probs[0]
        receivers = pos[pos != A]
        realized = int(roll["final_state"][A - P])
        a32 = int(cap.logits[0, A].argmax())
        v_star = int(r_A.argmax())
        acc["vstar_eq_realized"].append(float(v_star == realized))
        acc["vstar_eq_final_argmax"].append(float(v_star == a32))
        acc["final_argmax_eq_realized"].append(float(a32 == realized))
        acc["source_maxr"].append(float(r_A.max()))
        log_q0 = F.log_softmax(cap.logits[0, receivers].float(), -1)
        tg = {}
        for mode in modes:
            t = teacher_targets(gm, x, attn, A, receivers, r_A, realized, cfg.teacher, rng, mode=mode)
            tg[mode] = t.log_q
            acc[f"kl_{mode}_vs_q0"].append(float(kl_between_logprobs(t.log_q, log_q0).mean()))
            acc[f"forwards_{mode}"].append(float(t.info["teacher_forwards"]))
            if mode == "topk":
                acc["topk_retained_mass"].append(float(t.info["retained_mass"]))
        if "realized" in tg and "mc" in tg:
            acc["kl_realized_vs_mc"].append(float(kl_between_logprobs(tg["realized"], tg["mc"]).mean()))
        if oracle and "realized" in tg:
            log_p_patch, delta = oracle_patch_log_probs(gm, x, attn, ctx, A, realized, receivers)
            kl_full = kl_between_logprobs(tg["realized"], log_q0)
            kl_patch = kl_between_logprobs(tg["realized"], log_p_patch)
            acc["oracle_kl_before"].append(float(kl_full.mean()))
            acc["oracle_kl_after_patch"].append(float(kl_patch.mean()))
            acc["oracle_delta_norm"].append(float(delta.float().norm()))
    rep = {"states_with_receivers": n, "gated_states": n_gated, "gate_coverage": n_gated / max(1, n)}
    rep.update({k: _mean(v) for k, v in acc.items()})
    if acc.get("oracle_kl_before"):
        before, after = sum(acc["oracle_kl_before"]), sum(acc["oracle_kl_after_patch"])
        rep["oracle_recovery_ratio_of_sums"] = 1.0 - after / before if before > 0 else float("nan")
    return rep


@torch.no_grad()
def consistency_report(gm: GuidedModel, rollouts: List[dict], cfg: Config, max_steps: Optional[int] = None,
                       commit_threshold: Optional[float] = None) -> Dict[str, object]:
    """Teacher reveal response and stability, plus an actual guidance-off counterfactual.

    Induced-commit fidelity is supported for greedy threshold traces only. It compares against
    the token eventually committed by an independent off continuation of the same pre-step state,
    even if that takes multiple steps. It is agreement with a baseline, not token correctness.
    """
    device = gm.device
    MASK = gm.backend.mask_token_id
    acc = defaultdict(list)
    fidelity_status = defaultdict(int)
    n = 0
    pair_steps = 0
    for _, _, roll, step in iter_states(rollouts):
        cpos = step["committed_positions"].long()
        if cpos.numel() == 0:
            continue
        if max_steps is not None and n >= max_steps:
            break
        n += 1
        x, attn, ctx, _ = state_tensors(roll, step, device)
        ctoks = step["committed_tokens"].long()
        econf = step["eligible_confidences"].float()
        epos = step["eligible_positions"].long()
        conf_at = {int(p): float(c) for p, c in zip(epos, econf)}
        status = _induced_commit_fidelity(gm, roll, step, x, attn, ctx, cfg, acc, commit_threshold)
        fidelity_status[status] += 1
        if cpos.numel() < 2:
            continue
        pair_steps += 1
        if step["gate_open"] and step.get("source_was_committed", False):
            A = int(step["source_pos"])
        else:
            A = int(cpos[torch.tensor([conf_at[int(p)] for p in cpos]).argmax()])
        a = int(ctoks[(cpos == A).nonzero().flatten()[0]])
        recv_mask = cpos != A
        B_pos = cpos[recv_mask].to(device)
        B_tok = ctoks[recv_mask].to(device)
        log_q0 = log_probs_at(gm, x[None], attn[None], B_pos)[0]
        log_qc = log_probs_at(gm, reveal_view(x, A, a)[None], attn[None], B_pos)[0]
        psa = (log_qc.argmax(-1) == B_tok).float()
        pmi = log_qc.gather(-1, B_tok[:, None]).squeeze(-1) - log_q0.gather(-1, B_tok[:, None]).squeeze(-1)
        acc["psa"].extend(psa.tolist()); acc["pmi"].extend(pmi.tolist())
        acc["pmi_negative"].extend((pmi < 0).float().tolist())
        acc["psa_vs_unconditional"].extend((log_q0.argmax(-1) == B_tok).float().tolist())
        # one-step regret: next state with B re-masked
        nxt = x.clone()
        nxt[cpos.to(device)] = ctoks.to(device)
        regret = []
        for bp, bt in zip(B_pos.tolist(), B_tok.tolist()):
            view = nxt.clone(); view[bp] = MASK
            lp = log_probs_at(gm, view[None], attn[None], torch.tensor([bp], device=device))[0, 0]
            regret.append(float(int(lp.argmax()) != bt))
        acc["regret"].extend(regret)
        acc["committed_per_step"].append(float(cpos.numel()))
    rep = {"steps": pair_steps, "states_examined": n, "receiver_tokens": len(acc["psa"]),
           "induced_fidelity_status": dict(fidelity_status),
           "induced_fidelity_reference": "guidance-off continuation until B actually commits"}
    rep.update({k: _mean(v) for k, v in acc.items()})
    rep["induced_commits"] = len(acc["induced_fidelity"])
    return rep


def _induced_commit_fidelity(gm, roll, step, x, attn, ctx, cfg, acc, threshold_override):
    meta = roll["meta"]
    if meta.get("guidance_mode") != "single_soft" or not step["gate_open"]:
        return "no_guidance"
    if "generation" not in meta or meta.get("confidence_policy") != "raw_model_probability":
        return "missing_protocol_metadata"  # old traces may use renormalised/rounded confidence
    gen = replace(cfg.generation, **meta["generation"])
    if gen.mode != "threshold":
        return "unsupported_fixed_budget"
    if gen.temperature != 0:
        return "unsupported_sampling"  # no unrecorded RNG coupling is invented
    tau = gen.commit_threshold
    if threshold_override is not None and threshold_override != tau:
        raise ValueError("fidelity threshold must match the recorded commit protocol")
    with _autocast(gen, x.device):
        off_logits = gm(x[None], attn[None], mode="off").logits
    _, off_conf = choose_tokens(off_logits, 0.0, None,
                                gm.backend.mask_token_id if gen.forbid_mask_candidate else None)
    off_commit = select_commits(ctx.eligible_mask, off_conf, "threshold", tau)[0]
    guided_conf = {int(p): float(c) for p, c in zip(step["eligible_positions"], step["eligible_confidences"])}
    induced = [(int(p), int(t)) for p, t in zip(step["committed_positions"], step["committed_tokens"])
               if int(p) != int(step["source_pos"]) and guided_conf[int(p)] > tau and not bool(off_commit[int(p)])]
    if induced:
        commits = baseline_block_commitments(
            gm, x[None], attn[None], ctx, gen,
            torch.tensor([p for p, _ in induced], device=x.device), tau, initial_logits=off_logits)
        for pos, token in induced:
            off_token, wait = commits[pos]
            acc["induced_fidelity"].append(float(token == off_token))
            acc["induced_baseline_wait_steps"].append(float(wait))
    return "supported"
