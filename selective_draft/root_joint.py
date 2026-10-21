"""Joint sibling-PI training with the root_success_v2 sampler.

Per rank and round: take the next unvisited training question (one seeded permutation shared by all ranks, no revisits
in the first pass), roll the current Student out from the prompt at T_train up to K times and stop at the first answer
the verifier accepts. From that successful trace draw ONE state uniformly among the states whose own natural draft
formed a group (|G| >= 2, G = readout candidates with raw confidence >= the decoder's tau). The reference tuple is the
trace's own final tokens on G. If the tuple is incompatible with the state, the regenerated G differs from the recorded
one, or any leave-one-out Teacher argmax disagrees, the question is skipped: no redraw, no extra continuation, no other
trace. Otherwise the injector is trained with the full-vocab forward KL to the leave-one-out Teacher distributions, the
Student seeing only S and its own soft drafts.

Failed attempts are kept in the per-question ledger (question_attempts_rank<r>.jsonl), never as training data.
Every RNG is derived from (seed, question id, attempt), so an exact resume only needs the round counter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import Example, tokenize_prompt
from .generation import generate
from .losses import forward_kl
from .model import GuidanceSpec
from .scoring import grade
from .trace import StateRecorder
from .training import BaseTrainer, StateRef, state_tensors

COUNTERS = ("question_visits", "root_attempts", "questions_with_correct_rollout", "questions_with_eligible_state",
            "states_drawn", "invalid_reference_states", "reselect_mismatch_states", "loo_rejected_states",
            "accepted_states", "supervised_positions", "trace_states", "eligible_trace_states")


def derived_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


def reference_tuple(roll: dict, step: dict, positions: torch.Tensor, mask_id: int) -> Tuple[Optional[torch.Tensor], str]:
    """pi = final tokens of the same trajectory on G, or (None, reason). Positions are absolute."""
    P = int(roll["meta"]["prompt_len"])
    final = roll["final_state"].long()
    state = step["state"].long()
    n = state.numel()
    if final.numel() < n:
        return None, "final_shorter_than_state"
    known = state != mask_id
    if not torch.equal(final[:n][known], state[known]):
        return None, "known_token_changed"
    rel = positions.cpu().long() - P
    if bool((rel < 0).any()) or bool((rel >= n).any()):
        return None, "outside_state"
    if not bool((state[rel] == mask_id).all()):
        return None, "group_not_masked"
    if bool((positions.cpu() < step["block_start"]).any()) or bool((positions.cpu() >= step["block_end"]).any()):
        return None, "outside_current_block"
    eos = roll["eos_position"]
    if eos is not None and bool((rel >= int(eos)).any()):
        return None, "at_or_after_eos"  # the valid completion is [P, first EOS)
    pi = final[rel]
    if bool((pi == mask_id).any()):
        return None, "reference_is_mask"
    return pi, "ok"


@torch.no_grad()
def certify_tuple(gm, x, positions, tokens, batch_size=1):
    """Leave target i masked; reveal only its peers in G. Each row is a separate input.
    -> (log_q [n, V] or None when some Teacher argmax disagrees, stats)"""
    if positions.numel() < 2 or positions.numel() != tokens.numel():
        raise ValueError("certification requires at least two paired positions/tokens")
    x = x[0]
    if positions.unique().numel() != positions.numel() or bool((x[positions] != gm.backend.mask_token_id).any()):
        raise ValueError("certification positions must be distinct MASK positions")
    if bool((tokens == gm.backend.mask_token_id).any()) or batch_size < 1:
        raise ValueError("invalid candidate tokens or teacher batch size")
    logqs, stats = [], {"teacher_forwards": 0, "top1": [], "reference_probs": []}
    for start in range(0, len(positions), batch_size):
        pp = positions[start:start + batch_size]
        inputs = x[None].expand(len(pp), -1).clone()
        inputs[:, positions] = tokens[None]
        rows = torch.arange(len(pp), device=x.device)
        inputs[rows, pp] = gm.backend.mask_token_id
        lq = F.log_softmax(gm.base_forward(inputs)[rows, pp].float(), -1)
        if not bool(torch.isfinite(lq).all()):
            raise RuntimeError("non-finite Teacher log probabilities")
        top1 = lq.argmax(-1)
        stats["teacher_forwards"] += 1
        stats["top1"].extend(top1.tolist())
        stats["reference_probs"].extend(lq.exp()[rows, tokens[start:start + len(pp)]].tolist())
        if not torch.equal(top1, tokens[start:start + len(pp)]):
            return None, stats
        logqs.append(lq)
    return torch.cat(logqs), stats


class RootSuccessTrainer(BaseTrainer):
    stage = "joint"

    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.guidance.mode != "group_soft":
            raise ValueError("joint training needs guidance.mode=group_soft")
        super().__init__(gm, cfg, out_dir, dist_ctx)
        self.examples = list(examples)
        if not self.examples:
            raise RuntimeError("no training questions")
        self.tokenizer = tokenizer
        g = torch.Generator().manual_seed(cfg.training.seed)  # identical on every rank
        self.order = torch.randperm(len(self.examples), generator=g).tolist()
        self.data_signature = hashlib.sha256(json.dumps(
            [(e.example_id, e.prompt_text, e.answer_value) for e in self.examples]).encode()).hexdigest()
        self.state.update({"question_round": 0, **{k: 0 for k in COUNTERS}, "busy_seconds": 0.0})
        self.ledger_path = os.path.join(out_dir, f"question_attempts_rank{self.dist.rank}.jsonl")
        self.reference_dir = os.path.join(out_dir, "references")
        os.makedirs(self.reference_dir, exist_ok=True)
        self.tau = float(cfg.generation.commit_threshold)

    @property
    def finished(self) -> bool:
        return int(self.state["question_round"]) * self.dist.world_size >= len(self.order)

    def _bump(self, key: str, by: int = 1) -> None:
        self.state[key] = int(self.state[key]) + by

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def rollout(self, ex: Example, attempt: int, rng: torch.Generator) -> dict:
        """One Student rollout from the prompt at the training temperature, with its full state trace."""
        ids = tokenize_prompt(self.tokenizer, ex.prompt_text, self.device)
        gen_cfg = replace(self.cfg.generation, temperature=self.cfg.training.rollout_temperature)
        rec = StateRecorder(self.gm.backend.mask_token_id, self.gm.backend.eos_token_id, ids.shape[1],
                            {"example_id": ex.example_id, "prompt_ids": ids[0].tolist(), "gold": ex.answer_value,
                             "policy_version": int(self.state["optimizer_step"]), "attempt": attempt})
        res = generate(self.gm, ids, gen_cfg, guidance_mode="group_soft", recorder=rec, sample_rng=rng)
        text = self.tokenizer.decode(res.completion, skip_special_tokens=True)
        g = grade(ex.answer_value, text, ex.task)
        return rec.finalize(res, {"generation": text, "parsed_answer": g["parsed_answer"],
                                  "is_correct": g["is_correct"], "stats": res.stats})

    def root_question(self, ex: Example, gidx: int) -> Tuple[Optional[StateRef], Dict[str, object]]:
        seed, policy = self.cfg.training.seed, int(self.state["optimizer_step"])
        ledger: Dict[str, object] = {"question_index": gidx, "example_id": ex.example_id, "policy_version": policy,
                                     "attempts": []}
        self._bump("question_visits")
        success = None
        for a in range(self.cfg.training.max_rollout_attempts):
            rng = torch.Generator(device=self.device).manual_seed(derived_seed(seed, ex.example_id, 0, a))
            t0 = time.perf_counter()
            roll = self.rollout(ex, a, rng)
            self._bump("root_attempts")
            ledger["attempts"].append({"attempt": a, "is_correct": bool(roll["is_correct"]),
                                       "nfe": roll["stats"]["nfe"], "output_tokens": roll["stats"]["output_tokens"],
                                       "stop_reason": roll["stop_reason"], "seconds": time.perf_counter() - t0,
                                       "generation": roll["generation"]})
            if roll["is_correct"]:
                success = roll
                break
        if success is None:
            ledger["outcome"] = "root_budget_exhausted"
            return None, ledger
        self._bump("questions_with_correct_rollout")
        ledger["first_success_attempt"] = len(ledger["attempts"]) - 1
        steps = success["steps"]
        eligible = [s for s in steps if len(s.get("group_positions", ())) >= 2]
        self._bump("trace_states", len(steps))
        self._bump("eligible_trace_states", len(eligible))
        ledger.update(trace_states=len(steps), eligible_states=len(eligible))
        if not eligible:
            ledger["outcome"] = "no_eligible_state"
            return None, ledger
        self._bump("questions_with_eligible_state")
        pick = int(torch.randint(len(eligible), (1,),
                                 generator=torch.Generator().manual_seed(derived_seed(seed, ex.example_id, "state"))))
        step = eligible[pick]
        self._bump("states_drawn")
        ledger.update(state_step_id=int(step["step_id"]), state_block=int(step["block_id"]),
                      group_size=len(step["group_positions"]))
        pi, why = reference_tuple(success, step, step["group_positions"].long(), self.gm.backend.mask_token_id)
        if pi is None:
            self._bump("invalid_reference_states")
            ledger["outcome"] = f"invalid_reference:{why}"
            return None, ledger
        ledger["outcome"] = "drawn"
        return StateRef(success, step, policy), ledger

    # ------------------------------------------------------------------ loss on the drawn state
    def state_loss(self, ref: StateRef, ledger: Dict[str, object]):
        P = int(ref.roll["meta"]["prompt_len"])
        end = P + int(ref.roll["stats"]["max_new_tokens_effective"])
        x, ctx, elig_pos = state_tensors(ref.roll, ref.step, self.device, commit_threshold=self.tau, end=end)
        with torch.no_grad():
            cap = self.gm(x, context=ctx, mode="group_readout")
        gi = cap.guidance
        native = cap.logits[0]
        del cap
        valid = gi.group_pos[0] >= 0
        positions = gi.group_pos[0, valid]
        if not torch.equal(positions.cpu(), ref.step["group_positions"].long()):
            self._bump("reselect_mismatch_states")
            ledger["outcome"] = "reselect_mismatch"
            return None, {}
        pi, _ = reference_tuple(ref.roll, ref.step, positions, self.gm.backend.mask_token_id)
        pi = pi.to(self.device)
        log_q, cert = certify_tuple(self.gm, x, positions, pi, self.cfg.training.loo_batch_size)
        ledger.update(teacher_forwards=cert["teacher_forwards"],
                      teacher_reference_prob_min=min(cert["reference_probs"]) if cert["reference_probs"] else None)
        probs = gi.group_probs[0, valid]
        top_p, top_v = probs.topk(min(8, probs.shape[-1]), dim=-1)
        torch.save({"policy_version": ref.policy_version, "meta": ref.roll["meta"], "state": ref.step,
                    "group_positions": positions.cpu(), "reference_tokens": pi.cpu(), "certification": cert,
                    "accepted": log_q is not None, "draft_topk_probs": top_p.cpu(), "draft_topk_ids": top_v.cpu()},
                   os.path.join(self.reference_dir, f"q{ledger['question_index']:06d}_rank{self.dist.rank}.pt"))
        if log_q is None:
            self._bump("loo_rejected_states")
            ledger["outcome"] = "loo_rejected"
            return None, {}
        out = self.gm(x, context=ctx, mode="group_soft", spec=GuidanceSpec(gi.group_pos, gi.group_probs))
        kl = forward_kl(log_q, out.logits[0, positions])
        self._bump("accepted_states")
        self._bump("supervised_positions", len(positions))
        ledger["outcome"] = "accepted"
        with torch.no_grad():
            p = out.logits[0, positions].float().softmax(-1)
            conf, pred = p.max(-1)
            stats = {"group_size": float(len(positions)), "kl_sum": float(kl.sum()),
                     "reference_agreement": float((pred == pi).float().mean()),
                     "draft_reference_agreement": float((probs.argmax(-1) == pi).float().mean()),
                     "high_conf_wrong": float(((conf >= self.tau) & (pred != pi)).sum()),
                     "delta_norm": float(out.guidance.delta_norm[0]), "teacher_forwards": float(cert["teacher_forwards"])}
            outside = elig_pos[~torch.isin(elig_pos, positions)]
            if outside.numel():  # positions outside G get no loss but see the injection through the tail
                lq0 = F.log_softmax(native[outside].float(), -1)
                c0, a0 = lq0.exp().max(-1)
                c1, a1 = out.logits[0, outside].float().softmax(-1).max(-1)
                stats.update(outside_kl=float(forward_kl(lq0, out.logits[0, outside]).mean()),
                             outside_commit_flips=float(((c0 >= self.tau) != (c1 >= self.tau)).sum() +
                                                        ((c0 >= self.tau) & (c1 >= self.tau) & (a0 != a1)).sum()))
        return kl.mean(), stats

    # ------------------------------------------------------------------ one round = one optimizer update
    def update(self) -> Dict[str, float]:
        t_start = time.perf_counter()
        rnd = int(self.state["question_round"])
        gidx = rnd * self.dist.world_size + self.dist.rank
        self.state["question_round"] = rnd + 1
        loss, stats, ledger = None, {}, {"question_index": gidx, "outcome": "first_pass_exhausted"}
        if gidx < len(self.order):
            ref, ledger = self.root_question(self.examples[self.order[gidx]], gidx)
            if ref is not None:
                loss, stats = self.state_loss(ref, ledger)
        used, local_loss = 0, 0.0
        if loss is not None:
            loss.backward()
            local_loss, used = float(loss.detach()), 1
        busy = time.perf_counter() - t_start
        self.state["busy_seconds"] = float(self.state["busy_seconds"]) + busy
        ledger["seconds"] = busy
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(ledger, default=str) + "\n")
        global_n = self.dist.sum_scalar(used, self.device)
        rec: Dict[str, float] = {"question_round": rnd, "outcome": ledger["outcome"], "local_states": used,
                                 "round_seconds": busy, **stats}
        for key in ("question_visits", "accepted_states"):
            rec[f"global_{key}"] = self.dist.sum_scalar(int(self.state[key]), self.device)
        if global_n == 0:
            self._skip_update()
            rec["skipped"] = 1
            self._log(rec)
            return rec
        self.state["consecutive_empty"] = 0
        for p in self.params:
            if p.grad is not None:
                p.grad.div_(global_n)
        rec.update(self._apply_update(local_loss, used, global_n))
        rec["loss"] = local_loss if used else None
        self._log(rec)
        return rec

    def _log(self, rec):
        self.log(dict(rec, **{k: self.state[k] for k in COUNTERS}))
