"""Verified self-drafter OPSD (doc/plan_rethink_opsd.md section 4 + option 2 of 2026-11-11). sibling.target = "drafter".

States: the CURRENT student (base + LoRA) decodes each training question at the deployment threshold
tau = generation.commit_threshold (0.95). Attempt 0 is greedy (T = 0, exactly the deployed decoder); attempts 1..K-1
sample at T_train (0.9). The first attempt that is verified correct, ended with EOS and contains a \\boxed answer is
accepted; a question without one gives no gradient (correctness filter). states_per_rollout states S are drawn at
random from the accepted rollout (S = the state before a step, the step's commits still MASK).

On each state, one student forward p (with grad) and:
  candidates  C(S) = in-block MASK positions before the rollout's EOS with max p >= 1/2 (a majority proposal),
              c^_i = argmax p_i. Up to max_positions_per_state of them are supervised; all of C stay revealed.
  LOO verdict frozen base (LoRA off) on S + c^_{C\\i}, position i masked -> q_i. V = {i : argmax q_i = c^_i}.
  throttle    i in V:   [log cap - log p_i(c^_i)]_+             (push an approved proposal to the deploy threshold)
  brake       i in C\\V: KL(t_i || p_i), t_i = q(. | S + c~_{C\\i}, i), c~_j ~ p_j (one sample of the siblings from
              the student: an unbiased one-sample estimate of the sibling-marginalised teacher, the loss is linear in t)
  content     the step's committed tokens c_l of the accepted rollout: [log cap - log p_l(c_l)]_+ . Zero where the
              student was already sure; it carries the tokens of a sampled correct attempt that the greedy path would
              not have produced (the pass@k -> pass@1 part), weight sibling.content_weight (0 = the no-content ablation)
L = mean_V throttle + mean_{C\\V} brake + content_weight * mean content (empty terms are dropped). cap = sibling.path_cap
(the evaluation threshold). KL(p0 || p) on the block is logged every path_kl_every updates (monitor only).
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import Example
from .losses import forward_kl
from .path_opsd import capped_ce, path_steps
from .root_joint import derived_seed
from .sibling_opsd import SiblingOPSDTrainer, kl_from_probs, loo_teacher, pick, sibling_rollout, state_input

CANDIDATE_P = 0.5  # a proposal needs a majority of the student's probability; not tuned


def accepted(roll: dict) -> bool:
    """Strict training verifier: graded correct, ended with EOS (no truncated tail) and has a \\boxed answer."""
    return bool(roll["is_correct"]) and roll["stop_reason"] == "eos" and "\\boxed" in roll.get("generation", "")


class DrafterOPSDTrainer(SiblingOPSDTrainer):
    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.sibling.target != "drafter":
            raise ValueError("DrafterOPSDTrainer needs sibling.target=drafter")
        super().__init__(gm, cfg, out_dir, examples, tokenizer, dist_ctx)
        self.cap = float(cfg.sibling.path_cap)
        self.content_w = float(cfg.sibling.content_weight)
        self._kl_now = False
        self.sib_rng = torch.Generator(device=self.device).manual_seed(cfg.training.seed + 7919 * self.dist.rank)
        self.generators["siblings"] = self.sib_rng
        if self.dist.rank == 0:
            print(f"[drafter] rollout tau={self.tau} greedy first={cfg.sibling.first_attempt_greedy} "
                  f"T={cfg.training.rollout_temperature} K={cfg.training.max_rollout_attempts} cap={self.cap} "
                  f"content_weight={self.content_w} states/question={cfg.sibling.states_per_rollout} "
                  f"rows/state={cfg.sibling.max_positions_per_state}")

    # ---------------------------------------------------------------- rollouts (current student, correctness filter)
    def root_question(self, ex: Example, gidx: int):
        seed, policy = self.cfg.training.seed, int(self.state["optimizer_step"])
        ledger: Dict[str, object] = {"question_index": gidx, "example_id": ex.example_id, "policy_version": policy,
                                     "attempts": []}
        self._bump("question_visits")
        success = None
        for a in range(self.cfg.training.max_rollout_attempts):
            temp = 0.0 if (a == 0 and self.cfg.sibling.first_attempt_greedy) else self.cfg.training.rollout_temperature
            rng = torch.Generator(device=self.device).manual_seed(derived_seed(seed, ex.example_id, policy, a))
            t0 = time.perf_counter()
            roll = sibling_rollout(self.gm, self.tokenizer, ex, self.cfg, rng, temp, {"policy_version": policy,
                                                                                      "attempt": a})
            self._bump("root_attempts")
            st = roll["stats"]
            ok = accepted(roll)
            ledger["attempts"].append({"attempt": a, "temperature": temp, "is_correct": bool(roll["is_correct"]),
                                       "accepted": ok, "nfe": st["nfe"], "steps": int(roll["steps_total"]),
                                       "output_tokens": st["output_tokens"], "tpf_dinfer": st["tpf_dinfer"],
                                       "mean_commit": sum(st["commit_counts"]) / max(1, len(st["commit_counts"])),
                                       "stop_reason": roll["stop_reason"], "seconds": time.perf_counter() - t0})
            if ok:
                success = roll
                break
        if success is None:
            ledger["outcome"] = "root_budget_exhausted"
            return [], ledger
        self._bump("questions_with_correct_rollout")
        steps = path_steps(success)
        self._bump("trace_states", len(success["steps"]))
        self._bump("parallel_states", sum(len(ok) >= 2 for _, ok in steps))
        ledger.update(first_success_attempt=len(ledger["attempts"]) - 1, path_steps=int(success["steps_total"]),
                      path_tpf=float(success["stats"]["tpf_dinfer"]))
        if not steps:
            ledger["outcome"] = "no_supervised_step"
            return [], ledger
        self._bump("questions_with_parallel_state")
        keep = pick(len(steps), self.cfg.sibling.states_per_rollout, derived_seed(seed, ex.example_id, policy, "states"))
        items = [(success, steps[k][0], steps[k][1]) for k in keep]
        ledger.update(outcome="drawn", states_drawn=len(items))
        return items, ledger

    # ---------------------------------------------------------------- loss on one state
    def state_loss(self, roll: dict, step: dict, sup_idx: List[int]):
        dev = self.gm.device
        x = state_input(roll, step, dev)
        P = int(roll["meta"]["prompt_len"])
        eos = roll["eos_position"]
        elig = step["eligible_positions"].to(dev).long()
        if eos is not None:
            elig = elig[elig - P < int(eos)]
        logits = self.gm(x, mode="off").logits[0]  # student, every MASK still MASK
        terms, stats = [], {}

        # content: the accepted rollout's commits at this step
        cpos = step["committed_positions"].to(dev).long()[sup_idx]
        ctok = step["committed_tokens"].to(dev).long()[sup_idx]
        ce = capped_ce(logits[cpos], ctok, self.cap)
        if self.content_w > 0:
            terms.append(self.content_w * ce.mean())

        # candidates and the LOO verdict
        with torch.no_grad():
            lp = F.log_softmax(logits[elig].float(), -1)
            pmax, chat = lp.exp().max(-1)
            is_c = (pmax >= CANDIDATE_P) & (chat != self.mask_id)
        C, chat = elig[is_c], chat[is_c]
        n_c = int(C.numel())
        stats.update(content_ce=float(ce.mean()), n_content=float(len(ctok)), n_cand=float(n_c),
                     n_masked=float(elig.numel()),
                     content_reached_cap=float((ce == 0).float().mean()))
        if n_c:
            sup = torch.tensor(pick(n_c, self.cfg.sibling.max_positions_per_state,
                                    derived_seed(self.cfg.training.seed, roll["meta"]["example_id"],
                                                 int(step["step_id"]), "cand")), dtype=torch.long, device=dev)
            log_q = loo_teacher(self.gm, x, C, chat, sup, self.cfg.training.loo_batch_size)
            agree = log_q.argmax(-1) == chat[sup]
            lp_sup = F.log_softmax(logits[C[sup]].float(), -1)
            p_hat = lp_sup.gather(1, chat[sup][:, None]).squeeze(1)
            thr = (torch.log(torch.tensor(self.cap, device=dev)) - p_hat).clamp_min(0.0)
            if bool(agree.any()):
                terms.append(thr[agree].mean())
            brake = None
            if bool((~agree).any()):
                b_idx = sup[~agree]
                with torch.no_grad():  # one sample of every candidate's token from the student
                    probs = F.softmax(logits[C].detach().float(), -1)
                    probs[:, self.mask_id] = 0.0
                    sampled = torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=self.sib_rng)[:, 0]
                log_t = loo_teacher(self.gm, x, C, sampled, b_idx, self.cfg.training.loo_batch_size)
                brake = kl_from_probs(log_t.exp(), logits[C[b_idx]])
                terms.append(brake.mean())
            with torch.no_grad():
                p_sup = p_hat.exp()
                stats.update(n_sup=float(len(sup)), agree_frac=float(agree.float().mean()),
                             throttle=float(thr[agree].mean()) if bool(agree.any()) else float("nan"),
                             brake_kl=float(brake.mean()) if brake is not None else float("nan"),
                             p_hat_agree=float(p_sup[agree].mean()) if bool(agree.any()) else float("nan"),
                             p_hat_veto=float(p_sup[~agree].mean()) if bool((~agree).any()) else float("nan"),
                             q_chat=float(log_q.exp().gather(1, chat[sup][:, None]).mean()),
                             above_cap_frac=float((p_sup >= self.cap).float().mean()))
        else:
            stats.update(n_sup=0.0)
        if self._kl_now and elig.numel():
            with torch.no_grad():
                base = self.gm.teacher_forward(x)[0]
                stats["kl_p0_block"] = float(forward_kl(F.log_softmax(base[elig].float(), -1), logits[elig]).mean())
                del base
        loss = sum(terms) if terms else logits[cpos].float().sum() * 0.0
        stats["loss_terms"] = float(len(terms))
        return loss, stats

    def update(self) -> Dict[str, float]:
        every = int(self.cfg.sibling.path_kl_every)
        self._kl_now = every > 0 and int(self.state["optimizer_step"]) % every == 0
        return super().update()

    def _onpolicy(self, ledgers) -> Dict[str, float]:
        out = SiblingOPSDTrainer._onpolicy(ledgers)
        first = [l["attempts"][0] for l in ledgers if l.get("attempts")]
        if first:
            out["greedy_accept"] = sum(a["accepted"] for a in first) / len(first)
            out["greedy_truncated"] = sum(a["stop_reason"] == "length" for a in first) / len(first)
            out["greedy_tpf"] = sum(a["tpf_dinfer"] for a in first) / len(first)
            out["solved_frac"] = sum(any(a["accepted"] for a in l["attempts"]) for l in ledgers if l.get("attempts")) \
                / len(first)
        return out
