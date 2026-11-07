"""Hybrid Path OPSD, basic version (doc/plan_path_opsd.md). sibling.target = "path".

Teacher: the current model (base + LoRA) decodes each question at the aggressive threshold tau
(generation.commit_threshold, 0.5) and T_train (0.9), up to K attempts. sibling.path_select = "shortest" runs every
attempt and keeps the verified-correct rollout with the fewest decoding steps; "first" stops at the first correct one.

Student: the same model, teacher-forced on that path. For a step t with state S_{t-1} (prompt + everything committed
before t) and committed tokens c_l (l in C_t, EOS / post-EOS excluded):
    L = mean_l max(0, -log p(c_l | S_{t-1}) - (-log path_cap))
i.e. cross-entropy that stops once p(c_l) >= path_cap (= the evaluation threshold 0.95: then the evaluation decoder
commits the whole group in one step). States are drawn at random from the path (states_per_rollout per question);
they are fixed Teacher states, so their order does not matter. No LOO, no anchor term: KL(p0 || p) on the drawn
states' block MASK positions is logged every path_kl_every updates as a drift monitor only.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .data import Example
from .losses import forward_kl
from .root_joint import derived_seed
from .sibling_opsd import SiblingOPSDTrainer, pick, sibling_rollout, state_input


def path_steps(roll: dict) -> List[Tuple[dict, List[int]]]:
    """Every step with at least one committed token before the first EOS -> [(step, supervised indices)]."""
    P, eos_id = int(roll["meta"]["prompt_len"]), int(roll["meta"]["eos_id"])
    eos = roll["eos_position"]
    out = []
    for step in roll["steps"]:
        pos, tok = step["committed_positions"].long(), step["committed_tokens"].long()
        ok = [j for j in range(pos.numel())
              if int(tok[j]) != eos_id and (eos is None or int(pos[j]) - P < int(eos))]
        if ok:
            out.append((step, ok))
    return out


def capped_ce(logits: torch.Tensor, tok: torch.Tensor, cap: float) -> torch.Tensor:
    """Per row max(0, -log p(tok) + log cap); zero gradient once p(tok) >= cap."""
    nll = -F.log_softmax(logits.float(), -1).gather(1, tok[:, None]).squeeze(1)
    return (nll + math.log(cap)).clamp_min(0.0)


class PathOPSDTrainer(SiblingOPSDTrainer):
    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.sibling.target != "path":
            raise ValueError("PathOPSDTrainer needs sibling.target=path")
        super().__init__(gm, cfg, out_dir, examples, tokenizer, dist_ctx)
        self.cap = float(cfg.sibling.path_cap)
        self._kl_now = False
        if self.dist.rank == 0:
            print(f"[path] teacher tau={self.tau} T={cfg.training.rollout_temperature} K={cfg.training.max_rollout_attempts} "
                  f"select={cfg.sibling.path_select} cap={self.cap} states/question={cfg.sibling.states_per_rollout}")

    def root_question(self, ex: Example, gidx: int):
        seed, policy = self.cfg.training.seed, int(self.state["optimizer_step"])
        shortest = self.cfg.sibling.path_select == "shortest"
        ledger: Dict[str, object] = {"question_index": gidx, "example_id": ex.example_id, "policy_version": policy,
                                     "attempts": []}
        self._bump("question_visits")
        best = None
        for a in range(self.cfg.training.max_rollout_attempts):
            rng = torch.Generator(device=self.device).manual_seed(derived_seed(seed, ex.example_id, policy, a))
            t0 = time.perf_counter()
            roll = sibling_rollout(self.gm, self.tokenizer, ex, self.cfg, rng, self.cfg.training.rollout_temperature,
                                   {"policy_version": policy, "attempt": a})
            self._bump("root_attempts")
            st = roll["stats"]
            ledger["attempts"].append({"attempt": a, "is_correct": bool(roll["is_correct"]), "nfe": st["nfe"],
                                       "steps": int(roll["steps_total"]), "output_tokens": st["output_tokens"],
                                       "tpf_dinfer": st["tpf_dinfer"],
                                       "mean_commit": sum(st["commit_counts"]) / max(1, len(st["commit_counts"])),
                                       "stop_reason": roll["stop_reason"], "seconds": time.perf_counter() - t0})
            if roll["is_correct"]:
                if best is None or int(roll["steps_total"]) < int(best["steps_total"]):
                    best = roll
                if not shortest:
                    break
        if best is None:
            ledger["outcome"] = "root_budget_exhausted"
            return [], ledger
        self._bump("questions_with_correct_rollout")
        steps = path_steps(best)
        self._bump("trace_states", len(best["steps"]))
        self._bump("parallel_states", sum(len(ok) >= 2 for _, ok in steps))
        ok_steps = [a["steps"] for a in ledger["attempts"] if a["is_correct"]]
        ledger.update(path_attempt=int(best["meta"]["attempt"]), path_steps=int(best["steps_total"]),
                      path_tpf=float(best["stats"]["tpf_dinfer"]), correct_attempts=len(ok_steps),
                      correct_steps_mean=sum(ok_steps) / len(ok_steps), trace_states=len(best["steps"]))
        if not steps:
            ledger["outcome"] = "no_supervised_step"
            return [], ledger
        self._bump("questions_with_parallel_state")
        keep = pick(len(steps), self.cfg.sibling.states_per_rollout, derived_seed(seed, ex.example_id, policy, "states"))
        items = [(best, steps[k][0], steps[k][1]) for k in keep]
        ledger["outcome"] = "drawn"
        ledger["states_drawn"] = len(items)
        return items, ledger

    def state_loss(self, roll: dict, step: dict, sup_idx: List[int]):
        dev = self.gm.device
        x = state_input(roll, step, dev)
        cpos = step["committed_positions"].to(dev).long()
        ctok = step["committed_tokens"].to(dev).long()
        sup = torch.tensor(sup_idx, dtype=torch.long, device=dev)
        spos, stok = cpos[sup], ctok[sup]
        logits = self.gm(x, mode="off").logits[0]
        ce = capped_ce(logits[spos], stok, self.cap)
        loss = ce.mean()
        with torch.no_grad():
            p_c = F.softmax(logits[spos].float(), -1).gather(1, stok[:, None]).squeeze(1)
            stats = {"capped_ce": float(loss), "n_sup": float(len(stok)), "group_size": float(cpos.numel()),
                     "student_p_c": float(p_c.mean()), "reached_cap_frac": float((p_c >= self.cap).mean(dtype=torch.float)),
                     "student_argmax_c_frac": float((logits[spos].argmax(-1) == stok).float().mean())}
            if len(stok) >= 2:
                stats["reached_cap_frac_group"] = stats["reached_cap_frac"]
            if self._kl_now:
                block = step["eligible_positions"].to(dev).long()  # every MASK of the current block, C_t included
                base = self.gm.teacher_forward(x)[0]
                stats["kl_p0_block"] = float(forward_kl(F.log_softmax(base[block].float(), -1), logits[block]).mean())
                stats["p0_c"] = float(F.softmax(base[spos].float(), -1).gather(1, stok[:, None]).mean())
                del base
        return loss, stats

    def update(self) -> Dict[str, float]:
        every = int(self.cfg.sibling.path_kl_every)
        self._kl_now = every > 0 and int(self.state["optimizer_step"]) % every == 0
        return super().update()

    def _onpolicy(self, ledgers) -> Dict[str, float]:
        out = SiblingOPSDTrainer._onpolicy(ledgers)
        got = [l for l in ledgers if "path_steps" in l]
        if got:
            out.update(path_steps=sum(l["path_steps"] for l in got) / len(got),
                       path_tpf=sum(l["path_tpf"] for l in got) / len(got),
                       correct_attempts=sum(l["correct_attempts"] for l in got) / len(got),
                       correct_steps_mean=sum(l["correct_steps_mean"] for l in got) / len(got))
        return out
