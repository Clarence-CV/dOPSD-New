"""Hybrid Path OPSD, basic version (doc/plan_path_opsd.md). sibling.target = "path".

Teacher: by default the frozen pretrained model (sibling.path_teacher = "frozen": LoRA off, rollout seeds independent
of the training step, so the paths are a fixed dataset that could equally be precomputed once) decodes each question at
the aggressive threshold tau (generation.commit_threshold, 0.5) and T_train (0.9), up to K attempts. A path is accepted
only if it is verified correct AND ended with EOS: a truncated rollout has no EOS, so its whole looping tail would be
supervised (this, with path_teacher = "current", collapsed run path_v1; results/path_opsd/path_v1_collapse.md).
sibling.path_select = "shortest" runs every attempt and keeps the accepted rollout with the fewest decoding steps;
"first" stops at the first accepted one.

Offline paths (sibling.path_source = directory): `cli collect-paths` samples the frozen-teacher paths once for the
training questions and stores them as compact records (prompt, per-step commits, final state; states are rebuilt on
demand). The trainer then iterates only over questions that have a path: no rollouts, no empty updates, and every
ablation (cap, selection, filters) reuses the same paths. With path_source empty the paths are sampled online with the
same seeds, so the two modes see identical teacher paths.

Student: the same model, teacher-forced on that path. For a step t with state S_{t-1} (prompt + everything committed
before t) and committed tokens c_l (l in C_t, EOS / post-EOS excluded):
    L = mean_l max(0, -log p(c_l | S_{t-1}) - (-log path_cap))
i.e. cross-entropy that stops once p(c_l) >= path_cap (= the evaluation threshold 0.95: then the evaluation decoder
commits the whole group in one step). States are drawn at random from the path (states_per_rollout per question);
they are fixed Teacher states, so their order does not matter. No LOO, no anchor term: KL(p0 || p) on the drawn
states' block MASK positions is logged every path_kl_every updates as a drift monitor only.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import Example
from .losses import forward_kl
from .root_joint import derived_seed
from .sibling_opsd import SiblingOPSDTrainer, pick, sibling_rollout, state_input

PATHS_SCHEMA = 1


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


# ----------------------------------------------------------------------------- teacher paths: sampling and storage
def attempt_record(roll: dict, attempt: int, seconds: float) -> dict:
    st = roll["stats"]
    return {"attempt": attempt, "is_correct": bool(roll["is_correct"]),
            "accepted": bool(roll["is_correct"]) and roll["stop_reason"] == "eos",
            "nfe": st["nfe"], "steps": int(roll["steps_total"]), "output_tokens": st["output_tokens"],
            "tpf_dinfer": st["tpf_dinfer"], "mean_commit": sum(st["commit_counts"]) / max(1, len(st["commit_counts"])),
            "stop_reason": roll["stop_reason"], "seconds": seconds}


def sample_path(gm, tokenizer, ex: Example, cfg, device, teacher_policy: int, shortest: bool, lora_off: bool,
                meta: Optional[dict] = None) -> Tuple[Optional[dict], List[dict]]:
    """Up to K rollouts of the Teacher at (tau, T_train); -> (accepted path or None, attempt records).
    Accepted = verified correct and ended with EOS. Seeds: derived_seed(seed, example_id, teacher_policy, attempt)."""
    seed = cfg.training.seed
    lora_ctx = gm.lora.disabled if (lora_off and gm.lora is not None) else nullcontext
    best, attempts = None, []
    for a in range(cfg.training.max_rollout_attempts):
        rng = torch.Generator(device=device).manual_seed(derived_seed(seed, ex.example_id, teacher_policy, a))
        t0 = time.perf_counter()
        with lora_ctx():
            roll = sibling_rollout(gm, tokenizer, ex, cfg, rng, cfg.training.rollout_temperature,
                                   dict(meta or {}, attempt=a))
        rec = attempt_record(roll, a, time.perf_counter() - t0)
        attempts.append(rec)
        if rec["accepted"]:
            if best is None or int(roll["steps_total"]) < int(best["steps_total"]):
                best = roll
            if not shortest:
                break
    return best, attempts


def compact_path(roll: dict) -> dict:
    """Accepted rollout -> small record: prompt, per-step commits, final state (no per-step states / confidences)."""
    steps = [{"step_id": int(s["step_id"]), "block_start": int(s["block_start"]), "block_end": int(s["block_end"]),
              "committed_positions": s["committed_positions"].to(torch.int32).clone(),
              "committed_tokens": s["committed_tokens"].to(torch.int32).clone()} for s in roll["steps"]]
    m = roll["meta"]
    return {"example_id": m["example_id"], "prompt_ids": list(m["prompt_ids"]), "prompt_len": int(m["prompt_len"]),
            "mask_id": int(m["mask_id"]), "eos_id": int(m["eos_id"]), "eos_position": roll["eos_position"],
            "stop_reason": roll["stop_reason"], "steps_total": int(roll["steps_total"]),
            "final_state": roll["final_state"].to(torch.int32).clone(), "steps": steps,
            "attempt": int(m.get("attempt", 0)), "generation": roll.get("generation", ""),
            "stats": {k: roll["stats"][k] for k in ("nfe", "output_tokens", "tpf_dinfer")}}


def compact_supervised(rec: dict) -> List[Tuple[int, List[int]]]:
    """Steps of a compact record with >= 1 supervised commit -> [(step index, supervised indices)] (as path_steps)."""
    P, eos_id, eos = rec["prompt_len"], rec["eos_id"], rec["eos_position"]
    out = []
    for k, s in enumerate(rec["steps"]):
        pos, tok = s["committed_positions"].long(), s["committed_tokens"].long()
        ok = [j for j in range(pos.numel())
              if int(tok[j]) != eos_id and (eos is None or int(pos[j]) - P < int(eos))]
        if ok:
            out.append((k, ok))
    return out


def expand_path(rec: dict, k: int) -> Tuple[dict, dict]:
    """Rebuild (roll, step) for step k of a compact record, in the format state_loss / state_input expect:
    The generator extends x one block at a time, so the state at step k spans [P, block_end of step k): MASK
    everywhere except the commits of steps < k; eligible = the MASK positions of step k's block."""
    P, mask_id = rec["prompt_len"], rec["mask_id"]
    s = rec["steps"][k]
    state = torch.full((int(s["block_end"]) - P,), mask_id, dtype=torch.int32)
    for prev in rec["steps"][:k]:
        state[prev["committed_positions"].long() - P] = prev["committed_tokens"].to(torch.int32)
    blk = torch.arange(s["block_start"], s["block_end"])
    eligible = blk[state[blk - P] == mask_id].to(torch.int32)
    step = {"step_id": s["step_id"], "block_id": -1, "block_start": s["block_start"], "block_end": s["block_end"],
            "state": state, "eligible_positions": eligible,
            "committed_positions": s["committed_positions"], "committed_tokens": s["committed_tokens"]}
    roll = {"meta": {"example_id": rec["example_id"], "prompt_ids": rec["prompt_ids"], "prompt_len": P,
                     "mask_id": mask_id, "eos_id": rec["eos_id"], "attempt": rec["attempt"]},
            "eos_position": rec["eos_position"], "stop_reason": rec["stop_reason"],
            "steps_total": rec["steps_total"], "final_state": rec["final_state"]}
    return roll, step


def save_paths(path: str, records: List[dict], ledgers: List[dict]) -> None:
    tmp = path + ".tmp"
    torch.save({"schema": PATHS_SCHEMA, "records": records, "ledgers": ledgers}, tmp)
    os.replace(tmp, path)


def load_paths(source: str) -> Tuple[Dict[str, dict], List[dict]]:
    """Every shard under `source` -> ({example_id: record}, ledgers of every question tried)."""
    files = sorted(glob.glob(os.path.join(source, "paths*.pt")))
    if not files:
        raise FileNotFoundError(f"no paths*.pt under {source}")
    records, ledgers = {}, []
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if blob.get("schema") != PATHS_SCHEMA:
            raise ValueError(f"{f}: paths schema {blob.get('schema')} != {PATHS_SCHEMA}")
        for r in blob["records"]:
            records[r["example_id"]] = r
        ledgers.extend(blob["ledgers"])
    return records, ledgers


def question_order(examples: List[Example], seed: int) -> List[int]:
    """The trainer's fixed question order (identical on every rank and in collect-paths)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(len(examples), generator=g).tolist()


def collect_paths(gm, tokenizer, cfg, examples: List[Example], out_dir: str, shard: Tuple[int, int],
                  limit: Optional[int] = None, flush_every: int = 10) -> dict:
    """Frozen-teacher paths for the first `limit` questions of the trainer's order, every shard[1]-th starting at
    shard[0] -> out_dir/paths.shard<i>of<n>.pt (resumable: already stored questions are skipped)."""
    device = gm.device
    order = question_order(examples, cfg.training.seed)
    if limit:
        order = order[:limit]
    mine = order[shard[0]::shard[1]]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"paths.shard{shard[0]}of{shard[1]}.pt")
    records, ledgers = [], []
    if os.path.exists(out):
        blob = torch.load(out, map_location="cpu", weights_only=False)
        records, ledgers = list(blob["records"]), list(blob["ledgers"])
    done = {l["example_id"] for l in ledgers}
    shortest = cfg.sibling.path_select == "shortest"
    t_start, n_new = time.perf_counter(), 0
    for i, idx in enumerate(mine):
        ex = examples[idx]
        if ex.example_id in done:
            continue
        best, attempts = sample_path(gm, tokenizer, ex, cfg, device, 0, shortest, True,
                                     {"policy_version": 0, "teacher": "frozen"})
        ledger = {"question_index": int(idx), "example_id": ex.example_id, "attempts": attempts,
                  "outcome": "accepted" if best is not None else "root_budget_exhausted"}
        if best is not None:
            rec = compact_path(best)
            ledger.update(path_attempt=rec["attempt"], path_steps=rec["steps_total"],
                          path_tpf=rec["stats"]["tpf_dinfer"], output_tokens=rec["stats"]["output_tokens"])
            records.append(rec)
        ledgers.append(ledger)
        n_new += 1
        if n_new % flush_every == 0:
            save_paths(out, records, ledgers)
            print(json.dumps({"shard": list(shard), "done": len(ledgers), "of": len(mine), "accepted": len(records),
                              "elapsed_s": round(time.perf_counter() - t_start, 1)}), flush=True)
    save_paths(out, records, ledgers)
    summary = {"shard": list(shard), "questions": len(ledgers), "accepted": len(records),
               "attempts": sum(len(l["attempts"]) for l in ledgers),
               "truncated_attempts": sum(a["stop_reason"] == "length" for l in ledgers for a in l["attempts"]),
               "mean_path_steps": (sum(r["steps_total"] for r in records) / len(records)) if records else None,
               "mean_output_tokens": (sum(r["stats"]["output_tokens"] for r in records) / len(records)) if records else None,
               "elapsed_s": time.perf_counter() - t_start, "file": out}
    with open(os.path.join(out_dir, f"summary.shard{shard[0]}of{shard[1]}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


# ----------------------------------------------------------------------------- trainer
class PathOPSDTrainer(SiblingOPSDTrainer):
    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.sibling.target != "path":
            raise ValueError("PathOPSDTrainer needs sibling.target=path")
        super().__init__(gm, cfg, out_dir, examples, tokenizer, dist_ctx)
        self.cap = float(cfg.sibling.path_cap)
        self.frozen_teacher = cfg.sibling.path_teacher == "frozen"
        self._kl_now = False
        self.paths: Optional[Dict[str, dict]] = None
        if cfg.sibling.path_source:
            if not self.frozen_teacher:
                raise ValueError("sibling.path_source holds frozen-teacher paths; set path_teacher=frozen")
            self.paths, _ = load_paths(cfg.sibling.path_source)
            n_all = len(self.order)
            self.order = [i for i in self.order if self.examples[i].example_id in self.paths]
            if not self.order:
                raise RuntimeError(f"no stored path matches the training questions ({cfg.sibling.path_source})")
            if self.dist.rank == 0:
                print(f"[path] offline paths: {len(self.order)} of {n_all} training questions have a path "
                      f"({cfg.sibling.path_source})")
        if self.dist.rank == 0:
            print(f"[path] teacher={cfg.sibling.path_teacher} tau={self.tau} T={cfg.training.rollout_temperature} "
                  f"K={cfg.training.max_rollout_attempts} select={cfg.sibling.path_select} cap={self.cap} "
                  f"states/question={cfg.sibling.states_per_rollout}")

    def root_question(self, ex: Example, gidx: int):
        seed, policy = self.cfg.training.seed, int(self.state["optimizer_step"])
        ledger: Dict[str, object] = {"question_index": gidx, "example_id": ex.example_id, "policy_version": policy}
        self._bump("question_visits")
        if self.paths is not None:
            rec = self.paths[ex.example_id]
            steps = compact_supervised(rec)
            ledger.update(attempts=[], path_attempt=rec["attempt"], path_steps=rec["steps_total"],
                          path_tpf=float(rec["stats"]["tpf_dinfer"]), correct_attempts=1,
                          correct_steps_mean=float(rec["steps_total"]), trace_states=len(rec["steps"]))
            self._bump("questions_with_correct_rollout")
            self._bump("trace_states", len(rec["steps"]))
            self._bump("parallel_states", sum(len(ok) >= 2 for _, ok in steps))
            if not steps:
                ledger["outcome"] = "no_supervised_step"
                return [], ledger
            self._bump("questions_with_parallel_state")
            keep = pick(len(steps), self.cfg.sibling.states_per_rollout, derived_seed(seed, ex.example_id, policy, "states"))
            items = []
            for k in keep:
                roll, step = expand_path(rec, steps[k][0])
                items.append((roll, step, steps[k][1]))
            ledger.update(outcome="drawn", states_drawn=len(items))
            return items, ledger
        teacher_policy = 0 if self.frozen_teacher else policy  # frozen: the same paths whatever the training step
        best, attempts = sample_path(self.gm, self.tokenizer, ex, self.cfg, self.device, teacher_policy,
                                     self.cfg.sibling.path_select == "shortest", self.frozen_teacher,
                                     {"policy_version": policy})
        ledger["attempts"] = attempts
        self._bump("root_attempts", len(attempts))
        if best is None:
            ledger["outcome"] = "root_budget_exhausted"
            return [], ledger
        self._bump("questions_with_correct_rollout")
        steps = path_steps(best)
        self._bump("trace_states", len(best["steps"]))
        self._bump("parallel_states", sum(len(ok) >= 2 for _, ok in steps))
        ok_steps = [a["steps"] for a in attempts if a["accepted"]]
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
