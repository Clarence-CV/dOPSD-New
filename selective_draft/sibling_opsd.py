"""Sibling OPSD (doc/plan_a_loo_defer.md): teach every token to anticipate its same-step siblings.

Per question: the current Student (base + LoRA) decodes at the aggressive threshold tau (generation.commit_threshold)
and T_train, up to K attempts, stopping at the first answer the verifier accepts. In that successful rollout every step
that committed >= 2 tokens (before the first EOS) is a candidate state S with group C and committed tokens c.

Teacher for a supervised position i in C: the frozen base (LoRA off) on S with every OTHER member of C revealed as c_j
and i still MASK (leave-one-out) -> q_i. Target (mixed_target):
    t_i(c_i)    = alpha + (1 - alpha) * q_i(c_i),   alpha = max(0, 2*tau - 1)   (Config.sibling_alpha)
    t_i(v != c) = min((1 - alpha) * q_i(v), tau - cap_margin); the removed mass is spread uniformly over the vocabulary
so t_i(c_i) >= tau <=> q_i(c_i) >= 1/2 (a parallel commit of the target is a LOO fixed point), and no other token
reaches tau in the target (no redirect). For tau > 2/3 the cap never binds (1 - alpha < tau); for tau <= 2/3 it does.

Student: base + LoRA on S (siblings still MASK), one forward with grad -> p_i. Loss per state:
    mean_i KL(t_i || p_i) + anchor_weight * mean_k KL(p0_k || p_k)   (k = the block's other MASK positions,
                                                                      p0 = frozen base on S)
`run_sibling_diag` measures the same quantities on the frozen base without training (Phase 0).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import Example, tokenize_prompt
from .generation import generate
from .losses import forward_kl
from .root_joint import derived_seed
from .scoring import grade
from .trace import StateRecorder
from .training import BaseTrainer

COUNTERS = ("question_visits", "root_attempts", "questions_with_correct_rollout", "questions_with_parallel_state",
            "trace_states", "parallel_states", "states_used", "supervised_positions", "oom_skipped_states")


# ----------------------------------------------------------------------------- rollouts and states
def sibling_rollout(gm, tokenizer, ex: Example, cfg, rng: torch.Generator, temperature: float,
                    meta: Optional[dict] = None) -> dict:
    """One rollout of the current model (LoRA as set) from the prompt, with its full state trace."""
    ids = tokenize_prompt(tokenizer, ex.prompt_text, gm.device)
    gen_cfg = replace(cfg.generation, temperature=float(temperature))
    rec = StateRecorder(gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1],
                        dict({"example_id": ex.example_id, "prompt_ids": ids[0].tolist(), "gold": ex.answer_value},
                             **(meta or {})))
    res = generate(gm, ids, gen_cfg, guidance_mode="off", recorder=rec, sample_rng=rng)
    text = tokenizer.decode(res.completion, skip_special_tokens=True)
    g = grade(ex.answer_value, text, ex.task)
    return rec.finalize(res, {"generation": text, "parsed_answer": g["parsed_answer"], "is_correct": g["is_correct"],
                              "stats": res.stats})


def parallel_steps(roll: dict) -> List[Tuple[dict, List[int]]]:
    """Steps that committed >= 2 tokens before the first EOS -> [(step, indices into committed_positions to supervise)].
    All committed positions stay revealed for the Teacher; EOS / post-EOS commits are never supervised."""
    P, eos_id = int(roll["meta"]["prompt_len"]), int(roll["meta"]["eos_id"])
    eos = roll["eos_position"]
    out = []
    for step in roll["steps"]:
        pos, tok = step["committed_positions"].long(), step["committed_tokens"].long()
        if pos.numel() < 2:
            continue
        ok = [j for j in range(pos.numel())
              if int(tok[j]) != eos_id and (eos is None or int(pos[j]) - P < int(eos))]
        if len(ok) >= 2:
            out.append((step, ok))
    return out


def state_input(roll: dict, step: dict, device) -> torch.Tensor:
    prompt = torch.tensor(roll["meta"]["prompt_ids"], dtype=torch.long, device=device)
    return torch.cat([prompt, step["state"].to(device=device, dtype=torch.long)])[None]


@torch.no_grad()
def loo_teacher(gm, x: torch.Tensor, cpos: torch.Tensor, ctok: torch.Tensor, sup: torch.Tensor,
                batch_size: int = 4) -> torch.Tensor:
    """log q [len(sup), V]: frozen base (LoRA off) with every committed sibling revealed except the supervised one.
    x: [1, N] state before the commit (C still MASK); cpos/ctok: all positions/tokens committed at this step;
    sup: indices into cpos."""
    mask_id = gm.backend.mask_token_id
    if bool((x[0, cpos] != mask_id).any()):
        raise ValueError("committed positions must still be MASK in the state")
    filled = x[0].clone()
    filled[cpos] = ctok
    out = []
    for s in range(0, len(sup), max(1, batch_size)):
        idx = sup[s:s + batch_size]
        inputs = filled[None].expand(len(idx), -1).clone()
        rows = torch.arange(len(idx), device=x.device)
        inputs[rows, cpos[idx]] = mask_id
        lq = F.log_softmax(gm.teacher_forward(inputs)[rows, cpos[idx]].float(), -1)
        if not bool(torch.isfinite(lq).all()):
            raise RuntimeError("non-finite Teacher log probabilities")
        out.append(lq)
    return torch.cat(out)


def mixed_target(log_q: torch.Tensor, tok: torch.Tensor, alpha: float, tau: Optional[float] = None,
                 cap_margin: float = 0.01) -> torch.Tensor:
    """t = alpha*onehot(c) + (1-alpha)*q, then (tau given) every token other than c is capped at tau - cap_margin and the
    removed mass is spread uniformly over the whole vocabulary: the Teacher can lower c (defer) but never nominate a
    replacement above the threshold."""
    rows = torch.arange(len(tok), device=log_q.device)
    t = (1.0 - alpha) * log_q.exp()
    t[rows, tok] += alpha
    if tau is not None:
        cap = float(tau) - float(cap_margin)
        other = torch.ones_like(t, dtype=torch.bool)
        other[rows, tok] = False
        excess = ((t - cap).clamp_min(0.0) * other).sum(-1, keepdim=True)
        t = torch.where(other, t.clamp_max(cap), t) + excess / t.shape[-1]
    return t


def kl_from_probs(t: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(t || p) per row with t given as probabilities (zeros allowed)."""
    log_p = F.log_softmax(logits_p.float(), -1)
    return torch.where(t > 0, t * (torch.log(t.clamp_min(1e-30)) - log_p), torch.zeros_like(t)).sum(-1)


def pick(n: int, k: int, seed: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return sorted(torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k].tolist())


@torch.no_grad()
def teacher_view(gm, roll: dict, step: dict, sup_idx: List[int], batch_size: int):
    """Frozen-base quantities of one state: x, committed (pos, tok), supervised (pos, tok), log q, frozen logits p0 on
    the supervised and the anchor rows, anchor positions."""
    dev = gm.device
    x = state_input(roll, step, dev)
    cpos = step["committed_positions"].to(dev).long()
    ctok = step["committed_tokens"].to(dev).long()
    sup = torch.tensor(sup_idx, dtype=torch.long, device=dev)
    log_q = loo_teacher(gm, x, cpos, ctok, sup, batch_size)
    elig = step["eligible_positions"].to(dev).long()
    anchor = elig[~torch.isin(elig, cpos)]
    base_logits = gm.teacher_forward(x)[0]
    p0_sup = base_logits[cpos[sup]].float()
    p0_anchor = base_logits[anchor].float() if anchor.numel() else None
    del base_logits
    return x, cpos[sup], ctok[sup], log_q, p0_sup, anchor, p0_anchor


def position_stats(log_q: torch.Tensor, tok: torch.Tensor, p0_logits: torch.Tensor) -> Dict[str, List[float]]:
    rows = torch.arange(len(tok), device=tok.device)
    q = log_q.exp()
    lp0 = F.log_softmax(p0_logits, -1)
    return {"q_c": q[rows, tok].tolist(), "p0_c": lp0.exp()[rows, tok].tolist(),
            "fixed": (log_q.argmax(-1) == tok).tolist(), "kl_q_p0": forward_kl(log_q, p0_logits).tolist(),
            "q_top": q.max(-1).values.tolist()}


# ----------------------------------------------------------------------------- trainer
class SiblingOPSDTrainer(BaseTrainer):
    stage = "sibling"

    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.guidance.mode != "off":
            raise ValueError("sibling training decodes with guidance.mode=off (LoRA student)")
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
        self.alpha = cfg.sibling_alpha()
        self.tau = float(cfg.generation.commit_threshold)
        self.qpu = int(cfg.sibling.questions_per_update)
        if self.dist.rank == 0:
            print(f"[sibling] tau={cfg.generation.commit_threshold} alpha={self.alpha:.3f} cap={self.tau - cfg.sibling.cap_margin:.3f} T={cfg.training.rollout_temperature} "
                  f"K={cfg.training.max_rollout_attempts} lora={gm.lora.describe()}")

    @property
    def finished(self) -> bool:
        return int(self.state["question_round"]) * self.dist.world_size * self.qpu >= len(self.order)

    def _bump(self, key: str, by: int = 1) -> None:
        self.state[key] = int(self.state[key]) + by

    def root_question(self, ex: Example, gidx: int):
        seed, policy = self.cfg.training.seed, int(self.state["optimizer_step"])
        ledger: Dict[str, object] = {"question_index": gidx, "example_id": ex.example_id, "policy_version": policy,
                                     "attempts": []}
        self._bump("question_visits")
        success = None
        for a in range(self.cfg.training.max_rollout_attempts):
            rng = torch.Generator(device=self.device).manual_seed(derived_seed(seed, ex.example_id, policy, a))
            t0 = time.perf_counter()
            roll = sibling_rollout(self.gm, self.tokenizer, ex, self.cfg, rng, self.cfg.training.rollout_temperature,
                                   {"policy_version": policy, "attempt": a})
            self._bump("root_attempts")
            st = roll["stats"]
            ledger["attempts"].append({"attempt": a, "is_correct": bool(roll["is_correct"]), "nfe": st["nfe"],
                                       "output_tokens": st["output_tokens"], "tpf_dinfer": st["tpf_dinfer"],
                                       "mean_commit": sum(st["commit_counts"]) / max(1, len(st["commit_counts"])),
                                       "stop_reason": roll["stop_reason"], "seconds": time.perf_counter() - t0})
            if roll["is_correct"]:
                success = roll
                break
        if success is None:
            ledger["outcome"] = "root_budget_exhausted"
            return [], ledger
        self._bump("questions_with_correct_rollout")
        par = parallel_steps(success)
        self._bump("trace_states", len(success["steps"]))
        self._bump("parallel_states", len(par))
        ledger.update(first_success_attempt=len(ledger["attempts"]) - 1, trace_states=len(success["steps"]),
                      parallel_states=len(par))
        if not par:
            ledger["outcome"] = "no_parallel_state"
            return [], ledger
        self._bump("questions_with_parallel_state")
        keep = pick(len(par), self.cfg.sibling.states_per_rollout, derived_seed(seed, ex.example_id, policy, "states"))
        items = []
        for k in keep:
            step, ok = par[k]
            sub = pick(len(ok), self.cfg.sibling.max_positions_per_state,
                       derived_seed(seed, ex.example_id, policy, "pos", int(step["step_id"])))
            items.append((success, step, [ok[j] for j in sub]))
        ledger["outcome"] = "drawn"
        ledger["states_drawn"] = len(items)
        return items, ledger

    def state_loss(self, roll: dict, step: dict, sup_idx: List[int]):
        x, spos, stok, log_q, p0_sup, anchor, p0_anchor = teacher_view(self.gm, roll, step, sup_idx,
                                                                       self.cfg.training.loo_batch_size)
        t = mixed_target(log_q, stok, self.alpha, self.tau, self.cfg.sibling.cap_margin)
        logits = self.gm(x, mode="off").logits[0]  # Student: base + LoRA, siblings still MASK
        kl_t = kl_from_probs(t, logits[spos])
        loss = kl_t.mean()
        anchor_kl = None
        if p0_anchor is not None and self.cfg.sibling.anchor_weight > 0:
            anchor_kl = forward_kl(F.log_softmax(p0_anchor, -1), logits[anchor]).mean()
            loss = loss + self.cfg.sibling.anchor_weight * anchor_kl
        with torch.no_grad():
            rows = torch.arange(len(stok), device=stok.device)
            ps = logits[spos].float().softmax(-1)
            pst = position_stats(log_q, stok, p0_sup)
            tau = float(self.cfg.generation.commit_threshold)
            stats = {"kl_target": float(kl_t.mean()), "anchor_kl": float(anchor_kl) if anchor_kl is not None else 0.0,
                     "n_sup": float(len(stok)), "group_size": float(step["committed_positions"].numel()),
                     "fixed_point_frac": sum(pst["fixed"]) / len(stok),
                     "below_half_frac": sum(v < 0.5 for v in pst["q_c"]) / len(stok),
                     "q_c": sum(pst["q_c"]) / len(stok), "p0_c": sum(pst["p0_c"]) / len(stok),
                     "kl_q_p0": sum(pst["kl_q_p0"]) / len(stok),
                     "target_c": float(t[rows, stok].mean()),
                     "cap_bound_frac": float((((1 - self.alpha) * log_q.exp()).scatter(1, stok[:, None], 0.0).max(-1).values
                                              > tau - self.cfg.sibling.cap_margin).float().mean()),
                     "target_commit_frac": float((t[rows, stok] >= tau).float().mean()),
                     "student_p_c": float(ps[rows, stok].mean()),
                     "student_commit_frac": float((ps[rows, stok] >= tau).float().mean())}
        return loss, stats

    def update(self) -> Dict[str, float]:
        t_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        rnd = int(self.state["question_round"])
        self.state["question_round"] = rnd + 1
        items, outcomes, ledgers = [], [], []
        for j in range(self.qpu):
            gidx = (rnd * self.dist.world_size + self.dist.rank) * self.qpu + j
            if gidx >= len(self.order):
                outcomes.append("first_pass_exhausted")
                continue
            got, ledger = self.root_question(self.examples[self.order[gidx]], gidx)
            items.extend(got)
            outcomes.append(ledger["outcome"])
            ledgers.append(ledger)
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(ledger, default=str) + "\n")
        t_rollout = time.perf_counter() - t_start
        global_n = self.dist.sum_scalar(len(items), self.device)  # blocks until every rank finished its rollouts
        t_wait = time.perf_counter() - t_start - t_rollout
        rec: Dict[str, float] = {"question_round": rnd, "outcomes": ",".join(outcomes), "local_states": len(items),
                                 "rollout_seconds": t_rollout, "wait_seconds": t_wait, **self._onpolicy(ledgers)}
        if global_n == 0:
            self._skip_update()
            rec.update(skipped=1, round_seconds=time.perf_counter() - t_start)
            self._log(rec)
            return rec
        self.state["consecutive_empty"] = 0
        used, local_sum, agg = 0, 0.0, {}
        for roll, step, sup in items:
            try:
                loss, st = self.state_loss(roll, step, sup)
                (loss / global_n).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._bump("oom_skipped_states")
                continue
            used += 1
            local_sum += float(loss.detach())
            self._bump("states_used")
            self._bump("supervised_positions", int(st["n_sup"]))
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
        rec.update({k: v / max(1, used) for k, v in agg.items()})
        rec["loss_seconds"] = time.perf_counter() - t_start - t_rollout - t_wait
        rec.update(self._apply_update(local_sum, used, global_n))
        rec["loss"] = local_sum / used if used else None
        busy = time.perf_counter() - t_start
        self.state["busy_seconds"] = float(self.state["busy_seconds"]) + busy
        rec["round_seconds"] = busy
        self._log(rec)
        return rec

    @staticmethod
    def _onpolicy(ledgers) -> Dict[str, float]:
        """On-policy monitors of the current student at (tau, T): first-attempt accuracy, attempts, speed."""
        atts = [a for l in ledgers for a in l.get("attempts", [])]
        if not atts:
            return {}
        first = [l["attempts"][0] for l in ledgers if l.get("attempts")]
        return {"first_attempt_acc": sum(a["is_correct"] for a in first) / len(first),
                "attempts_per_question": len(atts) / len(first),
                "rollout_tpf": sum(a["tpf_dinfer"] for a in atts) / len(atts),
                "rollout_mean_commit": sum(a["mean_commit"] for a in atts) / len(atts),
                "rollout_output_tokens": sum(a["output_tokens"] for a in atts) / len(atts),
                "rollout_truncated": sum(a["stop_reason"] == "length" for a in atts) / len(atts)}

    def _log(self, rec):
        if self.device.type == "cuda":
            rec["peak_alloc_gib"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
            rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / 2 ** 30
        self.log(dict(rec, **{k: self.state[k] for k in COUNTERS}))


# ----------------------------------------------------------------------------- Phase 0 diagnostics
def run_sibling_diag(gm, tokenizer, examples: List[Example], cfg, out_dir: str, shard=(0, 1),
                     stop_at_success: bool = True) -> None:
    """Per question: up to K rollouts at (tau, T_train), stopping at the first correct one. For EVERY rollout (correct
    and wrong) draw states_per_rollout parallel-commit states and record the frozen-base LOO quantities of up to
    max_positions_per_state co-committed positions. One JSON file per question (resumable)."""
    si, sn = shard
    rows_dir = os.path.join(out_dir, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    seed, sb = cfg.training.seed, cfg.sibling
    for idx, ex in enumerate(examples):
        if idx % sn != si:
            continue
        path = os.path.join(rows_dir, f"{hashlib.sha256(ex.example_id.encode()).hexdigest()[:16]}.json")
        if os.path.exists(path):
            continue
        rec = {"example_id": ex.example_id, "tau": cfg.generation.commit_threshold,
               "temperature": cfg.training.rollout_temperature, "attempts": []}
        for a in range(cfg.training.max_rollout_attempts):
            rng = torch.Generator(device=gm.device).manual_seed(derived_seed(seed, ex.example_id, "diag", a))
            t0 = time.perf_counter()
            roll = sibling_rollout(gm, tokenizer, ex, cfg, rng, cfg.training.rollout_temperature, {"attempt": a})
            par = parallel_steps(roll)
            st = roll["stats"]
            att = {"attempt": a, "is_correct": bool(roll["is_correct"]), "steps": int(roll["steps_total"]),
                   "output_tokens": st["output_tokens"], "nfe_dinfer": st["nfe_dinfer"], "tpf_dinfer": st["tpf_dinfer"],
                   "stop_reason": roll["stop_reason"], "parallel_states": len(par),
                   "mean_commit": sum(st["commit_counts"]) / max(1, len(st["commit_counts"])),
                   "rollout_seconds": time.perf_counter() - t0, "positions": []}
            for k in pick(len(par), sb.states_per_rollout, derived_seed(seed, ex.example_id, "diag", a, "states")):
                step, ok = par[k]
                sub = [ok[j] for j in pick(len(ok), sb.max_positions_per_state,
                                           derived_seed(seed, ex.example_id, "diag", a, int(step["step_id"])))]
                _, _, stok, log_q, p0_sup, _, _ = teacher_view(gm, roll, step, sub, cfg.training.loo_batch_size)
                ps = position_stats(log_q, stok, p0_sup)
                gsize = int(step["committed_positions"].numel())
                for j in range(len(stok)):
                    att["positions"].append({"step_id": int(step["step_id"]), "group_size": gsize,
                                             **{key: ps[key][j] for key in ps}})
            rec["attempts"].append(att)
            if stop_at_success and roll["is_correct"]:
                break
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)


def summarize_sibling_diag(out_dir: str) -> Dict[str, object]:
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "rows", "*.json")))]
    if not recs:
        raise FileNotFoundError(f"no diagnostic rows under {out_dir}/rows")

    def agg(atts):
        pos = [p for a in atts for p in a["positions"]]
        n = len(pos)
        states = {}
        for a in atts:
            for p in a["positions"]:
                states.setdefault((id(a), p["step_id"]), []).append(p["fixed"])
        return {"rollouts": len(atts), "positions": n,
                "fixed_point_violation_rate": (sum(not p["fixed"] for p in pos) / n) if n else None,
                "state_violation_rate": (sum(not all(v) for v in states.values()) / len(states)) if states else None,
                "below_half_rate": (sum(p["q_c"] < 0.5 for p in pos) / n) if n else None,
                "strong_disagree_rate": (sum(p["q_c"] < 0.1 for p in pos) / n) if n else None,
                "mean_q_c": (sum(p["q_c"] for p in pos) / n) if n else None,
                "mean_p0_c": (sum(p["p0_c"] for p in pos) / n) if n else None,
                "mean_kl_q_p0": (sum(p["kl_q_p0"] for p in pos) / n) if n else None,
                "mean_tpf_dinfer": sum(a["tpf_dinfer"] for a in atts) / max(1, len(atts)),
                "mean_commit_per_step": sum(a["mean_commit"] for a in atts) / max(1, len(atts))}

    atts = [a for r in recs for a in r["attempts"]]
    solved = [r for r in recs if any(a["is_correct"] for a in r["attempts"])]
    out = {"n_questions": len(recs), "tau": recs[0]["tau"], "temperature": recs[0]["temperature"],
           "first_attempt_accuracy": sum(r["attempts"][0]["is_correct"] for r in recs) / len(recs),
           "solved_within_budget": len(solved) / len(recs),
           "mean_attempts_to_success": (sum(len(r["attempts"]) for r in solved) / len(solved)) if solved else None,
           "mean_rollout_seconds": sum(a["rollout_seconds"] for a in atts) / max(1, len(atts)),
           "all": agg(atts), "correct": agg([a for a in atts if a["is_correct"]]),
           "wrong": agg([a for a in atts if not a["is_correct"]])}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------------------------------------------------------- memory / speed probe
def sibling_memprobe(gm, cfg, lengths=(1024, 2048, 2336), group: int = 8, loo_batches=(4, 8), repeats: int = 2):
    """Peak memory and time of one training state (LOO Teacher + p0 + Student forward/backward) on synthetic states
    whose last block holds `group` co-committed positions. Also times a plain no-grad forward (one decoding step)."""
    import random
    dev, V, BL = gm.device, gm.backend.vocab_size, cfg.generation.block_length
    mask_id = gm.backend.mask_token_id
    gm.set_stage("sibling")
    res = []
    for N in lengths:
        N = N // BL * BL
        P = BL
        rng = random.Random(N)
        toks = [rng.randrange(1000, 100000) for _ in range(N)]
        state = torch.tensor(toks[P:], dtype=torch.int32)
        bs = N - BL
        state[bs - P:] = mask_id
        cpos = torch.arange(bs, bs + group, dtype=torch.int32)
        roll = {"meta": {"prompt_ids": toks[:P], "prompt_len": P, "eos_id": gm.backend.eos_token_id}}
        step = {"state": state, "committed_positions": cpos, "committed_tokens": torch.tensor(toks[bs:bs + group], dtype=torch.int32),
                "eligible_positions": torch.arange(bs, N, dtype=torch.int32), "block_start": bs, "block_end": N, "step_id": 0}
        sup = list(range(min(group, cfg.sibling.max_positions_per_state)))
        for lb in loo_batches:
            row = {"length": N, "group": group, "loo_batch": lb}
            try:
                for r in range(repeats):
                    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
                    t0 = time.perf_counter()
                    x, spos, stok, log_q, p0_sup, anchor, p0_anchor = teacher_view(gm, roll, step, sup, lb)
                    torch.cuda.synchronize(dev); t1 = time.perf_counter()
                    t = mixed_target(log_q, stok, cfg.sibling_alpha(), cfg.generation.commit_threshold, cfg.sibling.cap_margin)
                    logits = gm(x, mode="off").logits[0]
                    loss = kl_from_probs(t, logits[spos]).mean() + cfg.sibling.anchor_weight * forward_kl(
                        F.log_softmax(p0_anchor, -1), logits[anchor]).mean()
                    loss.backward()
                    torch.cuda.synchronize(dev); t2 = time.perf_counter()
                    gm.zero_grad(set_to_none=True)
                    del logits, loss, t, log_q
                    with torch.no_grad():
                        t3 = time.perf_counter(); gm(x, mode="off"); torch.cuda.synchronize(dev); t4 = time.perf_counter()
                row.update(teacher_s=t1 - t0, student_fwd_bwd_s=t2 - t1, decode_forward_s=t4 - t3,
                           peak_alloc_gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(dev) / 2 ** 30, oom=False)
            except torch.cuda.OutOfMemoryError:
                gm.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                row.update(oom=True)
            print(json.dumps(row), flush=True)
            res.append(row)
    return res
