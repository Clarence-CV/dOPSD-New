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
                 cap_margin: float = 0.01, mode: str = "capped_uniform", p_student: Optional[torch.Tensor] = None,
                 mask_id: Optional[int] = None) -> torch.Tensor:
    """Target t over the vocabulary for supervised positions with trajectory tokens `tok` (= c).

    capped_uniform: t = alpha*onehot(c) + (1-alpha)*q, then (tau given) every token other than c is capped at
        tau - cap_margin and the removed mass is spread uniformly over the whole vocabulary.
    mask_sink (needs tau, p_student = the Student's own detached probabilities on the same rows, mask_id):
        t(c) = alpha + (1-alpha) q(c), exactly as above, so t(c) >= tau <=> q(c) >= 1/2 (LOO fixed point).
        Commit rows (t(c) >= tau): t(v) = min((1-alpha) q(v), tau - cap_margin) for real tokens v != c.
        Defer rows (t(c) < tau): t(v) = min((1-alpha) q(v), p_student(v), tau - cap_margin): no real token gains
            probability over what the Student already gave it (the Teacher can veto c but never nominate a replacement).
        t(MASK) = everything left (>= (1-alpha) q(MASK)): the decoder never commits a position whose argmax is MASK,
            so the freed mass means "keep this position masked for now"; no real token absorbs it.
    copy: t = onehot(c) (alpha = 1; the SwiftDLM-style hard target on newly committed tokens, a control)."""
    rows = torch.arange(len(tok), device=log_q.device)
    if mode == "copy":
        t = torch.zeros_like(log_q)
        t[rows, tok] = 1.0
        return t
    t = (1.0 - alpha) * log_q.exp()
    t[rows, tok] += alpha
    if mode == "mask_sink":
        if tau is None or p_student is None or mask_id is None:
            raise ValueError("mask_sink needs tau, p_student and mask_id")
        if bool((tok == mask_id).any()):
            raise ValueError("a supervised trajectory token cannot be MASK")
        tc = t[rows, tok].clone()
        cap = float(tau) - float(cap_margin)
        defer = (tc < float(tau))[:, None]
        lim = torch.where(defer, torch.minimum(t, p_student.to(t.dtype)), t).clamp_max(cap)
        lim[rows, tok] = tc
        lim[:, mask_id] = 0.0
        lim[:, mask_id] = (1.0 - lim.sum(-1)).clamp_min(0.0)
        return lim
    if mode != "capped_uniform":
        raise ValueError(f"unknown target mode {mode!r}")
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


def mask_stats(t: torch.Tensor, ps: torch.Tensor, tok: torch.Tensor, tau: float, mask_id: int,
               pa: Optional[torch.Tensor] = None, p0a: Optional[torch.Tensor] = None,
               p0s: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """MASK-probability monitors. t / ps: target / Student probabilities on the supervised rows; pa / p0a: Student /
    frozen base on the anchor rows (the block's other MASK positions); p0s: frozen base on the supervised rows.
    defer rows = target says t(c) < tau. A row "would defer" when no real token reaches tau or its argmax is MASK."""
    rows = torch.arange(len(tok), device=tok.device)
    defer = t[rows, tok] < tau
    real = ps.clone()
    real[:, mask_id] = 0.0
    would_defer = (real.max(-1).values < tau) | (ps.argmax(-1) == mask_id)
    f = lambda v, m: float(v[m].mean()) if bool(m.any()) else float("nan")
    out = {"defer_target_frac": float(defer.float().mean()), "target_mask": float(t[:, mask_id].mean()),
           "student_mask_sup": float(ps[:, mask_id].mean()),
           "student_mask_defer": f(ps[:, mask_id], defer), "student_mask_commit": f(ps[:, mask_id], ~defer),
           "student_c_defer": f(ps[rows, tok], defer), "student_would_defer_on_defer": f(would_defer.float(), defer),
           "student_would_defer_on_commit": f(would_defer.float(), ~defer)}
    if p0s is not None:
        out["base_mask_sup"] = float(p0s[:, mask_id].mean())
    if pa is not None:
        out.update(student_mask_anchor=float(pa[:, mask_id].mean()), student_mask_anchor_max=float(pa[:, mask_id].max()),
                   student_mask_argmax_anchor=float((pa.argmax(-1) == mask_id).float().mean()))
    if p0a is not None:
        out.update(base_mask_anchor=float(p0a[:, mask_id].mean()), base_mask_anchor_max=float(p0a[:, mask_id].max()))
    return out


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
        self.target_mode = cfg.sibling.target
        self.mask_id = int(gm.backend.mask_token_id)
        if self.target_mode == "mask_sink" and self.tau < 0.5:
            raise ValueError("sibling.target=mask_sink needs commit_threshold >= 0.5 (below it alpha is clipped to 0 "
                             "and t(c) >= tau no longer implies a LOO fixed point)")
        if self.dist.rank == 0:
            print(f"[sibling] tau={cfg.generation.commit_threshold} alpha={self.alpha:.3f} cap={self.tau - cfg.sibling.cap_margin:.3f} T={cfg.training.rollout_temperature} "
                  f"K={cfg.training.max_rollout_attempts} target={self.target_mode} lora={gm.lora.describe()}")

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
        logits = self.gm(x, mode="off").logits[0]  # Student: base + LoRA, siblings still MASK
        p_stu = logits[spos].detach().float().softmax(-1)
        t = mixed_target(log_q, stok, self.alpha, self.tau, self.cfg.sibling.cap_margin, self.target_mode, p_stu,
                         self.mask_id)
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
                     "student_commit_frac": float((ps[rows, stok] >= tau).float().mean()),
                     **mask_stats(t, ps, stok, tau, self.mask_id,
                                  logits[anchor].float().softmax(-1) if anchor.numel() else None,
                                  F.softmax(p0_anchor, -1) if p0_anchor is not None else None,
                                  F.softmax(p0_sup, -1))}
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
            for k, v in st.items():  # NaN = the monitor had no rows in this state (e.g. no defer position)
                if v == v:
                    s_, n_ = agg.get(k, (0.0, 0))
                    agg[k] = (s_ + v, n_ + 1)
        rec.update({k: s_ / n_ for k, (s_, n_) in agg.items()})
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
                    logits = gm(x, mode="off").logits[0]
                    t = mixed_target(log_q, stok, cfg.sibling_alpha(), cfg.generation.commit_threshold,
                                     cfg.sibling.cap_margin, cfg.sibling.target,
                                     logits[spos].detach().float().softmax(-1), gm.backend.mask_token_id)
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


# ----------------------------------------------------------------------------- MASK-sink probe
def _row_metrics(t, ps, tok, tau, mask_id, pa=None, p0a_log=None):
    """Per-row tensors for the probe: supervised rows (target defer flag, Student p(MASK), p(c), would-defer) and anchor
    rows (Student p(MASK), argmax MASK, KL(p0 || p))."""
    rows = torch.arange(len(tok), device=tok.device)
    real = ps.clone()
    real[:, mask_id] = 0.0
    out = {"defer": t[rows, tok] < tau, "p_mask": ps[:, mask_id], "p_c": ps[rows, tok],
           "would_defer": (real.max(-1).values < tau) | (ps.argmax(-1) == mask_id),
           "argmax_mask": ps.argmax(-1) == mask_id, "kl_t": kl_from_probs(t, ps.clamp_min(1e-30).log())}
    if pa is not None:
        out.update(a_p_mask=pa[:, mask_id], a_argmax_mask=pa.argmax(-1) == mask_id,
                   a_kl=forward_kl(p0a_log, pa.clamp_min(1e-30).log()))
    return out


def sibling_maskprobe(gm, tokenizer, examples: List[Example], cfg, out_dir: str, lrs=(2e-5, 1e-4), steps: int = 30,
                      batch: int = 8, eval_every: int = 5, holdout_frac: float = 0.25, max_eval_states: int = 48) -> dict:
    """Can the Student learn the mask_sink target, and does MASK leak to positions that should not defer?

    1. Base rollouts (fresh LoRA = base) at (tau, T_train), up to K attempts, first correct kept; parallel states and
       frozen-base LOO Teacher exactly as in training; cached on the CPU. Reports base p(MASK) on supervised and anchor
       rows (does LLaDA2 ever give MASK mass at a masked position?).
    2. Questions split into train / held-out. For each lr: reset LoRA, `steps` AdamW updates of the real training
       loss (mask_sink target + anchor), each on `batch` train states, half of them drawn from states that contain a
       defer row (q(c) < 1/2): a stress test with far more MASK pressure than training (~4% defer rows).
    3. Every `eval_every` updates, no-grad Student metrics on the train defer states and on held-out states: p(MASK) and
       would-defer on defer rows vs commit rows, and on anchor rows (leak), KL(p0 || p) on anchors (drift)."""
    dev, mask_id, tau, sb = gm.device, int(gm.backend.mask_token_id), float(cfg.generation.commit_threshold), cfg.sibling
    alpha, seed = cfg.sibling_alpha(), cfg.training.seed
    if tau < 0.5:
        raise ValueError("maskprobe needs commit_threshold >= 0.5")
    os.makedirs(out_dir, exist_ok=True)
    params = gm.set_stage("sibling")
    init = {k: v.detach().clone() for k, v in gm.lora.state_dict().items()}
    t0 = time.perf_counter()
    states, qinfo = [], []
    for qi, ex in enumerate(examples):
        success, n_att = None, 0
        for a in range(cfg.training.max_rollout_attempts):
            rng = torch.Generator(device=dev).manual_seed(derived_seed(seed, ex.example_id, "maskprobe", a))
            roll = sibling_rollout(gm, tokenizer, ex, cfg, rng, cfg.training.rollout_temperature, {"attempt": a})
            n_att += 1
            if roll["is_correct"]:
                success = roll
                break
        qinfo.append({"example_id": ex.example_id, "attempts": n_att, "solved": success is not None})
        if success is None:
            continue
        par = parallel_steps(success)
        for k in pick(len(par), sb.states_per_rollout, derived_seed(seed, ex.example_id, "maskprobe", "states")):
            step, ok = par[k]
            sub = [ok[j] for j in pick(len(ok), sb.max_positions_per_state,
                                       derived_seed(seed, ex.example_id, "maskprobe", int(step["step_id"])))]
            x, spos, stok, log_q, p0_sup, anchor, p0_anchor = teacher_view(gm, success, step, sub,
                                                                           cfg.training.loo_batch_size)
            tc = alpha + (1 - alpha) * log_q.exp()[torch.arange(len(stok), device=dev), stok]
            states.append({"q": qi, "x": x.cpu(), "spos": spos.cpu(), "stok": stok.cpu(), "log_q": log_q.cpu(),
                           "p0_sup_log": F.log_softmax(p0_sup, -1).cpu(), "anchor": anchor.cpu(),
                           "p0a_log": F.log_softmax(p0_anchor, -1).cpu() if p0_anchor is not None else None,
                           "has_defer": bool((tc < tau).any())})
        print(f"[maskprobe] q{qi} attempts={n_att} solved={success is not None} states={len(states)} "
              f"defer_states={sum(s['has_defer'] for s in states)} {time.perf_counter() - t0:.0f}s", flush=True)
    if not states:
        raise RuntimeError("no parallel states collected")
    collect_s = time.perf_counter() - t0

    # base MASK statistics
    cat = lambda key: torch.cat([s[key] for s in states if s[key] is not None])
    p0s_mask, q_mask = cat("p0_sup_log")[:, mask_id].exp(), cat("log_q")[:, mask_id].exp()
    p0a_mask = cat("p0a_log")[:, mask_id].exp() if any(s["p0a_log"] is not None for s in states) else torch.zeros(1)
    q_c = torch.cat([s["log_q"].exp()[torch.arange(len(s["stok"])), s["stok"]] for s in states])
    qd = lambda v: {"mean": float(v.mean()), "p99": float(v.float().quantile(0.99)) if v.numel() > 1 else float(v.max()),
                    "max": float(v.max()), "frac_gt_1e-3": float((v > 1e-3).float().mean())}
    base = {"questions": len(examples), "solved": sum(q["solved"] for q in qinfo), "states": len(states),
            "defer_states": sum(s["has_defer"] for s in states), "sup_rows": int(q_c.numel()),
            "defer_rows": int((alpha + (1 - alpha) * q_c < tau).sum()), "anchor_rows": int(p0a_mask.numel()),
            "base_p_mask_sup": qd(p0s_mask), "teacher_q_mask_sup": qd(q_mask), "base_p_mask_anchor": qd(p0a_mask),
            "collect_seconds": collect_s}
    print("[maskprobe] base " + json.dumps(base), flush=True)

    nq = len(examples)
    held_q = set(range(nq - max(1, int(round(holdout_frac * nq))), nq))
    train = [s for s in states if s["q"] not in held_q]
    held = [s for s in states if s["q"] in held_q][:max_eval_states]
    tr_def, tr_oth = [s for s in train if s["has_defer"]], [s for s in train if not s["has_defer"]]
    if not tr_def:
        raise RuntimeError("no train state with a defer row; raise the number of questions")
    evalsets = {"train_defer": tr_def[:max_eval_states], "heldout": held}

    def to_dev(s):
        return (s["x"].to(dev), s["spos"].to(dev), s["stok"].to(dev), s["log_q"].to(dev), s["anchor"].to(dev),
                s["p0a_log"].to(dev) if s["p0a_log"] is not None else None)

    @torch.no_grad()
    def evaluate(step_no):
        rec = {"step": step_no}
        for name, ss in evalsets.items():
            acc: Dict[str, List[torch.Tensor]] = {}
            for s in ss:
                x, spos, stok, log_q, anchor, p0a = to_dev(s)
                logits = gm(x, mode="off").logits[0].float()
                ps = logits[spos].softmax(-1)
                t = mixed_target(log_q, stok, alpha, tau, sb.cap_margin, "mask_sink", ps, mask_id)
                pa = logits[anchor].softmax(-1) if anchor.numel() else None
                for k, v in _row_metrics(t, ps, stok, tau, mask_id, pa, p0a).items():
                    acc.setdefault(k, []).append(v.float().cpu())
                del logits
            if not acc:
                continue
            c = {k: torch.cat(v) for k, v in acc.items()}
            d = c["defer"].bool()
            m = lambda key, sel: float(c[key][sel].mean()) if bool(sel.any()) else None
            rec[name] = {"defer_rows": int(d.sum()), "commit_rows": int((~d).sum()),
                         "p_mask_defer": m("p_mask", d), "p_mask_commit": m("p_mask", ~d),
                         "p_c_defer": m("p_c", d), "p_c_commit": m("p_c", ~d),
                         "would_defer_on_defer": m("would_defer", d), "would_defer_on_commit": m("would_defer", ~d),
                         "argmax_mask_defer": m("argmax_mask", d), "kl_target_sup": float(c["kl_t"].mean())}
            if "a_p_mask" in c:
                rec[name].update(anchor_rows=int(c["a_p_mask"].numel()), p_mask_anchor=float(c["a_p_mask"].mean()),
                                 p_mask_anchor_max=float(c["a_p_mask"].max()),
                                 argmax_mask_anchor=float(c["a_argmax_mask"].float().mean()),
                                 kl_p0_anchor=float(c["a_kl"].mean()))
        print("[maskprobe] eval " + json.dumps(rec), flush=True)
        return rec

    runs = []
    for lr in lrs:
        gm.lora.load_state_dict(init)
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=cfg.training.weight_decay,
                                betas=(cfg.training.adam_beta1, cfg.training.adam_beta2))
        g = torch.Generator().manual_seed(seed)
        hist, t1 = [evaluate(0)], time.perf_counter()
        for it in range(1, steps + 1):
            nd = min(len(tr_def), batch // 2) if tr_oth else min(len(tr_def), batch)
            pick_d = torch.randperm(len(tr_def), generator=g)[:nd].tolist()
            pick_o = torch.randperm(len(tr_oth), generator=g)[:batch - nd].tolist() if tr_oth else []
            bs = [tr_def[i] for i in pick_d] + [tr_oth[i] for i in pick_o]
            tot = 0.0
            for s in bs:
                x, spos, stok, log_q, anchor, p0a = to_dev(s)
                logits = gm(x, mode="off").logits[0]
                ps = logits[spos].detach().float().softmax(-1)
                t = mixed_target(log_q, stok, alpha, tau, sb.cap_margin, "mask_sink", ps, mask_id)
                loss = kl_from_probs(t, logits[spos]).mean()
                if p0a is not None and sb.anchor_weight > 0:
                    loss = loss + sb.anchor_weight * forward_kl(p0a, logits[anchor]).mean()
                (loss / len(bs)).backward()
                tot += float(loss.detach())
                del logits, loss
            for grp in opt.param_groups:
                grp["lr"] = float(lr) * min(1.0, it / cfg.training.warmup_updates)
            gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            print(f"[maskprobe] lr={lr} step={it} loss={tot / len(bs):.4f} grad_norm={gn:.3f}", flush=True)
            if it % eval_every == 0 or it == steps:
                hist.append(dict(evaluate(it), loss=tot / len(bs), grad_norm=gn))
        runs.append({"lr": float(lr), "steps": steps, "batch": batch, "train_seconds": time.perf_counter() - t1,
                     "history": hist})
    gm.lora.load_state_dict(init)
    res = {"tau": tau, "alpha": alpha, "mask_id": mask_id, "train_states": len(train), "train_defer_states": len(tr_def),
           "heldout_states": len(held), "base": base, "runs": runs, "questions": qinfo}
    with open(os.path.join(out_dir, "maskprobe.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res
