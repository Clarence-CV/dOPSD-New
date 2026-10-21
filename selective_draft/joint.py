"""Joint-reference OPSD: same-state continuations, whole-tuple selection, and sibling PI.

Teacher = frozen native model. Student = the same model with simultaneous group_soft messages.
No selected reference token is ever passed into the Student forward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F

from .scoring import grade
from .generation import generate
from .losses import forward_kl
from .model import GuidanceSpec
from .modules import soft_message
from .trace import save_rollouts
from .training import GuidanceTrainer, StateRef, state_tensors


@dataclass
class JointReference:
    tokens: torch.Tensor
    log_q: torch.Tensor
    witness_index: int
    distance: float


@torch.no_grad()
def certify_tuple(gm, x, attention_mask, positions, tokens, batch_size=1):
    """Leave target i masked; reveal only its peers in G. Each row is a separate attention input."""
    if positions.numel() < 2 or positions.numel() != tokens.numel():
        raise ValueError("certification requires at least two paired positions/tokens")
    if positions.unique().numel() != positions.numel() or bool((x[positions] != gm.backend.mask_token_id).any()):
        raise ValueError("certification positions must be distinct MASK positions")
    if bool((tokens == gm.backend.mask_token_id).any()) or batch_size < 1:
        raise ValueError("invalid candidate tokens or teacher batch size")
    logqs, stats = [], {"teacher_forwards": 0, "teacher_inputs": 0, "top1": [],
                        "reference_probs": [], "top1_gaps": []}
    for start in range(0, len(positions), batch_size):
        pp = positions[start:start + batch_size]
        inputs = x[None].expand(len(pp), -1).clone()
        inputs[:, positions] = tokens[None]
        rows = torch.arange(len(pp), device=x.device)
        inputs[rows, pp] = gm.backend.mask_token_id
        logits = gm.base_forward(inputs, attention_mask[None].expand(len(pp), -1))[rows, pp]
        lq = F.log_softmax(logits.float(), -1)
        if not bool(torch.isfinite(lq).all()):
            raise RuntimeError("non-finite Teacher log probabilities")
        top1 = lq.argmax(-1)
        prob = lq.exp()
        top2 = prob.topk(2, dim=-1).values
        stats["teacher_forwards"] += 1
        stats["teacher_inputs"] += len(pp)
        stats["top1"].extend(top1.tolist())
        stats["reference_probs"].extend(prob[rows, tokens[start:start + len(pp)]].tolist())
        stats["top1_gaps"].extend((top2[:, 0] - top2[:, 1]).tolist())
        if not torch.equal(top1, tokens[start:start + len(pp)]):
            return None, stats
        logqs.append(lq)
    return torch.cat(logqs), stats


@torch.no_grad()
def select_reference(gm, x, attention_mask, positions, draft_probs, candidates, teacher_batch_size=1):
    """Sort validated compatible witnesses, then certify until the first success. Ties: tuple lexicographic."""
    message = soft_message(draft_probs, gm.backend.embedding_weight(), gm.cfg.message_vocab_chunk)
    bank = {}
    stats = {"candidate_attempts": len(candidates), "correct_candidates": 0, "compatible_candidates": 0,
             "unique_candidates": 0, "certified": 0, "teacher_forwards": 0, "teacher_inputs": 0}
    for index, roll in enumerate(candidates):
        if not roll.get("is_correct", False):
            continue
        stats["correct_candidates"] += 1
        P = roll["meta"]["prompt_len"]
        witness = roll["final_state"].to(device=x.device, dtype=torch.long)
        known = x[P:] != gm.backend.mask_token_id
        n = x.numel() - P
        completed = int(roll["computed_length"])
        if witness.numel() < n or completed > witness.numel() or completed <= int(positions.max()) - P:
            continue
        if bool((witness[:completed] == gm.backend.mask_token_id).any()):
            continue
        if not torch.equal(witness[:n][known], x[P:][known]):
            continue
        if roll["meta"]["prompt_ids"] != x[:P].tolist():
            continue
        stats["compatible_candidates"] += 1
        tokens = witness[positions - P]
        key = tuple(tokens.tolist())
        if key in bank:
            continue
        emb = gm.backend.embedding_weight()[tokens].float()
        distance = float((message.float() - emb).square().sum())
        bank[key] = (distance, index, tokens)
    stats["unique_candidates"] = len(bank)
    audit = []
    for key, (distance, index, tokens) in sorted(bank.items(), key=lambda item: (item[1][0], item[0])):
        log_q, check = certify_tuple(gm, x, attention_mask, positions, tokens, teacher_batch_size)
        stats["teacher_forwards"] += check["teacher_forwards"]
        stats["teacher_inputs"] += check["teacher_inputs"]
        audit.append(dict(tokens=list(key), distance=distance, witness_index=index,
                          accepted=log_q is not None, **check))
        if log_q is not None:
            stats["certified"] = 1
            return JointReference(tokens, log_q, index, distance), stats, audit
    return None, stats, audit


class JointTrainer(GuidanceTrainer):
    stage = "joint"
    guidance_mode = "group_soft"

    def __init__(self, gm, cfg, out_dir, examples, tokenizer, dist_ctx=None):
        if cfg.teacher.mode != "joint" or cfg.guidance.mode != "group_soft" or cfg.generation.mode != "threshold":
            raise ValueError("JointTrainer requires joint Teacher, group_soft Student and threshold decoding")
        if cfg.rollout_temperature() <= 0:
            raise ValueError("joint candidate exploration requires a positive rollout temperature")
        super().__init__(gm, cfg, out_dir, examples, tokenizer, dist_ctx)
        self.reference_dir = os.path.join(out_dir, "references")
        os.makedirs(self.reference_dir, exist_ok=True)
        self.round_stats = {}

    def refresh(self):
        """Natural on-policy states, including failed trajectories. Pass@K is spent at each selected state."""
        self.buffer, rolls = [], []
        self.round_stats = {}
        for _ in range(self.cfg.training.rollouts_per_update if self.examples else 0):
            roll = self.rollout(self._next_example())
            rolls.append(roll)
            candidates = [s for s in roll["steps"] if len(s.get("group_positions", [])) >= 2]
            k = min(self.cfg.training.states_per_rollout, len(candidates))
            pick = torch.randperm(len(candidates), generator=self.gen_data)[:k].tolist()
            self.buffer.extend(StateRef(roll, candidates[i], int(self.state["optimizer_step"])) for i in pick)
        rnd = int(self.state["rollout_round"])
        save_rollouts(os.path.join(self.rollout_dir, f"round-{rnd:05d}_rank{self.dist.rank}.pt"), rolls)
        self.state["rollout_round"] = rnd + 1
        return {"rollouts": len(rolls), "correct": sum(r["is_correct"] for r in rolls),
                "buffer_states": len(self.buffer)}

    @torch.no_grad()
    def continuation(self, ref, x, attempt):
        P = ref.roll["meta"]["prompt_len"]
        meta = ref.roll["meta"]
        ids = x[:P][None]
        pmask = torch.tensor(meta["prompt_attention_mask"], device=self.device)[None]
        gen_cfg = replace(self.cfg.generation, temperature=self.cfg.rollout_temperature())
        result = generate(self.gm, ids, pmask, gen_cfg, guidance_mode=self.guidance_mode,
                          sample_rng=self.gen_sample, initial_state=x[None], start_block=ref.step["block_id"],
                          initial_step=ref.step["step_id"])
        text = self.tokenizer.decode(result.completion_ids(0), skip_special_tokens=True)
        g = grade(meta.get("question", ""), meta["gold"], text, self.cfg.data.task, self.cfg.data.scorer)
        return {"meta": dict(meta, attempt=attempt, policy_version=int(self.state["optimizer_step"])),
                "final_state": result.sequences[0, P:].to(torch.int32).cpu(),
                "computed_length": result.computed_lengths[0], "completion_length": result.effective_lengths[0],
                "stop_reason": result.stop_reasons[0], "generation": text, "parsed_answer": g["parsed_answer"],
                "is_correct": g["is_correct"], "verdicts": g["verdicts"], "stats": result.stats}

    def state_loss(self, ref):
        self.round_stats["states_attempted"] = self.round_stats.get("states_attempted", 0) + 1
        x, attn, ctx, elig_pos = state_tensors(ref.roll, ref.step, self.device)
        with torch.no_grad():
            cap = self.gm(x[None], attn[None], context=ctx, mode="group_readout")
        gi = cap.guidance
        native = cap.logits[0]  # group_readout leaves hidden untouched: native logits
        del cap
        valid = gi.group_pos[0] >= 0
        positions = gi.group_pos[0, valid]
        self.state["reselect_checked"] = int(self.state.get("reselect_checked", 0)) + 1
        if not torch.equal(positions.cpu(), ref.step["group_positions"].long()):
            self.state["reselect_mismatch"] = int(self.state.get("reselect_mismatch", 0)) + 1
            self.round_stats["reselect_mismatch"] = self.round_stats.get("reselect_mismatch", 0) + 1
            return None, {}
        probs = gi.group_probs[0, valid]
        candidates = [self.continuation(ref, x, a) for a in range(self.cfg.training.max_rollout_attempts)]
        target, stats, audit = select_reference(self.gm, x, attn, positions, probs, candidates,
                                               self.cfg.teacher.candidate_batch_size)
        for key, value in stats.items():
            self.round_stats[key] = self.round_stats.get(key, 0) + value
        # Save the candidate bank once for this exact state/policy version, including failed attempts.
        sample_id = int(self.state.get("reference_states", 0))
        self.state["reference_states"] = sample_id + 1
        top_p, top_v = probs.topk(min(8, probs.shape[-1]), dim=-1)
        torch.save({"policy_version": int(self.state["optimizer_step"]), "state": ref.step,
                    "meta": ref.roll["meta"], "group_positions": positions.cpu(),
                    "draft_topk_probs": top_p.cpu(), "draft_topk_ids": top_v.cpu(),
                    "soft_messages": soft_message(probs, self.gm.backend.embedding_weight()).cpu(),
                    "candidates": candidates, "certification": audit, "stats": stats,
                    "selected_tokens": None if target is None else target.tokens.cpu(),
                    "selected_witness": None if target is None else target.witness_index},
                   os.path.join(self.reference_dir, f"state-{sample_id:07d}_rank{self.dist.rank}.pt"))
        if target is None:
            return None, {}
        spec = GuidanceSpec(gi.source_pos, gi.gate_open, gi.source_probs,
                            group_pos=gi.group_pos, group_probs=gi.group_probs)
        # The only supplied messages are natural draft probabilities. PI stays exclusively in log_q.
        out = self.gm(x[None], attn[None], context=ctx, mode="group_soft", spec=spec)
        kl = forward_kl(target.log_q, out.logits[0, positions])
        with torch.no_grad():
            p = out.logits[0, positions].float().softmax(-1)
            conf, pred = p.max(-1)
            stats.update(group_size=len(positions), kl_sum=float(kl.sum()),
                         reference_agreement=float((pred == target.tokens).float().mean()),
                         high_conf_wrong=int(((conf > self.cfg.generation.commit_threshold) &
                                              (pred != target.tokens)).sum()),
                         high_conf_count=int((conf > self.cfg.generation.commit_threshold).sum()),
                         delta_norm=float(out.guidance.delta_norm[0]),
                         policy_lag=int(self.state["optimizer_step"]) - ref.policy_version)
            # Positions outside G receive no loss but see the injection through the tail (doc §9).
            outside = elig_pos[~torch.isin(elig_pos, positions)]
            if outside.numel():
                tau = self.cfg.generation.commit_threshold
                lq0 = F.log_softmax(native[outside].float(), -1)
                c0, a0 = lq0.exp().max(-1)
                c1, a1 = out.logits[0, outside].float().softmax(-1).max(-1)
                stats.update(outside_kl=float(forward_kl(lq0, out.logits[0, outside]).mean()),
                             outside_commit_flips=int(((c0 > tau) != (c1 > tau)).sum() +
                                                      ((c0 > tau) & (c1 > tau) & (a0 != a1)).sum()))
        return kl.mean(), stats

    def log(self, rec):
        super().log(dict(rec, **{f"joint_{k}": v for k, v in self.round_stats.items()}))
