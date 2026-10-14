"""Stage 1 (EarlyReadout) and stage 2 (MessageInjector) training loops.

Single process by default; with WORLD_SIZE > 1 the loop uses manual gradient all-reduce
(the modules are ~1M parameters) so that no forward has to go through a DDP wrapper and the
'state count differs per rank' / 'empty rank' cases are handled explicitly at update boundaries."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint, save_snapshot
from .config import Config
from .data import Example, is_correct, parse_prediction, tokenize_prompts
from .generation import generate
from .losses import forward_kl
from .model import GuidanceSpec, GuidedModel, StepContext
from .teacher import teacher_targets
from .trace import StateRecorder, load_rollouts, save_rollouts


# ----------------------------------------------------------------------------- distributed
class DistCtx:
    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1
        if self.enabled and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend)

    def sum_scalar(self, value: float, device) -> float:
        if not self.enabled:
            return float(value)
        t = torch.tensor([float(value)], dtype=torch.float64, device=device if device.type == "cuda" else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def all_reduce_grads(self, params: List[torch.nn.Parameter]) -> None:
        if not self.enabled:
            return
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    def barrier(self):
        if self.enabled:
            dist.barrier()


# ----------------------------------------------------------------------------- states
@dataclass
class StateRef:
    roll: dict
    step: dict
    policy_version: int = -1


def state_tensors(roll: dict, step: dict, device) -> Tuple[torch.Tensor, torch.Tensor, StepContext, torch.Tensor]:
    """-> x [N], attn [N], context (B=1), eligible absolute positions [n]."""
    meta = roll["meta"]
    prompt = torch.tensor(meta["prompt_ids"], dtype=torch.long, device=device)
    pmask = torch.tensor(meta["prompt_attention_mask"], dtype=torch.long, device=device)
    state = step["state"].to(device=device, dtype=torch.long)
    x = torch.cat([prompt, state])
    attn = torch.cat([pmask, torch.ones_like(state)])
    elig_pos = step["eligible_positions"].to(device=device, dtype=torch.long)
    elig = torch.zeros(1, x.numel(), dtype=torch.bool, device=device)
    elig[0, elig_pos] = True
    ctx = StepContext(eligible_mask=elig, active_rows=torch.ones(1, dtype=torch.bool, device=device),
                      block_start=torch.tensor([step["block_start"]], device=device),
                      block_end=torch.tensor([step["block_end"]], device=device),
                      step_id=torch.tensor([step["step_id"]], device=device))
    return x, attn, ctx, elig_pos


def collect_state_refs(rollouts: List[dict], filter_correct_only: bool, require_gate: bool) -> List[StateRef]:
    refs = []
    for roll in rollouts:
        if filter_correct_only and not roll.get("is_correct", False):
            continue
        for step in roll["steps"]:
            if len(step["eligible_positions"]) == 0:
                continue
            if require_gate and not (step["gate_open"] and len(step["eligible_positions"]) >= 2):
                continue
            refs.append(StateRef(roll, step))
    return refs


# ----------------------------------------------------------------------------- base trainer
class BaseTrainer:
    stage = "none"

    def __init__(self, gm: GuidedModel, cfg: Config, out_dir: str, dist_ctx: Optional[DistCtx] = None):
        self.gm, self.cfg, self.out_dir = gm, cfg, out_dir
        self.dist = dist_ctx or DistCtx()
        self.device = gm.device
        self.params = gm.set_stage(self.stage)
        gm.assert_base_frozen()
        if not self.params:
            raise RuntimeError("no trainable parameters for this stage")
        lr = cfg.training.learning_rate_readout if self.stage == "readout" else cfg.training.learning_rate_guidance
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=cfg.training.weight_decay, betas=(0.9, 0.999))
        seed = cfg.training.seed + 1000 * self.dist.rank
        self.gen_data = torch.Generator().manual_seed(seed)
        self.gen_teacher = torch.Generator().manual_seed(cfg.teacher.seed + 1000 * self.dist.rank)
        self.gen_sample = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.generators = {"data": self.gen_data, "teacher": self.gen_teacher, "sample": self.gen_sample}
        self.state: Dict[str, object] = {"stage": self.stage, "optimizer_step": 0, "rollout_round": 0,
                                         "data_cursor": 0, "states_seen": 0, "skipped_updates": 0,
                                         "consecutive_empty": 0}
        self.base_checksum = gm.base_checksum()
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, f"log_rank{self.dist.rank}.jsonl")
        if self.dist.rank == 0:
            print(f"[{self.stage}] trainable params: {gm.trainable_parameter_names()} "
                  f"({sum(p.numel() for p in self.params)} values)")

    # ---- logging / saving ----
    def log(self, rec: Dict[str, object]) -> None:
        rec = dict(rec, time=time.time(), optimizer_step=self.state["optimizer_step"], rank=self.dist.rank)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if self.dist.rank == 0 and (self.state["optimizer_step"] % max(1, self.cfg.training.log_every) == 0):
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                              if k not in ("time",)}, default=str))

    def save(self, tag: Optional[str] = None, metrics: Optional[dict] = None) -> str:
        step = int(self.state["optimizer_step"])
        name = tag or f"checkpoint-step-{step}"
        path = os.path.join(self.out_dir, name)
        self.dist.barrier()
        save_checkpoint(path, self.gm, self.cfg, self.stage, self.optimizer, dict(self.state),
                        self.generators, rank=self.dist.rank, metrics=metrics)
        if self.dist.rank == 0:
            save_snapshot(os.path.join(self.out_dir, "snapshot"), self.gm, self.cfg, self.stage, metrics)
        self.dist.barrier()
        return path

    def resume(self, path: str) -> None:
        st = load_checkpoint(path, self.gm, self.cfg, self.optimizer, self.generators, rank=self.dist.rank)
        if st.get("stage") != self.stage:
            raise ValueError(f"checkpoint stage {st.get('stage')} != {self.stage}")
        self.state.update(st)
        self.params = self.gm.set_stage(self.stage)

    # ---- optimisation step shared by both stages ----
    def _apply_update(self, local_sum_loss: float, local_n: int, global_n: float) -> Dict[str, float]:
        if local_n == 0:
            # keep collectives aligned: a zero loss connected to every trainable parameter
            zero = sum((p.float().sum() * 0.0) for p in self.params)
            zero.backward()
        self.dist.all_reduce_grads(self.params)
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.training.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state["optimizer_step"] = int(self.state["optimizer_step"]) + 1
        self.state["states_seen"] = int(self.state["states_seen"]) + local_n
        cur = self.gm.base_checksum()
        if any(abs(a - b) > 1e-3 * max(1.0, abs(a)) for a, b in zip(cur, self.base_checksum)):
            raise RuntimeError("frozen base parameters changed during training")
        return {"grad_norm": float(gnorm), "global_states": global_n,
                "local_loss_mean": local_sum_loss / max(1, local_n)}

    def _skip_update(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.state["skipped_updates"] = int(self.state["skipped_updates"]) + 1
        self.state["consecutive_empty"] = int(self.state["consecutive_empty"]) + 1
        if self.state["consecutive_empty"] >= self.cfg.training.max_consecutive_empty_updates:
            raise RuntimeError(f"{self.state['consecutive_empty']} consecutive updates without a valid state "
                               "(gate never opens or nothing survives filtering); refusing to continue silently")


# ----------------------------------------------------------------------------- stage 1
class ReadoutTrainer(BaseTrainer):
    stage = "readout"

    def __init__(self, gm, cfg, out_dir, rollout_paths: List[str], dist_ctx=None):
        super().__init__(gm, cfg, out_dir, dist_ctx)
        rollouts = [r for p in rollout_paths for r in load_rollouts(p)]
        self.refs = collect_state_refs(rollouts, cfg.training.filter_correct_only, require_gate=False)
        if not self.refs:
            raise RuntimeError("no training states (check filter_correct_only and the rollout files)")
        # each rank takes an interleaved slice so ranks see disjoint states
        self.refs = self.refs[self.dist.rank::self.dist.world_size]
        self.order = torch.randperm(len(self.refs), generator=self.gen_data).tolist()
        if self.dist.rank == 0:
            print(f"[readout] {len(rollouts)} rollouts -> {len(self.refs)} states on rank 0")

    def _next_refs(self, n: int) -> List[StateRef]:
        out = []
        for _ in range(n):
            c = int(self.state["data_cursor"])
            if c >= len(self.order):
                self.order = torch.randperm(len(self.refs), generator=self.gen_data).tolist()
                c = 0
            out.append(self.refs[self.order[c]])
            self.state["data_cursor"] = c + 1
        return out

    def state_loss(self, ref: StateRef) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, attn, ctx, elig_pos = state_tensors(ref.roll, ref.step, self.device)
        with torch.no_grad():
            out = self.gm(x[None], attn[None], context=ctx, mode="off", capture_hidden=True)
            logits, H = out.logits[0], out.hidden_early[0]
        positions = elig_pos
        mx = self.cfg.training.max_positions_per_state
        if positions.numel() > mx:
            keep = torch.randperm(positions.numel(), generator=self.gen_data)[:mx].sort().values.to(self.device)
            positions = positions[keep]
        log_q0 = F.log_softmax(logits[positions].float(), dim=-1)
        log_r = self.gm.readout_log_probs(H[positions])
        kl = (log_q0.exp() * (log_q0 - log_r)).sum(-1)
        with torch.no_grad():
            maxr = log_r.exp().max(-1).values
            agree = (log_r.argmax(-1) == log_q0.argmax(-1)).float()
            src = int(maxr.argmax())
            stats = {"top1_agree": float(agree.mean()), "source_agree": float(agree[src]),
                     "source_maxr": float(maxr[src]), "gate_open_0.8": float(maxr[src] >= 0.8 and positions.numel() >= 2),
                     "n_positions": float(positions.numel())}
        return kl.mean(), stats

    def update(self) -> Dict[str, float]:
        refs = self._next_refs(self.cfg.training.microbatch_states)
        global_n = self.dist.sum_scalar(len(refs), self.device)
        if global_n == 0:
            self._skip_update()
            return {"skipped": 1}
        self.state["consecutive_empty"] = 0
        local_sum, agg = 0.0, {}
        for ref in refs:
            loss, st = self.state_loss(ref)
            (loss / global_n).backward()
            local_sum += float(loss.detach())
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v / len(refs)
        rec = self._apply_update(local_sum, len(refs), global_n)
        rec.update({"loss": local_sum / len(refs), **agg})
        self.log(rec)
        return rec


# ----------------------------------------------------------------------------- stage 2
class GuidanceTrainer(BaseTrainer):
    stage = "guidance"

    def __init__(self, gm, cfg, out_dir, examples: List[Example], tokenizer, dist_ctx=None):
        super().__init__(gm, cfg, out_dir, dist_ctx)
        self.examples = examples[self.dist.rank::self.dist.world_size]
        if not self.examples:
            raise RuntimeError("no examples on this rank")
        self.tokenizer = tokenizer
        self.order = torch.randperm(len(self.examples), generator=self.gen_data).tolist()
        self.buffer: List[StateRef] = []
        self.rollout_dir = os.path.join(out_dir, "rollouts")
        os.makedirs(self.rollout_dir, exist_ok=True)

    def _next_example(self) -> Example:
        c = int(self.state["data_cursor"])
        if c >= len(self.order):
            self.order = torch.randperm(len(self.examples), generator=self.gen_data).tolist()
            c = 0
        self.state["data_cursor"] = c + 1
        return self.examples[self.order[c]]

    @torch.no_grad()
    def rollout(self, ex: Example) -> dict:
        ids, mask = tokenize_prompts(self.tokenizer, [ex.prompt_text], self.device)
        meta = {"example_id": ex.example_id, "prompt_ids": ids[0].tolist(), "prompt_attention_mask": mask[0].tolist(),
                "policy_version": int(self.state["optimizer_step"]), "guidance_mode": "single_soft",
                "generation": self.cfg.generation.__dict__, "gold": ex.answer_value}
        rec = StateRecorder("state_only", self.gm.backend.mask_token_id, self.gm.backend.eos_token_id, ids.shape[1], [meta])
        res = generate(self.gm, ids, mask, self.cfg.generation, guidance_mode="single_soft", recorder=rec,
                       sample_rng=self.gen_sample)
        text = self.tokenizer.decode(res.completion_ids(0), skip_special_tokens=True)
        pred = parse_prediction(text)
        roll = rec.finalize(res, [{"generation": text, "parsed_answer": pred,
                                   "is_correct": is_correct(pred, ex.answer_value), "stats": res.stats}])[0]
        return roll

    def refresh(self) -> Dict[str, float]:
        self.buffer = []
        rolls, kept = [], 0
        for _ in range(self.cfg.training.rollouts_per_update):
            ex = self._next_example()
            chosen = None
            for _attempt in range(self.cfg.training.max_rollout_attempts):
                roll = self.rollout(ex)
                rolls.append(roll)
                if roll["is_correct"] or not self.cfg.training.filter_correct_only:
                    chosen = roll
                    break
            if chosen is None:
                continue
            kept += 1
            cands = [s for s in chosen["steps"] if s["gate_open"] and len(s["eligible_positions"]) >= 2]
            if not cands:
                continue
            k = min(self.cfg.training.states_per_rollout, len(cands))
            pick = torch.randperm(len(cands), generator=self.gen_data)[:k].tolist()
            self.buffer.extend(StateRef(chosen, cands[i], int(self.state["optimizer_step"])) for i in pick)
        rnd = int(self.state["rollout_round"])
        save_rollouts(os.path.join(self.rollout_dir, f"round-{rnd:05d}_rank{self.dist.rank}.pt"), rolls)
        self.state["rollout_round"] = rnd + 1
        n_steps = sum(len(r["steps"]) for r in rolls)
        return {"rollouts": len(rolls), "kept": kept, "correct": sum(r["is_correct"] for r in rolls),
                "gate_open_steps": sum(s["gate_open"] for r in rolls for s in r["steps"]),
                "rollout_steps": n_steps, "buffer_states": len(self.buffer)}

    def state_loss(self, ref: StateRef) -> Tuple[torch.Tensor, Dict[str, float]]:
        step, roll = ref.step, ref.roll
        x, attn, ctx, elig_pos = state_tensors(roll, step, self.device)
        P = roll["meta"]["prompt_len"]
        with torch.no_grad():
            cap = self.gm(x[None], attn[None], context=ctx, mode="readout_only")
        gi = cap.guidance
        A = int(gi.source_pos[0])
        if A != int(step["source_pos"]) or not bool(gi.gate_open[0]):
            raise RuntimeError(f"source re-selection mismatch: recorded A={step['source_pos']} gate={step['gate_open']}, "
                               f"now A={A} gate={bool(gi.gate_open[0])} (readout must be frozen and deterministic)")
        r_A = gi.source_probs[0]
        receivers = elig_pos[elig_pos != A]
        realized = int(roll["final_state"][A - P])
        tgt = teacher_targets(self.gm, x, attn, A, receivers, r_A, realized, self.cfg.teacher, self.gen_teacher)
        spec = GuidanceSpec(source_pos=gi.source_pos, gate_open=gi.gate_open, source_probs=gi.source_probs)
        out = self.gm(x[None], attn[None], context=ctx, mode="single_soft", spec=spec)
        kl = forward_kl(tgt.log_q, out.logits[0, receivers])
        with torch.no_grad():
            p_off = F.softmax(cap.logits[0, A].float(), -1)
            p_on = F.softmax(out.logits[0, A].float(), -1)
            stats = {"n_receivers": float(receivers.numel()), "teacher_forwards": float(tgt.info["teacher_forwards"]),
                     "delta_norm": float(out.guidance.delta_norm[0]), "source_conf_off": float(p_off.max()),
                     "source_conf_on": float(p_on.max()), "source_argmax_changed": float(p_on.argmax() != p_off.argmax()),
                     "readout_argmax_eq_realized": float(int(r_A.argmax()) == realized),
                     "final_argmax_eq_realized": float(int(p_off.argmax()) == realized),
                     "policy_lag": float(int(self.state["optimizer_step"]) - ref.policy_version)}
        return kl.mean(), stats

    def update(self) -> Dict[str, float]:
        rec: Dict[str, float] = {}
        step = int(self.state["optimizer_step"])
        if not self.buffer or step % self.cfg.training.refresh_updates == 0:
            rec.update({f"refresh_{k}": v for k, v in self.refresh().items()})
        refs, self.buffer = self.buffer, []
        global_n = self.dist.sum_scalar(len(refs), self.device)
        if global_n == 0:
            self._skip_update()
            rec["skipped"] = 1
            self.log(rec)
            return rec
        self.state["consecutive_empty"] = 0
        local_sum, agg = 0.0, {}
        for ref in refs:
            loss, st = self.state_loss(ref)
            (loss / global_n).backward()
            local_sum += float(loss.detach())
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v / len(refs)
        rec.update(self._apply_update(local_sum, len(refs), global_n))
        rec.update({"loss": local_sum / max(1, len(refs)), "local_states": len(refs), **agg})
        self.log(rec)
        return rec


# ----------------------------------------------------------------------------- driver
def run_training(trainer: BaseTrainer, max_steps: int, save_every: int, resume: Optional[str] = None) -> Dict[str, float]:
    if resume:
        trainer.resume(resume)
    last: Dict[str, float] = {}
    while int(trainer.state["optimizer_step"]) < max_steps:
        last = trainer.update()
        s = int(trainer.state["optimizer_step"])
        if save_every and s % save_every == 0 and "skipped" not in last:
            trainer.save(metrics=last)
    trainer.save(metrics=last)
    return last
