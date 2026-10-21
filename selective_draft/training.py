"""Shared training machinery (distributed helpers, state tensors, base trainer) and stage 1 (EarlyReadout).

Single process by default; with WORLD_SIZE > 1 the loop uses manual gradient all-reduce
(the modules are ~1M parameters) so that no forward has to go through a DDP wrapper and the
'state count differs per rank' / 'empty rank' cases are handled explicitly at update boundaries."""

from __future__ import annotations

import json
import hashlib
import os
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint, save_snapshot
from .config import Config
from .model import GuidedModel, StepContext
from .trace import load_rollouts


# ----------------------------------------------------------------------------- distributed
class DistCtx:
    def __init__(self, device=None):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1
        if device is not None and device.type == "cuda":
            torch.cuda.set_device(device)
        if self.enabled and not dist.is_initialized():
            backend = "nccl" if device is not None and device.type == "cuda" else "gloo"
            # Ranks only meet at update boundaries; joint updates (K continuations per state) can differ by
            # more than the 10-minute default before the first collective.
            dist.init_process_group(backend, timeout=timedelta(hours=2))
        if dist.is_initialized():
            self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
            self.enabled = self.world_size > 1

    def sync_modules(self, gm):
        if self.enabled:
            for value in gm.module_state_dict().values():
                dist.broadcast(value, src=0)

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


def state_tensors(roll: dict, step: dict, device, commit_threshold: Optional[float] = None,
                  end: Optional[int] = None) -> Tuple[torch.Tensor, StepContext, torch.Tensor]:
    """-> x [1, N], context (B=1), eligible absolute positions [n]. commit_threshold feeds the group rule; `end`
    (absolute) limits groups to positions before the floored generation end, as the decoder did when recording."""
    prompt = torch.tensor(roll["meta"]["prompt_ids"], dtype=torch.long, device=device)
    x = torch.cat([prompt, step["state"].to(device=device, dtype=torch.long)])[None]
    elig_pos = step["eligible_positions"].to(device=device, dtype=torch.long)
    elig = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
    elig[0, elig_pos] = True
    guidance_mask = None if end is None else elig & (torch.arange(x.shape[1], device=device) < int(end))[None]
    ctx = StepContext(eligible_mask=elig, active_rows=torch.ones(1, dtype=torch.bool, device=device),
                      block_start=torch.tensor([step["block_start"]], device=device),
                      block_end=torch.tensor([step["block_end"]], device=device),
                      step_id=torch.tensor([step["step_id"]], device=device),
                      commit_threshold=commit_threshold, guidance_mask=guidance_mask)
    return x, ctx, elig_pos


def collect_state_refs(rollouts: List[dict], filter_correct_only: bool, per_rollout: Optional[int] = None) -> List[StateRef]:
    """per_rollout=k: k states per rollout, uniform without replacement, seeded by the example id."""
    refs = []
    for roll in rollouts:
        if filter_correct_only and not roll.get("is_correct", False):
            continue
        steps = [step for step in roll["steps"] if len(step["eligible_positions"]) > 0]
        if per_rollout is not None and len(steps) > per_rollout:
            seed = int(hashlib.sha256(f"readout-train-v1|{roll['meta'].get('example_id', '')}".encode()).hexdigest()[:15], 16)
            keep = torch.randperm(len(steps), generator=torch.Generator().manual_seed(seed))[:per_rollout]
            steps = [steps[i] for i in sorted(keep.tolist())]
        refs.extend(StateRef(roll, step) for step in steps)
    return refs


# ----------------------------------------------------------------------------- base trainer
class BaseTrainer:
    stage = "none"

    def __init__(self, gm: GuidedModel, cfg: Config, out_dir: str, dist_ctx: Optional[DistCtx] = None):
        self.gm, self.cfg, self.out_dir = gm, cfg, out_dir
        self.device = gm.device
        self.dist = dist_ctx or DistCtx(self.device)
        self.dist.sync_modules(gm)  # manual all-reduce also needs identical initial parameters
        self.params = gm.set_stage(self.stage)
        gm.assert_base_frozen()
        if not self.params:
            raise RuntimeError("no trainable parameters for this stage")
        lr = cfg.training.learning_rate_readout if self.stage == "readout" else cfg.training.learning_rate_guidance
        self.learning_rate = lr
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=cfg.training.weight_decay,
                                          betas=(cfg.training.adam_beta1, cfg.training.adam_beta2))
        seed = cfg.training.seed + 1000 * self.dist.rank
        self.gen_data = torch.Generator().manual_seed(seed)
        self.generators = {"data": self.gen_data}
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
                        self.generators, rank=self.dist.rank, metrics=metrics,
                        sampler_state={"order": list(self.order), "data_signature": self.data_signature})
        if self.dist.rank == 0:
            save_snapshot(os.path.join(self.out_dir, "snapshot"), self.gm, self.cfg, self.stage, metrics)
        self.dist.barrier()
        return path

    def resume(self, path: str) -> None:
        st = load_checkpoint(path, self.gm, self.cfg, self.optimizer, self.generators, rank=self.dist.rank)
        if st.get("stage") != self.stage:
            raise ValueError(f"checkpoint stage {st.get('stage')} != {self.stage}")
        sampler = st.pop("_sampler_state")
        if not sampler or sampler.get("data_signature") != self.data_signature:
            raise ValueError("checkpoint sampler dataset does not match the current rank's data")
        order = sampler["order"]
        if sorted(order) != list(range(len(self.order))):
            raise ValueError("invalid checkpoint sampler permutation")
        if not 0 <= int(st["data_cursor"]) <= len(order):
            raise ValueError("invalid checkpoint data cursor")
        self.order = list(order)
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
        step = int(self.state["optimizer_step"])  # the first update already has a non-zero lr
        lr = self.learning_rate * min(1.0, (step + 1) / self.cfg.training.warmup_updates)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state["optimizer_step"] = int(self.state["optimizer_step"]) + 1
        self.state["states_seen"] = int(self.state["states_seen"]) + local_n
        cur = self.gm.base_checksum()
        if any(abs(a - b) > 1e-3 * max(1.0, abs(a)) for a, b in zip(cur, self.base_checksum)):
            raise RuntimeError("frozen base parameters changed during training")
        return {"grad_norm": float(gnorm), "global_states": global_n, "learning_rate": lr,
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
        self.refs = collect_state_refs(rollouts, cfg.training.filter_correct_only,
                                       per_rollout=cfg.training.readout_states_per_rollout)
        if not self.refs:
            raise RuntimeError("no training states (check filter_correct_only and the rollout files)")
        # each rank takes an interleaved slice so ranks see disjoint states
        self.refs = self.refs[self.dist.rank::self.dist.world_size]
        signature = hashlib.sha256()
        for ref in self.refs:
            signature.update(json.dumps(ref.roll["meta"]["prompt_ids"]).encode())
            signature.update(ref.step["state"].numpy().tobytes())
            signature.update(ref.step["eligible_positions"].numpy().tobytes())
        self.data_signature = signature.hexdigest()
        self.order = torch.randperm(len(self.refs), generator=self.gen_data).tolist()
        if self.dist.rank == 0:
            print(f"[readout] {len(rollouts)} rollouts -> {len(self.refs)} states on rank 0")

    def _next_refs(self, n: int) -> List[StateRef]:
        if not self.refs:
            return []  # another rank can still contribute to the global update
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
        x, ctx, elig_pos = state_tensors(ref.roll, ref.step, self.device)
        with torch.no_grad():
            out = self.gm(x, context=ctx, mode="off", capture_hidden=True)
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
            tau = self.cfg.generation.commit_threshold
            stats = {"top1_agree": float((log_r.argmax(-1) == log_q0.argmax(-1)).float().mean()),
                     "draft_above_tau": float((maxr >= tau).float().mean()), "n_positions": float(positions.numel())}
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
        rec.update({"loss": local_sum / max(1, len(refs)), **agg})
        self.log(rec)
        return rec


# ----------------------------------------------------------------------------- driver
def run_training(trainer: BaseTrainer, max_steps: int, save_every: int, resume: Optional[str] = None) -> Dict[str, float]:
    if resume:
        trainer.resume(resume)
    last: Dict[str, float] = {}
    while int(trainer.state["optimizer_step"]) < max_steps and not getattr(trainer, "finished", False):
        last = trainer.update()
        s = int(trainer.state["optimizer_step"])
        if save_every and s % save_every == 0 and "skipped" not in last:
            trainer.save(metrics=last)
    trainer.save(metrics=last)
    return last
