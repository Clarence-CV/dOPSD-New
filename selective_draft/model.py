"""FrozenBaseWithGuidance: the frozen base model plus the two trainable modules, wired through a
scoped forward hook on the early-layer block. Modes:
  off           native forward, no hook (exact base output);
  readout_only  hook computes the early readout and picks the source, but does not touch hidden;
  single_soft   additionally adds the message at the source position when the gate is open.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends.base import Backend
from .config import GuidanceConfig
from .modules import EarlyReadoutAdapter, MessageInjector, soft_message

MODES = ("off", "readout_only", "single_soft")


@dataclass
class StepContext:
    eligible_mask: torch.Tensor  # [B, N] bool: current block & MASK & live row
    active_rows: torch.Tensor  # [B] bool
    block_start: torch.Tensor  # [B] long, absolute, inclusive
    block_end: torch.Tensor  # [B] long, exclusive
    step_id: torch.Tensor  # [B] long, actual denoising step


@dataclass
class GuidanceSpec:
    """Externally fixed guidance for one forward (training re-uses the choice made on the same state)."""
    source_pos: torch.Tensor  # [B] long, -1 = none
    gate_open: torch.Tensor  # [B] bool
    source_probs: Optional[torch.Tensor] = None  # [B, V] fp32 (rows with closed gate ignored)
    override_delta: Optional[torch.Tensor] = None  # [B, d]: replaces the injector output (oracle patch)


@dataclass
class GuidanceInfo:
    source_pos: torch.Tensor  # [B] long
    source_confidence: torch.Tensor  # [B] fp32, max r_A (0 when no source)
    gate_open: torch.Tensor  # [B] bool
    skip_reason: List[str]  # open / below_threshold / no_receiver / no_eligible / inactive / disabled
    n_eligible: torch.Tensor  # [B] long
    source_probs: Optional[torch.Tensor] = None  # [B, V] fp32, detached; only rows with gate open are filled
    delta_norm: Optional[torch.Tensor] = None  # [B] fp32 (single_soft only)
    source_argmax: Optional[torch.Tensor] = None  # [B] long, argmax r_A (-1 when no source)


@dataclass
class StepOutput:
    logits: torch.Tensor  # [B, N, V] raw final logits (after guidance when enabled)
    guidance: Optional[GuidanceInfo]
    hidden_early: Optional[torch.Tensor] = None  # [B, N, d] detached boundary hidden (before injection)


@dataclass
class _Scratch:
    mode: str
    context: Optional[StepContext]
    spec: Optional[GuidanceSpec]
    capture: bool
    info: Optional[GuidanceInfo] = None
    hidden: Optional[torch.Tensor] = None
    fired: int = 0
    extra: dict = field(default_factory=dict)


class GuidedModel(nn.Module):
    def __init__(self, base: nn.Module, backend: Backend, cfg: GuidanceConfig):
        super().__init__()
        self.base = base
        self.backend = backend
        self.cfg = cfg
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        device = next(self.base.parameters()).device
        self.readout = EarlyReadoutAdapter(backend.d_model, cfg.readout_rank).to(device=device, dtype=torch.float32)
        self.injector = MessageInjector(backend.d_model, cfg.injector_rank).to(device=device, dtype=torch.float32)
        self._scratch: Optional[_Scratch] = None
        self.backend.boundary_module(cfg.early_layer)  # validates the layer index now

    # ------------------------------------------------------------------ housekeeping
    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # frozen base never leaves eval (dropout stays off)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def set_stage(self, stage: Optional[str]) -> List[nn.Parameter]:
        """stage: 'readout' | 'guidance' | None (nothing trainable). Returns the trainable params."""
        for p in self.readout.parameters():
            p.requires_grad_(stage == "readout")
        for p in self.injector.parameters():
            p.requires_grad_(stage == "guidance")
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def assert_base_frozen(self) -> None:
        bad = [n for n, p in self.base.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"base parameters with requires_grad=True: {bad[:5]}")

    @torch.no_grad()
    def base_checksum(self, n: int = 8) -> List[float]:
        out = []
        for i, p in enumerate(self.base.parameters()):
            if i % max(1, sum(1 for _ in self.base.parameters()) // n) == 0:
                out.append(float(p.float().sum()))
            if len(out) >= n:
                break
        return out

    def module_state_dict(self) -> dict:
        d = {f"readout.{k}": v for k, v in self.readout.state_dict().items()}
        d.update({f"injector.{k}": v for k, v in self.injector.state_dict().items()})
        return d

    def load_module_state_dict(self, sd: dict, strict: bool = True) -> None:
        r = {k[len("readout."):]: v for k, v in sd.items() if k.startswith("readout.")}
        j = {k[len("injector."):]: v for k, v in sd.items() if k.startswith("injector.")}
        if strict and (not r or not j):
            raise KeyError("module state dict must contain both readout.* and injector.*")
        for name, mod, part in (("readout", self.readout, r), ("injector", self.injector, j)):
            for k, v in part.items():
                cur = mod.state_dict()[k]
                if tuple(cur.shape) != tuple(v.shape):
                    raise ValueError(f"{name}.{k}: shape {tuple(v.shape)} != {tuple(cur.shape)}")
            mod.load_state_dict(part, strict=True)

    # ------------------------------------------------------------------ forward paths
    def base_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher / parity path: native forward, no hook installed."""
        if self._scratch is not None:
            raise RuntimeError("base_forward called while a guided forward is in progress")
        return self.backend.forward_logits(input_ids, attention_mask)

    def readout_log_probs(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """[n, d] boundary hidden -> [n, V] fp32 log-probs through g_phi + frozen norm/head (grad kept)."""
        outs = []
        for chunk in hidden_rows.split(self.cfg.readout_position_chunk):
            logits = self.backend.project(self.readout(chunk))
            outs.append(F.log_softmax(logits.float(), dim=-1))
        return torch.cat(outs, dim=0)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                context: Optional[StepContext] = None, mode: Optional[str] = None,
                spec: Optional[GuidanceSpec] = None, capture_hidden: bool = False) -> StepOutput:
        mode = self.cfg.mode if mode is None else mode
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if mode == "off" and not capture_hidden:
            return StepOutput(self.base_forward(input_ids, attention_mask), None, None)
        if mode != "off" and context is None:
            raise ValueError("context is required unless mode='off'")
        scratch = _Scratch(mode=mode, context=context, spec=spec, capture=capture_hidden)
        with self._scoped_hook(scratch):
            logits = self.backend.forward_logits(input_ids, attention_mask)
        if scratch.fired != 1:
            raise RuntimeError(f"boundary hook fired {scratch.fired} times (expected 1)")
        return StepOutput(logits, scratch.info, scratch.hidden)

    @contextmanager
    def _scoped_hook(self, scratch: _Scratch):
        if self._scratch is not None:
            raise RuntimeError("nested guided forwards are not supported on one instance")
        handle = self.backend.boundary_module(self.cfg.early_layer).register_forward_hook(self._hook)
        self._scratch = scratch
        try:
            yield
        finally:
            handle.remove()
            self._scratch = None

    # ------------------------------------------------------------------ the hook
    def _hook(self, module, inputs, output):
        sc = self._scratch
        sc.fired += 1
        h = self.backend.block_hidden(output)
        if sc.capture:
            sc.hidden = h.detach()
        if sc.mode == "off":
            return None
        info = self._select_source(h, sc.context) if sc.spec is None else self._info_from_spec(sc.spec, sc.context)
        sc.info = info
        if sc.mode == "readout_only":
            return None
        rows = info.gate_open.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            return None
        pos = info.source_pos[rows]
        if sc.spec is not None and sc.spec.override_delta is not None:
            delta = sc.spec.override_delta[rows].to(h.dtype)
        else:
            probs = info.source_probs[rows]
            msg = soft_message(probs, self.backend.embedding_weight(), self.cfg.message_vocab_chunk)
            delta = (self.injector(h[rows, pos].detach(), msg) * self.cfg.scale).to(h.dtype)
        add = torch.zeros_like(h).index_put((rows, pos), delta)  # out-of-place, differentiable in delta
        norms = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
        norms[rows] = delta.detach().float().norm(dim=-1)
        info.delta_norm = norms
        return self.backend.with_block_hidden(output, h + add)

    @torch.no_grad()
    def _select_source(self, h: torch.Tensor, ctx: StepContext) -> GuidanceInfo:
        B, N, _ = h.shape
        elig = ctx.eligible_mask & ctx.active_rows[:, None]
        n_elig = elig.sum(dim=1)
        source_pos = torch.full((B,), -1, dtype=torch.long, device=h.device)
        source_conf = torch.zeros(B, dtype=torch.float32, device=h.device)
        source_argmax = torch.full((B,), -1, dtype=torch.long, device=h.device)
        gate = torch.zeros(B, dtype=torch.bool, device=h.device)
        probs_out: Optional[torch.Tensor] = None
        reasons: List[str] = []
        for b in range(B):
            if not bool(ctx.active_rows[b]):
                reasons.append("inactive")
                continue
            idx = elig[b].nonzero(as_tuple=False).flatten()  # ascending absolute positions
            if idx.numel() == 0:
                reasons.append("no_eligible")
                continue
            best_p, best_i, best_probs = -1.0, -1, None
            for chunk in idx.split(self.cfg.readout_position_chunk):
                logits = self.backend.project(self.readout(h[b, chunk]))
                probs = torch.softmax(logits.float(), dim=-1)
                mp = probs.max(dim=-1).values
                j = int(mp.argmax())  # first maximum -> smallest position on ties
                if float(mp[j]) > best_p:  # strict: earlier chunk wins ties
                    best_p, best_i, best_probs = float(mp[j]), int(chunk[j]), probs[j]
            source_pos[b] = best_i
            source_conf[b] = best_p
            source_argmax[b] = int(best_probs.argmax())
            if idx.numel() < 2:
                reasons.append("no_receiver")
                continue
            if best_p >= self.cfg.message_threshold:
                gate[b] = True
                reasons.append("open")
                if probs_out is None:
                    probs_out = torch.zeros(B, best_probs.shape[-1], dtype=torch.float32, device=h.device)
                probs_out[b] = best_probs
            else:
                reasons.append("below_threshold")
        return GuidanceInfo(source_pos, source_conf, gate, reasons, n_elig, probs_out, None, source_argmax)

    @staticmethod
    def _info_from_spec(spec: GuidanceSpec, ctx: Optional[StepContext]) -> GuidanceInfo:
        B = spec.source_pos.shape[0]
        n_elig = (ctx.eligible_mask & ctx.active_rows[:, None]).sum(1) if ctx is not None else torch.zeros(B, dtype=torch.long)
        conf = torch.zeros(B, dtype=torch.float32, device=spec.source_pos.device)
        argmax = torch.full((B,), -1, dtype=torch.long, device=spec.source_pos.device)
        if spec.source_probs is not None:
            conf = spec.source_probs.max(dim=-1).values.float()
            argmax = spec.source_probs.argmax(dim=-1)
        reasons = ["open" if bool(g) else "spec_closed" for g in spec.gate_open]
        return GuidanceInfo(spec.source_pos, conf, spec.gate_open, reasons, n_elig, spec.source_probs, None, argmax)
