"""Typed configuration: JSON file + `--set a.b=value` overrides + cross-field validation."""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    path: str = "GSAI-ML/LLaDA-8B-Instruct"
    backend: str = "llada"  # llada | illada | fake_llada
    revision: Optional[str] = None
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    quantization: str = "none"
    trust_remote_code: bool = True
    mask_token_id: Optional[int] = None  # override; required for backends whose config lacks it
    eos_token_id: Optional[int] = None
    fake_kwargs: Dict[str, Any] = field(default_factory=dict)  # fake_llada only


@dataclass
class DataConfig:
    dataset_id: str = "openai/gsm8k"  # "fake" -> built-in toy problems (CPU tests)
    dataset_config: str = "main"
    dataset_revision: Optional[str] = None
    split_manifest: Optional[str] = None  # JSON written by `prepare-data`
    split: str = "dev"  # train | dev | analysis | test
    prompt_recipe: str = "legacy_boxed_reasoning"
    limit: Optional[int] = None  # first N examples of the split (after manifest order)
    manifest_sizes: Dict[str, int] = field(
        default_factory=lambda: {"train": 6000, "dev": 500, "analysis": 500}
    )
    seed: int = 42


@dataclass
class GuidanceConfig:
    early_layer: int = 28  # readout after block index early_layer-1 (1-based layer count)
    readout_rank: int = 64
    injector_rank: int = 64
    message_threshold: float = 0.8
    scale: float = 1.0
    mode: str = "off"  # off | readout_only | single_soft
    readout_position_chunk: int = 8
    message_vocab_chunk: int = 16384


@dataclass
class GenerationConfig:
    mode: str = "threshold"  # threshold | fixed_budget
    commit_threshold: float = 0.9
    steps: int = 128  # fixed_budget only
    max_new_tokens: int = 256
    block_length: int = 32
    temperature: float = 0.0
    window_mode: str = "fixed"  # fixed | growing
    stop_mode: str = "block_end"  # block_end | none
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    autocast: str = "none"  # none | bfloat16 | float16
    skip_empty_budget_steps: bool = True
    forbid_mask_candidate: bool = True  # drop the MASK id from the candidate softmax (base and guided alike)


@dataclass
class TeacherConfig:
    # realized: reveal A as the token A actually receives in the rollout (1 forward).
    # argmax:   reveal A as argmax r_A (1 forward).
    # mc/topk:  the plan's soft mixture; kept for the collapse diagnostic (see the review doc).
    mode: str = "realized"
    mc_samples: int = 4
    topk: int = 4
    candidate_batch_size: int = 1
    seed: int = 1234


@dataclass
class TrainingConfig:
    learning_rate_readout: float = 1e-4
    learning_rate_guidance: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    microbatch_states: int = 1
    states_per_rollout: int = 4
    rollouts_per_update: int = 1
    max_positions_per_state: int = 32  # stage 1: uniform subsample of in-block MASK positions
    filter_correct_only: bool = True
    refresh_updates: int = 1
    activation_checkpointing: bool = False
    max_steps: int = 2
    save_every: int = 0
    log_every: int = 1
    seed: int = 0
    max_rollout_attempts: int = 1  # >1 only meaningful with temperature > 0
    max_consecutive_empty_updates: int = 20  # stage 2 diagnostic exit: gate never opens / nothing survives filtering


@dataclass
class EvaluationConfig:
    batch_size: int = 1
    commit_thresholds: Optional[List[float]] = None  # sweep; None -> generation.commit_threshold
    warmup: int = 1
    repeats: int = 1
    limit: Optional[int] = None


@dataclass
class TraceConfig:
    level: str = "none"  # none | state_only | diagnostic
    topk: int = 8


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    # ---------- IO ----------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section, values in d.items():
            if section.startswith("_"):
                continue  # comments
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section '{section}'")
            sub = getattr(cfg, section)
            for k, v in values.items():
                if not hasattr(sub, k):
                    raise KeyError(f"unknown config field '{section}.{k}'")
                setattr(sub, k, v)
        return cfg

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    # ---------- overrides ----------
    def apply_overrides(self, items: Optional[List[str]]) -> "Config":
        """`--set section.field=value`; value parsed as JSON when possible, else string."""
        for item in items or []:
            if "=" not in item:
                raise ValueError(f"override '{item}' must look like section.field=value")
            key, raw = item.split("=", 1)
            parts = key.split(".")
            if len(parts) != 2:
                raise ValueError(f"override key '{key}' must be section.field")
            section, fname = parts
            if not hasattr(self, section) or not hasattr(getattr(self, section), fname):
                raise KeyError(f"unknown config field '{key}'")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            setattr(getattr(self, section), fname, value)
        return self

    # ---------- validation ----------
    def validate(self) -> "Config":
        m, d, g, gen, t, tr, ev, tc = (self.model, self.data, self.guidance, self.generation,
                                       self.teacher, self.training, self.evaluation, self.trace)
        _choice("model.backend", m.backend, {"llada", "illada", "fake_llada"})
        _choice("model.dtype", m.dtype, {"bfloat16", "float16", "float32"})
        _choice("model.quantization", m.quantization, {"none"})
        _choice("data.split", d.split, {"train", "dev", "analysis", "test"})
        _choice("data.prompt_recipe", d.prompt_recipe, {"legacy_boxed_reasoning", "fake"})
        _choice("guidance.mode", g.mode, {"off", "readout_only", "single_soft"})
        if g.early_layer < 1:
            raise ValueError("guidance.early_layer must be >= 1 (1-based count of completed blocks)")
        if not (0.0 <= g.message_threshold <= 1.0):
            raise ValueError("guidance.message_threshold must be in [0, 1]")
        if g.readout_rank < 1 or g.injector_rank < 1:
            raise ValueError("ranks must be >= 1")
        if g.readout_position_chunk < 1 or g.message_vocab_chunk < 1:
            raise ValueError("chunks must be >= 1")
        _choice("generation.mode", gen.mode, {"threshold", "fixed_budget"})
        if not (0.0 <= gen.commit_threshold <= 1.0):
            raise ValueError("generation.commit_threshold must be in [0, 1]")
        if gen.max_new_tokens < 1 or gen.block_length < 1 or gen.max_new_tokens % gen.block_length:
            raise ValueError("generation.max_new_tokens must be a positive multiple of block_length")
        num_blocks = gen.max_new_tokens // gen.block_length
        if gen.mode == "fixed_budget":
            if gen.steps < 1 or gen.steps % num_blocks:
                raise ValueError(
                    f"generation.steps={gen.steps} must be a positive multiple of num_blocks={num_blocks}"
                )
        if gen.temperature < 0:
            raise ValueError("generation.temperature must be >= 0")
        _choice("generation.window_mode", gen.window_mode, {"fixed", "growing"})
        _choice("generation.stop_mode", gen.stop_mode, {"block_end", "none"})
        if gen.cfg_scale != 0.0:
            raise ValueError("classifier-free guidance is not supported by this pipeline (cfg_scale must be 0)")
        if gen.remasking != "low_confidence":
            raise ValueError("only low_confidence remasking is supported")
        _choice("generation.autocast", gen.autocast, {"none", "bfloat16", "float16"})
        _choice("teacher.mode", t.mode, {"realized", "argmax", "mc", "topk"})
        if t.mc_samples < 1 or t.topk < 1 or t.candidate_batch_size < 1:
            raise ValueError("teacher.mc_samples / topk / candidate_batch_size must be >= 1")
        if tr.microbatch_states < 1 or tr.states_per_rollout < 1 or tr.rollouts_per_update < 1:
            raise ValueError("training batch sizes must be >= 1")
        if tr.refresh_updates < 1:
            raise ValueError("training.refresh_updates must be >= 1")
        if tr.max_rollout_attempts < 1:
            raise ValueError("training.max_rollout_attempts must be >= 1")
        if tr.max_rollout_attempts > 1 and gen.temperature == 0.0:
            raise ValueError("max_rollout_attempts > 1 needs temperature > 0 (greedy retries are identical)")
        if tr.activation_checkpointing:
            raise ValueError("activation checkpointing is disabled in this version (see plan §3.2)")
        if ev.batch_size < 1 or ev.warmup < 0 or ev.repeats < 1:
            raise ValueError("evaluation.batch_size>=1, warmup>=0, repeats>=1")
        for thr in ev.commit_thresholds or []:
            if not (0.0 <= float(thr) <= 1.0):
                raise ValueError("evaluation.commit_thresholds entries must be in [0, 1]")
        _choice("trace.level", tc.level, {"none", "state_only", "diagnostic"})
        if tc.topk < 1:
            raise ValueError("trace.topk must be >= 1")
        return self


def _choice(name: str, value: Any, allowed: set) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {sorted(allowed)}")
