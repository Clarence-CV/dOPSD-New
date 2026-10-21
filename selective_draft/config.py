"""Typed configuration: JSON file + `--set section.field=value` overrides + cross-field validation."""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    path: str = "inclusionAI/LLaDA2.0-mini"
    backend: str = "llada2_mini"  # llada2_mini | fake_llada2 (CPU tests)
    revision: Optional[str] = None
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # sdpa (default, explicit 4D block mask) | eager (reference)
    mask_token_id: Optional[int] = None  # cross-checked against the tokenizer when set
    eos_token_id: Optional[int] = None
    fake_kwargs: Dict[str, Any] = field(default_factory=dict)  # fake_llada2 only


@dataclass
class DataConfig:
    task: str = "math"  # math | fake
    dataset_id: str = "ankner/math-500"  # "fake" -> built-in toy problems (CPU tests)
    dataset_revision: Optional[str] = None
    test_dataset_id: Optional[str] = "HuggingFaceH4/MATH-500"  # separate source for the test split
    test_dataset_revision: Optional[str] = None
    split_manifest: Optional[str] = None  # JSON written by `prepare-data`
    split: str = "dev"  # train | dev | analysis | test
    subsplit: Optional[str] = None  # data.SUBSPLITS, e.g. analysis:readout_val, dev:dev_fast
    prompt_recipe: str = "dmax_boxed"  # dmax_boxed | fake
    limit: Optional[int] = None  # first N examples of the split (after manifest order)
    manifest_sizes: Dict[str, int] = field(default_factory=lambda: {"train": 6000, "dev": 500, "analysis": 500})
    seed: int = 42


@dataclass
class GuidanceConfig:
    early_layer: Optional[int] = None  # readout after block index early_layer-1 (1-based); null = only mode off
    readout_rank: int = 64
    injector_rank: int = 64
    scale: float = 1.0
    mode: str = "off"  # off | group_readout | group_soft
    readout_position_chunk: int = 8
    message_vocab_chunk: int = 16384


@dataclass
class GenerationConfig:
    """DMax / dInfer evaluation semantics (see generation.py for the exact rules)."""
    commit_threshold: float = 0.95
    max_new_tokens: int = 2048  # total length is floored to a multiple of block_length, as dInfer's harness does
    block_length: int = 32
    temperature: float = 0.0
    early_stop: bool = True  # stop after the block that contains an EOS
    cache_mode: str = "none"  # none | prefix (dInfer-style prefix KV cache; guidance off only)
    dinfer_expected_tpf: int = 4  # dInfer eval script (expected_tpf / maximum_unroll): only its NFE accounting
    dinfer_unroll: int = 4


@dataclass
class TrainingConfig:
    learning_rate_readout: float = 1e-4
    learning_rate_guidance: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    warmup_updates: int = 5  # lr = base * min((step+1)/warmup, 1)
    microbatch_states: int = 1  # readout: states per rank per update
    readout_states_per_rollout: Optional[int] = 1  # readout: states drawn per recorded rollout (null = all)
    max_positions_per_state: int = 32  # readout: uniform subsample of in-block MASK positions
    filter_correct_only: bool = False  # readout: keep only correct rollouts
    loo_batch_size: int = 1  # leave-one-out Teacher inputs per forward
    rollout_temperature: float = 0.9  # joint root rollouts (training only)
    max_rollout_attempts: int = 8  # joint: root rollouts per question incl. the first; stop at first correct
    max_steps: int = 50
    save_every: int = 25
    log_every: int = 1
    seed: int = 42
    max_consecutive_empty_updates: int = 200


@dataclass
class EvaluationConfig:
    commit_thresholds: Optional[List[float]] = None  # sweep; None -> generation.commit_threshold
    warmup: int = 1
    repeats: int = 1
    limit: Optional[int] = None


@dataclass
class TraceConfig:
    level: str = "none"  # none | state_only


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
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
        m, d, g, gen, tr, ev, tc = (self.model, self.data, self.guidance, self.generation, self.training,
                                    self.evaluation, self.trace)
        _choice("model.backend", m.backend, {"llada2_mini", "fake_llada2"})
        _choice("model.dtype", m.dtype, {"bfloat16", "float16", "float32"})
        if m.attn_implementation is not None:
            _choice("model.attn_implementation", m.attn_implementation, {"sdpa", "eager"})
        _choice("data.split", d.split, {"train", "dev", "analysis", "test"})
        _choice("data.task", d.task, {"math", "fake"})
        _choice("data.prompt_recipe", d.prompt_recipe, {"dmax_boxed", "fake"})
        if (d.dataset_id == "fake") != (d.task == "fake"):
            raise ValueError("data.task=fake goes with data.dataset_id=fake and vice versa")
        if d.subsplit is not None:
            from .data import SUBSPLITS
            if d.split not in SUBSPLITS or d.subsplit not in dict(SUBSPLITS[d.split][1]):
                raise ValueError(f"data.subsplit={d.subsplit!r} is not a sub-split of data.split={d.split!r}")
        _choice("guidance.mode", g.mode, {"off", "group_readout", "group_soft"})
        if g.early_layer is None:
            if g.mode != "off":
                raise ValueError("guidance.early_layer is unresolved (null): only guidance.mode=off can run; "
                                 "set it from layer_decision.json")
        elif g.early_layer < 1:
            raise ValueError("guidance.early_layer must be >= 1 (1-based count of completed blocks)")
        if g.readout_rank < 1 or g.injector_rank < 1:
            raise ValueError("ranks must be >= 1")
        if g.readout_position_chunk < 1 or g.message_vocab_chunk < 1:
            raise ValueError("chunks must be >= 1")
        if not (0.0 <= gen.commit_threshold <= 1.0):
            raise ValueError("generation.commit_threshold must be in [0, 1]")
        if gen.max_new_tokens < 1 or gen.block_length < 1:
            raise ValueError("generation.max_new_tokens and block_length must be positive")
        if gen.temperature < 0:
            raise ValueError("generation.temperature must be >= 0")
        _choice("generation.cache_mode", gen.cache_mode, {"none", "prefix"})
        if gen.dinfer_expected_tpf < 1 or gen.dinfer_unroll < 1:
            raise ValueError("generation.dinfer_expected_tpf / dinfer_unroll must be >= 1")
        if tr.microbatch_states < 1 or tr.loo_batch_size < 1 or tr.max_rollout_attempts < 1:
            raise ValueError("training batch sizes and attempts must be >= 1")
        if tr.warmup_updates < 1:
            raise ValueError("training.warmup_updates must be >= 1")
        if not (0 <= tr.adam_beta1 < 1 and 0 <= tr.adam_beta2 < 1):
            raise ValueError("invalid Adam betas")
        if tr.max_rollout_attempts > 1 and tr.rollout_temperature <= 0.0:
            raise ValueError("max_rollout_attempts > 1 needs training.rollout_temperature > 0 (greedy retries are identical)")
        if ev.warmup < 0 or ev.repeats < 1:
            raise ValueError("evaluation.warmup>=0, repeats>=1")
        for thr in ev.commit_thresholds or []:
            if not (0.0 <= float(thr) <= 1.0):
                raise ValueError("evaluation.commit_thresholds entries must be in [0, 1]")
        _choice("trace.level", tc.level, {"none", "state_only"})
        return self


def _choice(name: str, value: Any, allowed: set) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {sorted(allowed)}")
