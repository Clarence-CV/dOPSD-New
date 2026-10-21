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
    backend: str = "llada"  # llada | illada | llada2_mini | fake_llada | fake_llada2
    revision: Optional[str] = None
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    quantization: str = "none"
    trust_remote_code: bool = True
    mask_token_id: Optional[int] = None  # override; required for backends whose config lacks it
    eos_token_id: Optional[int] = None
    attn_implementation: Optional[str] = None  # llada2_mini: sdpa (explicit 4D block mask) | eager (reference)
    fake_kwargs: Dict[str, Any] = field(default_factory=dict)  # fake_llada / fake_llada2 only


@dataclass
class DataConfig:
    task: str = "gsm8k"  # gsm8k | math | fake  (answer extraction + equivalence)
    dataset_id: str = "openai/gsm8k"  # "fake" -> built-in toy problems (CPU tests); math: "ankner/math-500"
    dataset_config: Optional[str] = "main"  # null for ankner/math-500
    dataset_revision: Optional[str] = None
    test_dataset_id: Optional[str] = None  # separate source for the test split (math: "HuggingFaceH4/MATH-500", as THU eval)
    test_dataset_revision: Optional[str] = None  # pinned separately from dataset_revision (plan §6.1)
    split_manifest: Optional[str] = None  # JSON written by `prepare-data`
    split: str = "dev"  # train | dev | analysis | test
    subsplit: Optional[str] = None  # data.SUBSPLITS, e.g. analysis:readout_val, dev:dev_fast (plan §6.5)
    prompt_recipe: str = "legacy_boxed_reasoning"
    scorer: str = "legacy"  # legacy (eval/parser_helper is_equiv) | klass (KLASS extract_math_answer/compare_answers)
    limit: Optional[int] = None  # first N examples of the split (after manifest order)
    manifest_sizes: Dict[str, int] = field(
        default_factory=lambda: {"train": 6000, "dev": 500, "analysis": 500}
    )
    seed: int = 42


@dataclass
class GuidanceConfig:
    early_layer: Optional[int] = 28  # readout after block index early_layer-1 (1-based); null = only mode off
    readout_rank: int = 64
    injector_rank: int = 64
    message_threshold: float = 0.8
    scale: float = 1.0
    mode: str = "off"  # off | readout_only | single_soft | group_readout | group_soft
    group_size: int = 4  # group_policy=topk only: highest-confidence MASK positions, no single-source gate
    # topk (Joint-v1) | confidence_threshold_all: every eligible position whose early-readout candidate has raw
    # confidence > the decoder's commit threshold (passed per step, never a second number); < 2 -> no message.
    group_policy: str = "topk"
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
    window_mode: str = "fixed"  # fixed | growing | current_block_prefix (block-causal models)
    stop_mode: str = "block_end"  # block_end | none | prefix_complete_eos (block_origin=sequence_zero only)
    # prompt_end: blocks start at the first generated position (LLaDA). sequence_zero: blocks are absolute
    # 32-position cells counted from position 0, so the prompt tail can share the first block (LLaDA2.0).
    block_origin: str = "prompt_end"
    steps_per_block: Optional[int] = None  # sequence_zero: HF `steps` (per block); must equal block_length
    # The four switches below exist for block_origin=sequence_zero only; defaults keep the HF-generate behaviour.
    # commit_rule  hf: conf > tau else the single best (next-best token if the argmax is MASK);
    #              dinfer: DMax's dInfer rule, conf >= min(tau, max-1e-5), positions whose argmax is MASK excluded.
    # length_policy cap: cap = max_new_tokens (physical end rounded up); dinfer_floor32: total length
    #              floor((P+G)/block_length)*block_length, as dInfer's eval harness buckets it.
    # output_view  cut_at_eos: completion ends at the first EOS; drop_eos: every generated non-EOS token (dInfer).
    # cache_mode   prefix: dInfer-style prefix KV cache + cross-block sealing (guidance off only; else no cache).
    commit_rule: str = "hf"
    length_policy: str = "cap"
    output_view: str = "cut_at_eos"
    cache_mode: str = "none"
    dinfer_expected_tpf: int = 4  # dInfer eval script: expected_tpf / maximum_unroll, used for its NFE accounting
    dinfer_unroll: int = 4
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    autocast: str = "none"  # none | bfloat16 | float16
    skip_empty_budget_steps: bool = True
    forbid_mask_candidate: bool = True  # forbid selecting MASK; confidence uses the original softmax


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
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_ratio: float = 0.0  # linear warmup, then constant learning rate
    microbatch_states: int = 1
    states_per_rollout: int = 4
    rollouts_per_update: int = 1
    max_positions_per_state: int = 32  # stage 1: uniform subsample of in-block MASK positions
    readout_states_per_rollout: Optional[int] = None  # stage 1: k states per recorded rollout (null = all states)
    filter_correct_only: bool = True
    refresh_updates: int = 1
    activation_checkpointing: bool = False
    max_steps: int = 2
    save_every: int = 0
    log_every: int = 1
    seed: int = 0
    max_rollout_attempts: int = 1  # legacy: retries until correct; joint v1: K same-state continuations;
    # joint root_success_v2: root rollouts from the prompt per question (incl. the first), stop at first correct
    sampling_protocol: str = "same_state_v1"  # same_state_v1 | root_success_v2 (plan §6.4)
    warmup_updates: Optional[int] = None  # absolute warmup, lr = base * min((step+1)/warmup, 1); overrides ratio
    rollout_temperature: Optional[float] = None  # stage-2 training rollouts only; None -> generation.temperature
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

    def rollout_temperature(self) -> float:
        t = self.training.rollout_temperature
        return float(self.generation.temperature if t is None else t)

    # ---------- validation ----------
    def validate(self) -> "Config":
        m, d, g, gen, t, tr, ev, tc = (self.model, self.data, self.guidance, self.generation,
                                       self.teacher, self.training, self.evaluation, self.trace)
        _choice("model.backend", m.backend, {"llada", "illada", "llada2_mini", "fake_llada", "fake_llada2"})
        block_causal = m.backend in ("llada2_mini", "fake_llada2")
        if m.attn_implementation is not None:
            _choice("model.attn_implementation", m.attn_implementation, {"sdpa", "eager"})
        _choice("model.dtype", m.dtype, {"bfloat16", "float16", "float32"})
        _choice("model.quantization", m.quantization, {"none"})
        _choice("data.split", d.split, {"train", "dev", "analysis", "test"})
        _choice("data.task", d.task, {"gsm8k", "math", "fake"})
        _choice("data.prompt_recipe", d.prompt_recipe, {"legacy_boxed_reasoning", "legacy_math_boxed_reasoning",
                                                      "klass_final_answer", "boxed_step_by_step", "dmax_boxed", "fake"})
        _choice("data.scorer", d.scorer, {"legacy", "klass", "dmax"})
        if d.subsplit is not None:
            from .data import SUBSPLITS
            if d.split not in SUBSPLITS or d.subsplit not in dict(SUBSPLITS[d.split][1]):
                raise ValueError(f"data.subsplit={d.subsplit!r} is not a sub-split of data.split={d.split!r}")
        if d.scorer == "klass" and d.task not in ("math", "fake"):
            raise ValueError("data.scorer=klass is implemented for MATH only")
        if (d.dataset_id == "fake") != (d.task == "fake"):
            raise ValueError("data.task=fake goes with data.dataset_id=fake and vice versa")
        _choice("guidance.mode", g.mode, {"off", "readout_only", "single_soft", "group_readout", "group_soft"})
        _choice("guidance.group_policy", g.group_policy, {"topk", "confidence_threshold_all"})
        if g.group_size < 2:
            raise ValueError("guidance.group_size must be >= 2")
        if g.early_layer is None:
            if g.mode != "off":
                raise ValueError("guidance.early_layer is unresolved (null): only guidance.mode=off can run; "
                                 "set it from layer_decision.json (plan §7.1)")
        elif g.early_layer < 1:
            raise ValueError("guidance.early_layer must be >= 1 (1-based count of completed blocks)")
        if g.group_policy == "confidence_threshold_all" and gen.mode != "threshold":
            raise ValueError("group_policy=confidence_threshold_all is bound to the threshold decoder's tau")
        if not (0.0 <= g.message_threshold <= 1.0):
            raise ValueError("guidance.message_threshold must be in [0, 1]")
        if g.readout_rank < 1 or g.injector_rank < 1:
            raise ValueError("ranks must be >= 1")
        if g.readout_position_chunk < 1 or g.message_vocab_chunk < 1:
            raise ValueError("chunks must be >= 1")
        _choice("generation.mode", gen.mode, {"threshold", "fixed_budget"})
        if not (0.0 <= gen.commit_threshold <= 1.0):
            raise ValueError("generation.commit_threshold must be in [0, 1]")
        _choice("generation.block_origin", gen.block_origin, {"prompt_end", "sequence_zero"})
        if gen.block_origin == "sequence_zero":
            # Block grid counted from position 0; the last block is rounded up physically (plan §5.4).
            if gen.max_new_tokens < 1 or gen.block_length < 1:
                raise ValueError("generation.max_new_tokens and block_length must be positive")
            if gen.window_mode != "current_block_prefix":
                raise ValueError("block_origin=sequence_zero requires window_mode=current_block_prefix")
            if gen.mode != "threshold":
                raise ValueError("block_origin=sequence_zero currently implements the threshold decoder only")
            if gen.steps_per_block != gen.block_length:
                raise ValueError("steps_per_block must equal block_length (minimum one commit per step, plan §5.3)")
            if ev.batch_size != 1:
                raise ValueError("block_origin=sequence_zero is batch-1 (padding would shift the absolute grid)")
        elif gen.max_new_tokens < 1 or gen.block_length < 1 or gen.max_new_tokens % gen.block_length:
            raise ValueError("generation.max_new_tokens must be a positive multiple of block_length")
        if block_causal != (gen.block_origin == "sequence_zero"):
            raise ValueError("llada2 backends go with generation.block_origin=sequence_zero and vice versa")
        _choice("generation.commit_rule", gen.commit_rule, {"hf", "dinfer"})
        _choice("generation.length_policy", gen.length_policy, {"cap", "dinfer_floor32"})
        _choice("generation.output_view", gen.output_view, {"cut_at_eos", "drop_eos"})
        _choice("generation.cache_mode", gen.cache_mode, {"none", "prefix"})
        if gen.dinfer_expected_tpf < 1 or gen.dinfer_unroll < 1:
            raise ValueError("generation.dinfer_expected_tpf / dinfer_unroll must be >= 1")
        if gen.block_origin != "sequence_zero" and (gen.commit_rule, gen.length_policy, gen.output_view,
                                                    gen.cache_mode) != ("hf", "cap", "cut_at_eos", "none"):
            raise ValueError("commit_rule/length_policy/output_view/cache_mode are implemented for "
                             "block_origin=sequence_zero only")
        if gen.stop_mode == "prefix_complete_eos" and gen.block_origin != "sequence_zero":
            raise ValueError("stop_mode=prefix_complete_eos is implemented for block_origin=sequence_zero")
        num_blocks = max(1, gen.max_new_tokens // gen.block_length)
        if gen.mode == "fixed_budget":
            if gen.steps < 1 or gen.steps % num_blocks:
                raise ValueError(
                    f"generation.steps={gen.steps} must be a positive multiple of num_blocks={num_blocks}"
                )
        if gen.temperature < 0:
            raise ValueError("generation.temperature must be >= 0")
        _choice("generation.window_mode", gen.window_mode, {"fixed", "growing", "current_block_prefix"})
        if gen.window_mode == "current_block_prefix" and gen.block_origin != "sequence_zero":
            raise ValueError("window_mode=current_block_prefix requires block_origin=sequence_zero")
        _choice("generation.stop_mode", gen.stop_mode, {"block_end", "none", "prefix_complete_eos"})
        if gen.cfg_scale != 0.0:
            raise ValueError("classifier-free guidance is not supported by this pipeline (cfg_scale must be 0)")
        if gen.remasking != "low_confidence":
            raise ValueError("only low_confidence remasking is supported")
        _choice("generation.autocast", gen.autocast, {"none", "bfloat16", "float16"})
        _choice("teacher.mode", t.mode, {"realized", "argmax", "mc", "topk", "joint"})
        if t.mode == "joint" and gen.mode != "threshold":
            raise ValueError("teacher.mode=joint requires threshold decoding")
        if t.mc_samples < 1 or t.topk < 1 or t.candidate_batch_size < 1:
            raise ValueError("teacher.mc_samples / topk / candidate_batch_size must be >= 1")
        if tr.microbatch_states < 1 or tr.states_per_rollout < 1 or tr.rollouts_per_update < 1:
            raise ValueError("training batch sizes must be >= 1")
        if tr.refresh_updates < 1:
            raise ValueError("training.refresh_updates must be >= 1")
        if not (0 <= tr.adam_beta1 < 1 and 0 <= tr.adam_beta2 < 1 and 0 <= tr.warmup_ratio <= 1):
            raise ValueError("invalid Adam betas or warmup_ratio")
        if tr.max_rollout_attempts < 1:
            raise ValueError("training.max_rollout_attempts must be >= 1")
        if tr.rollout_temperature is not None and tr.rollout_temperature < 0:
            raise ValueError("training.rollout_temperature must be >= 0")
        if tr.max_rollout_attempts > 1 and self.rollout_temperature() == 0.0:
            raise ValueError("max_rollout_attempts > 1 needs a rollout temperature > 0 (greedy retries are identical)")
        _choice("training.sampling_protocol", tr.sampling_protocol, {"same_state_v1", "root_success_v2"})
        if tr.warmup_updates is not None and tr.warmup_updates < 1:
            raise ValueError("training.warmup_updates must be >= 1 (or null)")
        if tr.sampling_protocol == "root_success_v2":
            # guidance.mode is checked by the trainer: the same file also drives off/readout evaluations.
            if t.mode != "joint" or g.group_policy != "confidence_threshold_all":
                raise ValueError("root_success_v2 requires teacher.mode=joint and "
                                 "guidance.group_policy=confidence_threshold_all")
            if self.rollout_temperature() <= 0:
                raise ValueError("root_success_v2 root rollouts need training.rollout_temperature > 0")
            if tr.rollouts_per_update != 1 or tr.states_per_rollout != 1:
                raise ValueError("root_success_v2 uses one question per rank round and one state per success")
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
