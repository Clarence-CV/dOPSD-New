"""Hybrid Path OPSD (sibling.target=path) on the CPU toy model: capped CE, shortest-correct path selection,
teacher-forced states, only LoRA moves, exact resume."""
import json
import math
import os
import tempfile
from unittest.mock import patch

import torch

from _helpers import build, mini_config, run_all
from selective_draft.data import load_split
from selective_draft.path_opsd import (PathOPSDTrainer, capped_ce, collect_paths, compact_path, compact_supervised,
                                       expand_path, load_paths, path_steps)
from selective_draft.sibling_opsd import sibling_rollout
from selective_draft.training import run_training
from test_sibling import _freeze_vocab, _text_grade


def _eos_rollout(*a, **k):
    """The toy model almost never emits EOS, and a path is only accepted when the rollout ended with EOS. These
    training-mechanics tests flag every rollout as EOS-terminated; the acceptance rule has its own test below."""
    roll = sibling_rollout(*a, **k)
    roll["stop_reason"] = "eos"
    return roll


def _cfg(**over):
    base = {"sibling.lora_rank": 4, "sibling.lora_alpha": 8.0, "sibling.learning_rate": 0.05, "sibling.target": "path",
            "sibling.states_per_rollout": 3, "sibling.path_cap": 0.95, "sibling.path_kl_every": 2,
            "generation.commit_threshold": 0.3, "training.max_rollout_attempts": 3, "training.rollout_temperature": 0.9,
            "training.weight_decay": 0.0, "training.max_steps": 50}
    return mini_config(**dict(base, **over))


def test_capped_ce_stops_at_the_cap():
    V = 10
    for pc in (0.2, 0.9, 0.96, 0.99):  # not exactly 0.95: log(0.95) rounding differs across CPUs (aarch64)
        logits = torch.full((1, V), math.log((1 - pc) / (V - 1)))
        logits[0, 3] = math.log(pc)
        logits.requires_grad_(True)
        l = capped_ce(logits, torch.tensor([3]), 0.95)
        want = max(0.0, -math.log(pc) + math.log(0.95))
        assert abs(float(l) - want) < 1e-5, (pc, float(l), want)
        l.sum().backward()
        if pc >= 0.95:
            assert float(logits.grad.abs().sum()) == 0.0, "no gradient once p(c) >= cap"
        else:
            assert float(logits.grad[0, 3]) < 0, "below the cap the gradient raises p(c)"


def test_path_steps_cover_every_commit_before_eos():
    cfg = _cfg()
    gm, tok, _ = build(cfg)
    examples = load_split(cfg, tok)
    _freeze_vocab(tok, examples)
    with patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        roll = sibling_rollout(gm, tok, examples[0], cfg, torch.Generator().manual_seed(0), 0.9)
    P, eos, eos_id = roll["meta"]["prompt_len"], roll["eos_position"], roll["meta"]["eos_id"]
    steps = path_steps(roll)
    assert steps, "toy rollout must have supervised steps"
    n_ok = 0
    for step in roll["steps"]:
        for j in range(step["committed_positions"].numel()):
            t, p = int(step["committed_tokens"][j]), int(step["committed_positions"][j])
            n_ok += t != eos_id and (eos is None or p - P < eos)
    assert sum(len(ok) for _, ok in steps) == n_ok, "every non-EOS commit before EOS is supervised once"


def test_trainer_picks_shortest_correct_and_moves_only_lora_and_resumes():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade), \
            patch("selective_draft.path_opsd.sibling_rollout", side_effect=_eos_rollout):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        ro0 = {k: v.clone() for k, v in gm.module_state_dict().items() if not k.startswith("lora.")}
        tr = PathOPSDTrainer(gm, cfg, os.path.join(d, "a"), examples, tok)
        run_training(tr, 50, 0)
        assert tr.finished and tr.state["states_used"] > 0
        assert any(float(p.abs().sum()) > 0 for p in gm.lora.B.values()), "LoRA B must move"
        for k, v in gm.module_state_dict().items():
            if not k.startswith("lora."):
                assert torch.equal(v, ro0[k]), f"{k} changed"
        logs = [json.loads(l) for l in open(os.path.join(d, "a", "log_rank0.jsonl"))]
        used = [r for r in logs if r.get("loss") is not None]
        assert used and all(r["loss"] >= 0 and 0 <= r["reached_cap_frac"] <= 1 for r in used)
        assert any("kl_p0_block" in r for r in used), "the drift monitor is logged on some updates"
        ledger = [json.loads(l) for l in open(os.path.join(d, "a", "question_attempts_rank0.jsonl"))]
        for r in ledger:
            assert len(r["attempts"]) == cfg.training.max_rollout_attempts, "shortest runs every attempt"
            ok = [a for a in r["attempts"] if a["accepted"]]
            if ok:
                assert r["path_steps"] == min(a["steps"] for a in ok)
                assert r["path_attempt"] == min(a["attempt"] for a in ok if a["steps"] == r["path_steps"])
        final_a = {k: v.clone() for k, v in gm.module_state_dict().items()}

        gm2, _, _ = build(cfg)
        tr2 = PathOPSDTrainer(gm2, cfg, os.path.join(d, "b"), examples, tok)
        tr2.update()
        tr2.update()
        ckpt = tr2.save()
        gm3, _, _ = build(cfg)
        tr3 = PathOPSDTrainer(gm3, cfg, os.path.join(d, "b"), examples, tok)
        run_training(tr3, 50, 0, resume=ckpt)
        for k, v in gm3.module_state_dict().items():
            assert torch.allclose(v, final_a[k], atol=1e-6), f"resume diverged at {k}"


def test_first_select_stops_at_first_success():
    cfg = _cfg(**{"sibling.path_select": "first"})
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade), \
            patch("selective_draft.path_opsd.sibling_rollout", side_effect=_eos_rollout):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        tr = PathOPSDTrainer(gm, cfg, d, examples, tok)
        run_training(tr, 4, 0)
        for r in map(json.loads, open(os.path.join(d, "question_attempts_rank0.jsonl"))):
            oks = [a["accepted"] for a in r["attempts"]]
            assert not any(oks[:-1])


def test_truncated_rollouts_are_never_accepted():
    """A correct answer inside a rollout that hit max_new_tokens (no EOS) must not become a teacher path."""
    cfg = _cfg(**{"sibling.path_select": "first"})

    def truncated(*a, **k):
        roll = sibling_rollout(*a, **k)
        roll["stop_reason"], roll["is_correct"] = "length", True
        return roll

    with tempfile.TemporaryDirectory() as d, patch("selective_draft.path_opsd.sibling_rollout", side_effect=truncated):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        tr = PathOPSDTrainer(gm, cfg, d, examples, tok)
        items, ledger = tr.root_question(examples[0], 0)
        assert items == [] and ledger["outcome"] == "root_budget_exhausted"
        assert all(a["is_correct"] and not a["accepted"] for a in ledger["attempts"])
        assert len(ledger["attempts"]) == cfg.training.max_rollout_attempts


def test_frozen_teacher_paths_ignore_lora_and_the_training_step():
    """path_teacher=frozen: the same rollouts whatever the LoRA weights or optimizer step; current: LoRA matters."""
    ex_i = 0

    def attempts(cfg, perturb, step):
        with patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
            gm, tok, _ = build(cfg)
            examples = load_split(cfg, tok)
            _freeze_vocab(tok, examples)
            if perturb:
                with torch.no_grad():
                    for p in gm.lora.B.values():
                        p.normal_(0, 0.5)
            with tempfile.TemporaryDirectory() as d:
                tr = PathOPSDTrainer(gm, cfg, d, examples, tok)
                tr.state["optimizer_step"] = step
                _, ledger = tr.root_question(examples[ex_i], ex_i)
        return [(a["steps"], a["output_tokens"], a["is_correct"], a["stop_reason"]) for a in ledger["attempts"]]

    frozen = _cfg()
    assert frozen.sibling.path_teacher == "frozen", "frozen is the default"
    ref = attempts(frozen, False, 0)
    assert attempts(frozen, True, 7) == ref
    current = _cfg(**{"sibling.path_teacher": "current"})
    assert attempts(current, False, 0) == ref, "with zero LoRA and step 0 both teachers coincide"
    assert attempts(current, True, 0) != ref, "the current-model teacher follows the LoRA"


def test_compact_path_rebuilds_every_recorded_state():
    """compact_path drops per-step states; expand_path must rebuild them (and the block's MASK set) exactly."""
    cfg = _cfg()
    gm, tok, _ = build(cfg)
    examples = load_split(cfg, tok)
    _freeze_vocab(tok, examples)
    with patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        roll = sibling_rollout(gm, tok, examples[1], cfg, torch.Generator().manual_seed(3), 0.9)
    rec = compact_path(roll)
    assert [k for k, _ in compact_supervised(rec)] == [roll["steps"].index(s) for s, _ in path_steps(roll)]
    assert [ok for _, ok in compact_supervised(rec)] == [ok for _, ok in path_steps(roll)]
    for k, orig in enumerate(roll["steps"]):
        r2, step = expand_path(rec, k)
        assert torch.equal(step["state"], orig["state"]), f"state differs at step {k}"
        assert torch.equal(step["eligible_positions"], orig["eligible_positions"]), f"eligible differs at step {k}"
        assert torch.equal(step["committed_tokens"], orig["committed_tokens"])
        assert r2["meta"]["prompt_ids"] == roll["meta"]["prompt_ids"] and r2["eos_position"] == roll["eos_position"]
    assert rec["final_state"].numel() == roll["final_state"].numel()
    # the stored record is small: no per-step state / confidence tensors
    assert all(set(s) == {"step_id", "block_start", "block_end", "committed_positions", "committed_tokens"}
               for s in rec["steps"])


def test_offline_paths_match_online_sampling_and_train():
    """collect-paths (two shards) stores exactly the paths the online frozen teacher samples; the offline trainer
    computes the same loss on the same states and finishes when the stored questions are used up."""
    cfg = _cfg(**{"sibling.path_select": "first", "sibling.states_per_rollout": 2})
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade), \
            patch("selective_draft.path_opsd.sibling_rollout", side_effect=_eos_rollout):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        pdir = os.path.join(d, "paths")
        s0 = collect_paths(gm, tok, cfg, examples, pdir, (0, 2))
        s1 = collect_paths(gm, tok, cfg, examples, pdir, (1, 2))
        assert s0["questions"] + s1["questions"] == len(examples)
        recs, ledgers = load_paths(pdir)
        assert len(recs) == s0["accepted"] + s1["accepted"] >= 1 and len(ledgers) == len(examples)
        # resume: a second call finds everything done and samples nothing
        with patch("selective_draft.path_opsd.sample_path", side_effect=AssertionError("resampled")):
            assert collect_paths(gm, tok, cfg, examples, pdir, (0, 2))["questions"] == s0["questions"]

        online = PathOPSDTrainer(gm, cfg, os.path.join(d, "on"), examples, tok)
        off_cfg = _cfg(**{"sibling.path_select": "first", "sibling.states_per_rollout": 2,
                          "sibling.path_source": json.dumps(pdir)})
        offline = PathOPSDTrainer(gm, off_cfg, os.path.join(d, "off"), examples, tok)
        assert set(examples[i].example_id for i in offline.order) == set(recs)
        for i in offline.order:
            on_items, _ = online.root_question(examples[i], i)
            off_items, _ = offline.root_question(examples[i], i)
            assert len(on_items) == len(off_items) >= 1
            for (r1, s1_, ok1), (r2, s2_, ok2) in zip(on_items, off_items):
                assert ok1 == ok2 and torch.equal(s1_["state"], s2_["state"])
                l1, _ = online.state_loss(r1, s1_, ok1)
                l2, _ = offline.state_loss(r2, s2_, ok2)
                assert torch.allclose(l1, l2)
        with patch("selective_draft.path_opsd.sample_path", side_effect=AssertionError("offline must not roll out")):
            run_training(offline, 50, 0)
        assert offline.finished and offline.state["states_used"] > 0
        # every stored question visited once by run_training (+ once by the manual root_question loop above)
        assert offline.state["question_visits"] == 2 * len(offline.order)


if __name__ == "__main__":
    run_all(globals())
