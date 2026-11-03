"""Sibling OPSD on the CPU toy model: LoRA hooks (zero init, Teacher path off), the alpha = 2*tau - 1 target rule,
leave-one-out sibling inputs, the trainer (only LoRA moves, exact resume), snapshots, and the Phase 0 diagnostics."""
import hashlib
import json
import os
import tempfile
from unittest.mock import patch

import torch

from _helpers import MASK, build, mini_config, run_all
from selective_draft.checkpoint import load_snapshot, save_snapshot
from selective_draft.data import load_split
from selective_draft.sibling_opsd import (SiblingOPSDTrainer, loo_teacher, mixed_target, parallel_steps,
                                          run_sibling_diag, sibling_rollout, summarize_sibling_diag)
from selective_draft.training import run_training


def _cfg(**over):
    base = {"sibling.lora_rank": 4, "sibling.lora_alpha": 8.0, "sibling.learning_rate": 0.05,
            "sibling.states_per_rollout": 3, "sibling.max_positions_per_state": 3,
            "generation.commit_threshold": 0.3, "training.max_rollout_attempts": 3, "training.rollout_temperature": 0.9,
            "training.weight_decay": 0.0, "training.max_steps": 50}
    return mini_config(**dict(base, **over))


def _text_grade(gold, text, task):
    return {"parsed_answer": None, "is_correct": int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 0}


def _freeze_vocab(tok, examples):
    for e in examples:  # the toy tokenizer builds its id->word map lazily
        tok.encode(e.prompt_text)


def test_lora_starts_as_identity_and_the_teacher_path_ignores_it():
    gm, tok, cfg = build(_cfg())
    x = torch.randint(4, 60, (1, 16))
    ref = gm.base_forward(x).clone()
    assert torch.equal(gm.teacher_forward(x), ref)
    with torch.no_grad():
        for p in gm.lora.B.values():
            p.normal_(0, 0.5)
    assert not torch.allclose(gm.base_forward(x), ref), "a non-zero adapter must change the student"
    assert torch.equal(gm.teacher_forward(x), ref), "the Teacher is the frozen base"
    gm.assert_base_frozen()
    params = gm.set_stage("sibling")
    assert params and all(n.startswith("lora.") for n in gm.trainable_parameter_names())
    assert len(gm.lora.names) == 2 * gm.backend.n_layers  # query_key_value + dense per layer


def test_alpha_rule_makes_parallel_commit_a_fixed_point_and_never_redirects():
    cfg = _cfg(**{"generation.commit_threshold": 0.7})
    a = cfg.sibling_alpha()
    assert abs(a - 0.4) < 1e-9
    V = 10
    for qc, expect_commit in ((0.9, True), (0.55, True), (0.45, False), (0.05, False)):
        q = torch.full((1, V), (1 - qc) / (V - 1))
        q[0, 0] = qc
        t = mixed_target(q.log(), torch.tensor([0]), a)
        assert abs(float(t.sum()) - 1) < 1e-6
        assert (float(t[0, 0]) >= 0.7) == expect_commit, (qc, float(t[0, 0]))
        assert float(t[0, 1:].max()) < 0.7  # another token can never pass tau from the Teacher alone
    assert abs(_cfg(**{"sibling.alpha_mix": 0.25}).sibling_alpha() - 0.25) < 1e-12


def test_loo_teacher_reveals_every_sibling_but_the_target():
    gm, tok, cfg = build(_cfg())
    seen = []
    orig = gm.teacher_forward

    def spy(ids):
        seen.append(ids.clone())
        return orig(ids)
    gm.teacher_forward = spy
    x = torch.randint(4, 60, (1, 16))
    cpos = torch.tensor([9, 11, 12])
    x[0, 8:16] = MASK
    ctok = torch.tensor([20, 21, 22])
    log_q = loo_teacher(gm, x, cpos, ctok, torch.tensor([0, 2]), batch_size=1)
    assert log_q.shape == (2, gm.backend.vocab_size)
    assert len(seen) == 2
    for row, target in zip(seen, (0, 2)):
        r = row[0]
        assert int(r[cpos[target]]) == MASK
        for j in range(3):
            if j != target:
                assert int(r[cpos[j]]) == int(ctok[j])
        others = [p for p in range(8, 16) if p not in cpos.tolist()]
        assert bool((r[others] == MASK).all())


def test_trainer_moves_only_lora_and_resumes_exactly():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        ro0 = {k: v.clone() for k, v in gm.module_state_dict().items() if not k.startswith("lora.")}
        tr = SiblingOPSDTrainer(gm, cfg, os.path.join(d, "a"), examples, tok)
        run_training(tr, 50, 0)
        assert tr.finished
        assert tr.state["states_used"] > 0, "the toy setting must produce parallel-commit states"
        assert any(float(p.abs().sum()) > 0 for p in gm.lora.B.values()), "LoRA B must move"
        for k, v in gm.module_state_dict().items():
            if not k.startswith("lora."):
                assert torch.equal(v, ro0[k]), f"{k} changed"
        logs = [json.loads(l) for l in open(os.path.join(d, "a", "log_rank0.jsonl"))]
        used = [r for r in logs if r.get("loss") is not None]
        assert used and all(0.0 <= r["fixed_point_frac"] <= 1.0 for r in used)
        ledger = [json.loads(l) for l in open(os.path.join(d, "a", "question_attempts_rank0.jsonl"))]
        assert sorted(r["example_id"] for r in ledger) == sorted(e.example_id for e in examples)
        for r in ledger:
            oks = [x["is_correct"] for x in r["attempts"]]
            assert not any(oks[:-1]), "retries stop at the first success"
        final_a = {k: v.clone() for k, v in gm.module_state_dict().items()}

        gm2, _, _ = build(cfg)
        tr2 = SiblingOPSDTrainer(gm2, cfg, os.path.join(d, "b"), examples, tok)
        tr2.update()
        tr2.update()
        ckpt = tr2.save()
        gm3, _, _ = build(cfg)
        tr3 = SiblingOPSDTrainer(gm3, cfg, os.path.join(d, "b"), examples, tok)
        run_training(tr3, 50, 0, resume=ckpt)
        for k, v in gm3.module_state_dict().items():
            assert torch.allclose(v, final_a[k], atol=1e-6), f"resume diverged at {k}"


def test_questions_per_update_batches_questions():
    cfg = _cfg(**{"sibling.questions_per_update": 2})
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        tr = SiblingOPSDTrainer(gm, cfg, d, examples, tok)
        tr.update()
        assert tr.state["question_visits"] == 2


def test_snapshot_round_trip_keeps_the_adapters():
    cfg = _cfg()
    gm, tok, _ = build(cfg)
    with torch.no_grad():
        for p in gm.lora.B.values():
            p.normal_(0, 0.3)
    x = torch.randint(4, 60, (1, 16))
    want = gm.base_forward(x)
    with tempfile.TemporaryDirectory() as d:
        save_snapshot(os.path.join(d, "s"), gm, cfg, "sibling")
        gm2, _, _ = build(cfg)
        load_snapshot(os.path.join(d, "s"), gm2, cfg)
        assert torch.allclose(gm2.base_forward(x), want, atol=1e-6)
        cfg0 = mini_config(**{"generation.commit_threshold": 0.3})
        gm0, _, _ = build(cfg0)
        try:
            load_snapshot(os.path.join(d, "s"), gm0, cfg0)
            raise AssertionError("a LoRA snapshot must not load into a model without adapters")
        except ValueError:
            pass


def test_parallel_steps_skip_eos_and_single_commits():
    cfg = _cfg()
    gm, tok, _ = build(cfg)
    examples = load_split(cfg, tok)
    _freeze_vocab(tok, examples)
    with patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        roll = sibling_rollout(gm, tok, examples[0], cfg, torch.Generator().manual_seed(0), 0.9)
    P, eos = roll["meta"]["prompt_len"], roll["eos_position"]
    for step, ok in parallel_steps(roll):
        assert len(ok) >= 2 and step["committed_positions"].numel() >= 2
        for j in ok:
            assert int(step["committed_tokens"][j]) != roll["meta"]["eos_id"]
            assert eos is None or int(step["committed_positions"][j]) - P < eos


def test_diag_rows_and_summary():
    cfg = _cfg(**{"training.max_rollout_attempts": 2})
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)[:3]
        _freeze_vocab(tok, examples)
        run_sibling_diag(gm, tok, examples, cfg, d, shard=(1, 2))
        run_sibling_diag(gm, tok, examples, cfg, d, shard=(0, 2))
        assert len(os.listdir(os.path.join(d, "rows"))) == 3
        s = summarize_sibling_diag(d)
        assert s["n_questions"] == 3 and 0.0 <= s["solved_within_budget"] <= 1.0
        for key in ("all", "correct", "wrong"):
            assert "fixed_point_violation_rate" in s[key]
        for r in map(lambda p: json.load(open(os.path.join(d, "rows", p))), os.listdir(os.path.join(d, "rows"))):
            oks = [a["is_correct"] for a in r["attempts"]]
            assert not any(oks[:-1])
            for a in r["attempts"]:
                for p in a["positions"]:
                    assert 0.0 <= p["q_c"] <= 1.0 and p["group_size"] >= 2


if __name__ == "__main__":
    run_all(globals())
