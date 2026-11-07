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
from selective_draft.path_opsd import PathOPSDTrainer, capped_ce, path_steps
from selective_draft.sibling_opsd import sibling_rollout
from selective_draft.training import run_training
from test_sibling import _freeze_vocab, _text_grade


def _cfg(**over):
    base = {"sibling.lora_rank": 4, "sibling.lora_alpha": 8.0, "sibling.learning_rate": 0.05, "sibling.target": "path",
            "sibling.states_per_rollout": 3, "sibling.path_cap": 0.95, "sibling.path_kl_every": 2,
            "generation.commit_threshold": 0.3, "training.max_rollout_attempts": 3, "training.rollout_temperature": 0.9,
            "training.weight_decay": 0.0, "training.max_steps": 50}
    return mini_config(**dict(base, **over))


def test_capped_ce_stops_at_the_cap():
    V = 10
    for pc in (0.2, 0.9, 0.95, 0.99):
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
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
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
            ok = [a for a in r["attempts"] if a["is_correct"]]
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
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        tr = PathOPSDTrainer(gm, cfg, d, examples, tok)
        run_training(tr, 4, 0)
        for r in map(json.loads, open(os.path.join(d, "question_attempts_rank0.jsonl"))):
            oks = [a["is_correct"] for a in r["attempts"]]
            assert not any(oks[:-1])


if __name__ == "__main__":
    run_all(globals())
