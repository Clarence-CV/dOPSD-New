"""Verified self-drafter OPSD (sibling.target=drafter) on the CPU toy model: strict acceptance, greedy first attempt,
candidate / LOO verdict / throttle / brake / content terms, only LoRA moves, exact resume."""
import json
import os
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

from _helpers import build, mini_config, run_all
from selective_draft import drafter_opsd
from selective_draft.data import load_split
from selective_draft.drafter_opsd import DrafterOPSDTrainer, accepted
from selective_draft.path_opsd import path_steps
from selective_draft.sibling_opsd import sibling_rollout
from selective_draft.training import run_training
from test_sibling import _freeze_vocab, _text_grade


def _cfg(**over):
    base = {"sibling.lora_rank": 4, "sibling.lora_alpha": 8.0, "sibling.learning_rate": 0.05,
            "sibling.target": "drafter", "sibling.states_per_rollout": 3, "sibling.max_positions_per_state": 4,
            "sibling.path_cap": 0.95, "sibling.path_kl_every": 2, "generation.commit_threshold": 0.3,
            "training.max_rollout_attempts": 3, "training.rollout_temperature": 0.9, "training.weight_decay": 0.0,
            "training.max_steps": 50}
    return mini_config(**dict(base, **over))


def _ok_rollout(*a, **k):
    """Toy rollouts never emit EOS or a boxed answer: flag them so the acceptance rule only follows the grader."""
    roll = sibling_rollout(*a, **k)
    roll["stop_reason"] = "eos"
    roll["generation"] = roll["generation"] + " \\boxed{1}"
    return roll


def test_acceptance_needs_correct_eos_and_boxed():
    good = {"is_correct": True, "stop_reason": "eos", "generation": "so \\boxed{3}"}
    assert accepted(good)
    assert not accepted(dict(good, is_correct=False))
    assert not accepted(dict(good, stop_reason="length")), "a truncated rollout is never a training path"
    assert not accepted(dict(good, generation="the answer is 3"))


def test_first_attempt_is_greedy_and_later_ones_sample():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade",
                                                   return_value={"parsed_answer": None, "is_correct": False}):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        tr = DrafterOPSDTrainer(gm, cfg, d, examples, tok)
        items, ledger = tr.root_question(examples[0], 0)
    assert items == [] and ledger["outcome"] == "root_budget_exhausted", "no correct rollout -> no gradient"
    temps = [a["temperature"] for a in ledger["attempts"]]
    assert temps == [0.0, 0.9, 0.9]


def _one_state(cfg):
    gm, tok, _ = build(cfg)
    examples = load_split(cfg, tok)
    _freeze_vocab(tok, examples)
    with patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade):
        roll = _ok_rollout(gm, tok, examples[0], cfg, torch.Generator().manual_seed(0), 0.9)
    step, ok = path_steps(roll)[0]
    return gm, tok, examples, roll, step, ok


def _loss_with_teacher(cfg, verdict):
    """state_loss with every masked position a candidate and a stub LOO teacher that agrees (verdict='agree') or
    vetoes (verdict='veto') every proposal."""
    gm, tok, examples, roll, step, ok = _one_state(cfg)
    with tempfile.TemporaryDirectory() as d:
        tr = DrafterOPSDTrainer(gm, cfg, d, examples, tok)

        def teacher(gm_, x, cpos, ctok, sup, bs):
            V = gm_.backend.vocab_size
            tgt = ctok[sup] if verdict == "agree" else (ctok[sup] + 1) % V
            return F.log_softmax(F.one_hot(tgt, V).float() * 20.0, -1)
        with patch.object(drafter_opsd, "CANDIDATE_P", 0.0), patch.object(drafter_opsd, "loo_teacher", teacher):
            loss, st = tr.state_loss(roll, step, ok)
    return loss, st


def test_agreeing_teacher_gives_throttle_only():
    loss, st = _loss_with_teacher(_cfg(), "agree")
    assert st["n_cand"] == st["n_masked"] > 0 and st["agree_frac"] == 1.0
    assert st["brake_kl"] != st["brake_kl"], "no vetoed row -> no brake term"
    want = st["throttle"] + st["content_ce"]
    assert abs(float(loss) - want) < 1e-4, (float(loss), want)


def test_vetoing_teacher_gives_brake_and_content_weight_zero_drops_content():
    loss, st = _loss_with_teacher(_cfg(), "veto")
    assert st["agree_frac"] == 0.0 and st["brake_kl"] > 0
    assert abs(float(loss) - (st["brake_kl"] + st["content_ce"])) < 1e-4
    loss0, st0 = _loss_with_teacher(_cfg(**{"sibling.content_weight": 0.0}), "veto")
    assert abs(float(loss0) - st0["brake_kl"]) < 1e-4, "content_weight=0 removes the content term"


def test_no_candidate_still_backpropagates_content():
    cfg = _cfg()
    gm, tok, examples, roll, step, ok = _one_state(cfg)
    with tempfile.TemporaryDirectory() as d:
        tr = DrafterOPSDTrainer(gm, cfg, d, examples, tok)
        with patch.object(drafter_opsd, "CANDIDATE_P", 1.01):
            loss, st = tr.state_loss(roll, step, ok)
    assert st["n_cand"] == 0 and st["n_sup"] == 0
    loss.backward()
    assert abs(float(loss) - st["content_ce"]) < 1e-5


def test_trainer_moves_only_lora_and_resumes_exactly():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.sibling_opsd.grade", side_effect=_text_grade), \
            patch("selective_draft.drafter_opsd.sibling_rollout", side_effect=_ok_rollout), \
            patch.object(drafter_opsd, "CANDIDATE_P", 0.05):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        _freeze_vocab(tok, examples)
        ro0 = {k: v.clone() for k, v in gm.module_state_dict().items() if not k.startswith("lora.")}
        tr = DrafterOPSDTrainer(gm, cfg, os.path.join(d, "a"), examples, tok)
        run_training(tr, 50, 0)
        assert tr.finished and tr.state["states_used"] > 0
        assert any(float(p.abs().sum()) > 0 for p in gm.lora.B.values()), "LoRA B must move"
        for k, v in gm.module_state_dict().items():
            if not k.startswith("lora."):
                assert torch.equal(v, ro0[k]), f"{k} changed"
        logs = [json.loads(l) for l in open(os.path.join(d, "a", "log_rank0.jsonl"))]
        used = [r for r in logs if r.get("loss") is not None]
        assert used and any("agree_frac" in r for r in used) and any("kl_p0_block" in r for r in used)
        assert all("greedy_accept" in r for r in used)
        final_a = {k: v.clone() for k, v in gm.module_state_dict().items()}

        gm2, _, _ = build(cfg)
        tr2 = DrafterOPSDTrainer(gm2, cfg, os.path.join(d, "b"), examples, tok)
        tr2.update()
        tr2.update()
        ckpt = tr2.save()
        gm3, _, _ = build(cfg)
        tr3 = DrafterOPSDTrainer(gm3, cfg, os.path.join(d, "b"), examples, tok)
        run_training(tr3, 50, 0, resume=ckpt)
        for k, v in gm3.module_state_dict().items():
            assert torch.allclose(v, final_a[k], atol=1e-6), f"resume diverged at {k}"


if __name__ == "__main__":
    run_all(globals())
