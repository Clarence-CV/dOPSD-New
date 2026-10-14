"""End-to-end on the fake model: collect -> stage 1 -> stage 2 -> resume equivalence -> evaluate -> diagnostics."""
import copy
import json
import os
import tempfile

import torch

from _fake_helpers import MASK, build, fake_config, run_all
from selective_draft.checkpoint import load_snapshot, save_snapshot
from selective_draft.data import load_split, tokenize_prompts, parse_prediction, is_correct
from selective_draft.diagnostics import consistency_report, readout_report, teacher_report
from selective_draft.evaluation import evaluate
from selective_draft.generation import generate
from selective_draft.trace import StateRecorder, load_rollouts, replay_check, save_rollouts
from selective_draft.training import GuidanceTrainer, ReadoutTrainer, run_training, collect_state_refs


def _collect(gm, tok, cfg, path, guidance_mode="off", limit=4):
    examples = load_split(cfg, tok)[:limit]
    rolls = []
    for e in examples:
        ids, mask = tokenize_prompts(tok, [e.prompt_text], torch.device("cpu"))
        meta = {"example_id": e.example_id, "prompt_ids": ids[0].tolist(), "prompt_attention_mask": mask[0].tolist(), "gold": e.answer_value}
        rec = StateRecorder("state_only", MASK, gm.backend.eos_token_id, ids.shape[1], [meta])
        res = generate(gm, ids, mask, cfg.generation, guidance_mode=guidance_mode, recorder=rec)
        text = tok.decode(res.completion_ids(0))
        pred = parse_prediction(text)
        rolls.extend(rec.finalize(res, [{"generation": text, "parsed_answer": pred, "is_correct": is_correct(pred, e.answer_value)}]))
    for r in rolls:
        replay_check(r)
    save_rollouts(path, rolls)
    return rolls


def test_parse_and_manifest_helpers():
    assert parse_prediction("so <answer>\\boxed{1,234}</answer>") == 1234.0
    assert parse_prediction("<answer> the result is 7 dollars </answer>") == 7.0
    assert parse_prediction("nothing") is None
    assert is_correct(5.0, 5.0) and not is_correct(None, 5.0)


def test_stage1_then_stage2_and_resume_equivalence():
    with tempfile.TemporaryDirectory() as d:
        gm, tok, cfg = build()
        rolls = _collect(gm, tok, cfg, os.path.join(d, "base.pt"))
        assert collect_state_refs(rolls, filter_correct_only=False, require_gate=False)
        # ---- stage 1
        ref_logits = gm(*_first_state(rolls), mode="off").logits.clone()
        t1 = ReadoutTrainer(gm, cfg, os.path.join(d, "s1"), [os.path.join(d, "base.pt")])
        assert t1.gm.trainable_parameter_names() == ["readout.down.weight", "readout.up.weight"]
        before = copy.deepcopy(gm.injector.state_dict())
        last = run_training(t1, max_steps=3, save_every=0)
        assert last["loss"] == last["loss"] and last["grad_norm"] > 0
        assert float(gm.readout.up.weight.abs().sum()) > 0
        assert all(torch.equal(before[k], v) for k, v in gm.injector.state_dict().items())
        # the final output path is untouched by stage 1 (parity with the base when the message is off)
        assert torch.equal(gm(*_first_state(rolls), mode="off").logits, ref_logits)
        snap = os.path.join(d, "s1", "snapshot")
        assert os.path.exists(os.path.join(snap, "modules.safetensors"))
        # ---- stage 2 (gate must open on the fake model: lower the threshold)
        cfg2 = fake_config(**{"guidance.message_threshold": 0.0, "training.max_consecutive_empty_updates": 3})
        gm2, tok2, _ = build(cfg2)
        load_snapshot(snap, gm2, cfg2)
        assert torch.equal(gm2.readout.up.weight, gm.readout.up.weight)
        examples = load_split(cfg2, tok2)
        t2 = GuidanceTrainer(gm2, cfg2, os.path.join(d, "s2a"), examples, tok2)
        assert t2.gm.trainable_parameter_names() == ["injector.down.weight", "injector.up.weight"]
        r_before = copy.deepcopy(gm2.readout.state_dict())
        run_training(t2, max_steps=2, save_every=2)
        assert float(gm2.injector.up.weight.abs().sum()) > 0
        assert all(torch.equal(r_before[k], v) for k, v in gm2.readout.state_dict().items())
        ck = os.path.join(d, "s2a", "checkpoint-step-2")
        assert os.path.exists(os.path.join(ck, "COMPLETE")) and os.path.exists(os.path.join(ck, "optimizer.pt"))
        # continue 2 more steps in the same process
        run_training(t2, max_steps=4, save_every=0)
        cont = {k: v.clone() for k, v in gm2.module_state_dict().items()}
        # fresh process: load snapshot of stage 1, resume from step-2, run to 4 -> identical modules
        gm3, tok3, _ = build(cfg2)
        t3 = GuidanceTrainer(gm3, cfg2, os.path.join(d, "s2b"), load_split(cfg2, tok3), tok3)
        run_training(t3, max_steps=4, save_every=0, resume=ck)
        for k, v in gm3.module_state_dict().items():
            assert torch.equal(v, cont[k]), f"resume mismatch in {k}"
        assert t3.state["optimizer_step"] == 4 and t3.state["rollout_round"] == t2.state["rollout_round"]
        log = [json.loads(l) for l in open(os.path.join(d, "s2a", "log_rank0.jsonl"))]
        assert all("teacher_forwards" in r and r["teacher_forwards"] == 1.0 for r in log if "loss" in r), "realized mode = 1 forward"


def _first_state(rolls):
    from selective_draft.training import state_tensors
    x, attn, ctx, _ = state_tensors(rolls[0], rolls[0]["steps"][0], torch.device("cpu"))
    return x[None], attn[None]


def test_empty_update_is_skipped_and_diagnosed():
    with tempfile.TemporaryDirectory() as d:
        cfg = fake_config(**{"guidance.message_threshold": 1.0, "training.max_consecutive_empty_updates": 2})
        gm, tok, _ = build(cfg)
        t = GuidanceTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        rec = t.update()
        assert rec.get("skipped") == 1 and t.state["optimizer_step"] == 0
        try:
            t.update()
            assert False, "second empty update must raise the diagnostic exit"
        except RuntimeError as e:
            assert "consecutive" in str(e)


def test_evaluate_and_diagnostics_run():
    with tempfile.TemporaryDirectory() as d:
        cfg = fake_config(**{"guidance.message_threshold": 0.3})
        gm, tok, _ = build(cfg)
        ex = load_split(cfg, tok)
        out = evaluate(gm, tok, ex, cfg, "off", os.path.join(d, "ev"), thresholds=[0.5, 0.9], trace_level="state_only")
        assert set(out["summaries"]) == {"thr0.5", "thr0.9"} and out["summaries"]["thr0.9"]["n"] == len(ex)
        assert out["summaries"]["thr0.5"]["mean_row_steps"] <= out["summaries"]["thr0.9"]["mean_row_steps"]
        assert os.path.exists(os.path.join(d, "ev", "rows_thr0.9.jsonl"))
        rolls = load_rollouts(os.path.join(d, "ev", "rollouts_thr0.9.pt"))
        r = readout_report(gm, rolls, cfg, max_states=6)
        assert r["states"] == 6 and r["kl_mean"] >= -1e-6 and 0 <= r["top1_agree"] <= 1
        t = teacher_report(gm, rolls, cfg, max_states=6)
        assert t["states_with_receivers"] >= 1
        if t["gated_states"]:
            for k in ("kl_realized_vs_q0", "kl_mc_vs_q0", "kl_argmax_vs_q0", "kl_topk_vs_q0", "oracle_recovery_ratio_of_sums"):
                assert k in t and t[k] == t[k]
        c = consistency_report(gm, rolls, cfg, max_steps=10)
        if c["steps"]:
            assert 0 <= c["psa"] <= 1 and 0 <= c["regret"] <= 1
        # snapshot strictness
        save_snapshot(os.path.join(d, "snap"), gm, cfg, "guidance")
        bad = fake_config(**{"guidance.readout_rank": 4})
        gm_bad, _, _ = build(bad)
        try:
            load_snapshot(os.path.join(d, "snap"), gm_bad, bad)
            assert False
        except ValueError:
            pass


if __name__ == "__main__":
    run_all(globals())
