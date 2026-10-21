"""Stage-1 readout training, root_success_v2 Joint training, leave-one-out certification, resumable sharded evaluation,
checkpoint resume. Everything runs on the CPU toy model."""
import hashlib
import json
import os
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

from _helpers import EOS, MASK, V, build, conf_logits, controlled, mini_config, prompt, run_all
from selective_draft.data import load_split
from selective_draft.evaluation import evaluate, merge_shards, summarize_rows
from selective_draft.generation import generate
from selective_draft.root_joint import RootSuccessTrainer, certify_tuple, reference_tuple
from selective_draft.trace import StateRecorder, replay_check, save_rollouts
from selective_draft.training import ReadoutTrainer, run_training


def _record(gm, cfg, ids, example_id):
    rec = StateRecorder(MASK, EOS, ids.shape[1], {"example_id": example_id, "prompt_ids": ids[0].tolist()})
    res = generate(gm, ids, cfg.generation, guidance_mode="off", recorder=rec)
    roll = rec.finalize(res, {"is_correct": True, "stats": res.stats})
    replay_check(roll)
    return roll


# ----------------------------------------------------------------------------- stage 1
def test_readout_training_updates_only_the_readout_and_resumes_exactly():
    cfg = mini_config(**{"generation.max_new_tokens": 24, "training.max_steps": 6})
    gm, _, _ = build(cfg)
    rolls = [_record(gm, cfg, prompt(P, P), f"p{P}") for P in (9, 12, 13, 17)]
    with tempfile.TemporaryDirectory() as d:
        save_rollouts(os.path.join(d, "r.pt"), rolls)
        paths = [os.path.join(d, "r.pt")]
        gm_a, _, _ = build(cfg)
        tr = ReadoutTrainer(gm_a, cfg, os.path.join(d, "a"), paths)
        frozen = {n: p.clone() for n, p in gm_a.named_parameters() if not p.requires_grad}
        run_training(tr, 6, 0)
        assert gm_a.readout.up.weight.abs().sum() > 0 and gm_a.injector.up.weight.abs().sum() == 0
        assert all(torch.equal(dict(gm_a.named_parameters())[n], p) for n, p in frozen.items())
        # 3 updates, checkpoint, new process state, 3 more == 6 straight
        gm_b, _, _ = build(cfg)
        tr_b = ReadoutTrainer(gm_b, cfg, os.path.join(d, "b"), paths)
        for _ in range(3):
            tr_b.update()
        ckpt = tr_b.save()
        gm_c, _, _ = build(cfg)
        tr_c = ReadoutTrainer(gm_c, cfg, os.path.join(d, "b"), paths)
        run_training(tr_c, 6, 0, resume=ckpt)
        for k, v in gm_c.module_state_dict().items():
            assert torch.equal(v, gm_a.module_state_dict()[k]), k
        assert os.path.exists(os.path.join(d, "a", "snapshot", "modules.safetensors"))


# ----------------------------------------------------------------------------- leave-one-out certification
def test_leave_one_out_reveals_only_peers_and_batches_independent_inputs():
    cfg = mini_config(**{"generation.max_new_tokens": 8})
    positions = torch.tensor([9, 11, 12])
    tokens = torch.tensor([20, 21, 22])
    x = torch.full((1, 16), MASK)
    x[0, :8] = 30

    def fn(ids):  # each Teacher row predicts the reference token at every position with confidence .9
        plan = {int(p): (int(t), .9) for p, t in zip(positions, tokens)}
        return conf_logits(plan)(ids)
    for batch_size in (1, 3):
        gm, _, _, calls = controlled(fn, cfg)
        log_q, stats = certify_tuple(gm, x, positions, tokens, batch_size)
        assert log_q is not None and log_q.shape == (3, V)
        rows = torch.cat(calls)
        assert rows.shape[0] == 3 and len(calls) == (3 if batch_size == 1 else 1)
        for i in range(3):
            assert int(rows[i, positions[i]]) == MASK  # the target stays masked
            for j in range(3):
                if j != i:
                    assert int(rows[i, positions[j]]) == int(tokens[j])  # peers are revealed
            assert bool((rows[i, :8] == 30).all()) and int(rows[i, 10]) == MASK  # nothing else changes
        assert torch.equal(log_q.exp().argmax(-1), tokens)
    # one coordinate disagrees -> the whole tuple is rejected
    bad = tokens.clone()
    bad[2] = 25
    gm, _, _, _ = controlled(fn, cfg)
    assert certify_tuple(gm, x, positions, bad, 1)[0] is None


def test_reference_tuple_rejects_incompatible_states():
    roll = {"meta": {"prompt_len": 8}, "final_state": torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]),
            "eos_position": None}
    state = torch.tensor([20, MASK, MASK, MASK, 24, 25, 26, 27])
    step = {"state": state, "block_start": 8, "block_end": 16}
    pos = torch.tensor([9, 10])
    pi, why = reference_tuple(roll, step, pos, MASK)
    assert why == "ok" and pi.tolist() == [21, 22]
    assert reference_tuple(dict(roll, eos_position=2), step, pos, MASK)[1] == "at_or_after_eos"
    assert reference_tuple(roll, dict(step, state=torch.tensor([99, MASK, MASK, MASK, 24, 25, 26, 27])), pos, MASK)[1] \
        == "known_token_changed"
    assert reference_tuple(roll, step, torch.tensor([8, 9]), MASK)[1] == "group_not_masked"
    assert reference_tuple(roll, dict(step, block_end=10), pos, MASK)[1] == "outside_current_block"


# ----------------------------------------------------------------------------- root_success_v2
def _root_cfg(**over):
    base = {"guidance.mode": '"group_soft"', "guidance.early_layer": 3, "generation.commit_threshold": 0.3,
            "training.max_steps": 50, "training.max_rollout_attempts": 3}
    return mini_config(**dict(base, **over))


def _text_grade(gold, text, task):
    """Deterministic in the generated text only, so a resumed run sees the same verdicts."""
    ok = int(hashlib.md5(text.encode()).hexdigest(), 16) % 3 == 0
    return {"parsed_answer": None, "is_correct": ok}


def test_root_success_first_success_one_state_no_redraw_and_exact_resume():
    cfg = _root_cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.root_joint.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        for e in examples:  # freeze the toy tokenizer's lazily built id->word map before any decoding
            tok.encode(e.prompt_text)
        tr = RootSuccessTrainer(gm, cfg, os.path.join(d, "a"), examples, tok)
        run_training(tr, 50, 0)
        assert tr.finished and int(tr.state["question_round"]) == len(examples)
        ledger = [json.loads(l) for l in open(os.path.join(d, "a", "question_attempts_rank0.jsonl"))]
        assert sorted(r["example_id"] for r in ledger) == sorted(e.example_id for e in examples)
        for r in ledger:
            oks = [a["is_correct"] for a in r["attempts"]]
            assert len(oks) <= 3 and not any(oks[:-1]), "root retries must stop at the first success"
            if r["outcome"] == "root_budget_exhausted":
                assert len(oks) == 3 and not any(oks)
            else:
                assert oks[-1] and r["first_success_attempt"] == len(oks) - 1
        assert tr.state["states_drawn"] <= tr.state["questions_with_correct_rollout"]  # at most one state each
        drawn = [r for r in ledger if "state_step_id" in r]
        assert tr.state["accepted_states"] + tr.state["loo_rejected_states"] + tr.state["reselect_mismatch_states"] \
            + tr.state["invalid_reference_states"] == len(drawn)
        for path in os.listdir(os.path.join(d, "a", "references")):
            rec = torch.load(os.path.join(d, "a", "references", path), weights_only=False)
            P = rec["meta"]["prompt_len"]
            assert len(rec["group_positions"]) >= 2
            assert torch.equal(rec["group_positions"], rec["state"]["group_positions"].long())
            assert bool((rec["state"]["state"][rec["group_positions"] - P] == MASK).all())
            assert not bool((rec["reference_tokens"] == MASK).any())
        final_a = {k: v.clone() for k, v in gm.module_state_dict().items()}

        # exact resume: 2 rounds, checkpoint, fresh model + trainer, the rest (the toy tokenizer is shared because
        # decoding is grading and it learns id->word lazily)
        gm2, _, _ = build(cfg)
        tr2 = RootSuccessTrainer(gm2, cfg, os.path.join(d, "b"), examples, tok)
        tr2.update()
        tr2.update()
        ckpt = tr2.save()
        gm3, _, _ = build(cfg)
        tr3 = RootSuccessTrainer(gm3, cfg, os.path.join(d, "b"), examples, tok)
        tr3.resume(ckpt)
        run_training(tr3, 50, 0)
        ledger_b = [json.loads(l) for l in open(os.path.join(d, "b", "question_attempts_rank0.jsonl"))]
        strip = lambda rows: [(r["example_id"], r["outcome"], r.get("state_step_id"),
                               [(a["is_correct"], a["nfe"], a["output_tokens"]) for a in r.get("attempts", [])])
                              for r in rows]
        assert strip(ledger) == strip(ledger_b), [(a, b) for a, b in zip(strip(ledger), strip(ledger_b)) if a != b]
        for k, v in gm3.module_state_dict().items():
            assert torch.allclose(v, final_a[k], atol=1e-6), k


def test_root_success_exhausted_budget_never_trains():
    cfg = _root_cfg(**{"training.max_rollout_attempts": 2})
    never = lambda gold, text, task: {"parsed_answer": None, "is_correct": False}
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.root_joint.grade", side_effect=never):
        gm, tok, _ = build(cfg)
        tr = RootSuccessTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        rec = tr.update()
        assert rec.get("skipped") == 1 and rec["outcome"] == "root_budget_exhausted"
        assert tr.state["root_attempts"] == 2 and tr.state["states_drawn"] == 0
        assert float(gm.injector.up.weight.abs().sum()) == 0.0


# ----------------------------------------------------------------------------- evaluation
def test_sharded_resumable_evaluation_merges_to_the_unsharded_rows():
    gm, tok, cfg = build(mini_config(**{"generation.max_new_tokens": 12}))
    examples = load_split(cfg, tok)[:3]
    read = lambda p: [(r["example_id"], r["generation"], r["row_steps"]) for r in map(json.loads, open(p))]
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full")
        evaluate(gm, tok, examples, cfg, "off", full)
        sh = os.path.join(d, "sh")
        evaluate(gm, tok, examples, cfg, "off", sh, shard=(1, 2))
        evaluate(gm, tok, examples[:1], cfg, "off", sh, shard=(0, 2))
        evaluate(gm, tok, examples, cfg, "off", sh, shard=(0, 2), resume=True)  # skips the done example
        merged = merge_shards(sh, [e.example_id for e in examples])
        assert read(os.path.join(full, "rows_thr0.95.jsonl")) == read(os.path.join(sh, "rows_thr0.95.jsonl"))
        assert merged["summaries"]["thr0.95"]["n"] == 3
        # a 2-GPU partial run resumed with 3 GPUs: done examples are skipped, the merge covers every example once
        mix = os.path.join(d, "mix")
        evaluate(gm, tok, examples, cfg, "off", mix, shard=(1, 2))
        for i in range(3):
            evaluate(gm, tok, examples, cfg, "off", mix, shard=(i, 3), resume=True)
        merge_shards(mix, [e.example_id for e in examples])
        assert read(os.path.join(full, "rows_thr0.95.jsonl")) == read(os.path.join(mix, "rows_thr0.95.jsonl"))
        # rows from another configuration are refused on resume
        cfg2 = cfg.copy()
        cfg2.generation.commit_threshold = 0.5
        try:
            evaluate(gm, tok, examples, cfg2, "off", sh, shard=(0, 2), resume=True)
            assert False, "resume across a changed fingerprint must fail"
        except RuntimeError:
            pass


def test_summary_counts_tpf_both_ways_and_flags_truncation():
    row = dict(is_correct=True, parsed_answer="1", stop_reason="eos", row_steps=10, output_tokens=30, nfe_dinfer=12,
               tpf_dinfer=30 / 12, latency_s=2.0, receiver_steps=8, group_steps=4, group_tokens=8, group_committed=6,
               group_identity_changed=1, fallback_rows=2, mask_argmax_fallbacks=0, cache_used=False)
    rows = [row, dict(row, is_correct=False, stop_reason="length", row_steps=20, output_tokens=20, nfe_dinfer=10,
                      tpf_dinfer=2.0, latency_s=4.0)]
    s = summarize_rows(rows)
    assert s["n"] == 2 and s["accuracy"] == 0.5 and s["truncation_rate"] == 0.5
    assert abs(s["tpf_dinfer_mean"] - (2.5 + 2.0) / 2) < 1e-9 and abs(s["tpf_dinfer_pooled"] - 50 / 22) < 1e-9
    assert abs(s["tokens_per_step"] - 50 / 30) < 1e-9 and s["latency_median_s"] == 3.0


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all(globals())
