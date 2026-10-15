"""Regression cases from the code audit; no pretrained weights or GPU are used."""
import copy
import json
import os
import tempfile
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from _fake_helpers import MASK, V, build, controlled, fake_config, run_all
from test_selective_draft_training import _collect
from selective_draft.cli import _device
from selective_draft.data import load_split
from selective_draft.diagnostics import consistency_report
from selective_draft.evaluation import evaluate, summarize_rows
from selective_draft.generation import choose_tokens
from selective_draft.trace import load_rollouts, replay_check
from selective_draft.training import DistCtx, GuidanceTrainer, ReadoutTrainer, run_training


def test_forbidden_candidate_does_not_inflate_confidence():
    p = torch.tensor([[[0.89, 0.06, 0.05], [0.20, 0.10, 0.70]]])
    for temperature in (0.0, 1.0):
        cand, conf = choose_tokens(p.log(), temperature, torch.Generator().manual_seed(1), forbidden_token=2)
        assert not (cand == 2).any()
        assert torch.allclose(conf, p.gather(-1, cand[..., None]).squeeze(-1))
        assert not bool((conf > 0.9).any())


def test_device_binding_uses_local_rank_before_model_loading():
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1"}), \
         patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_device") as bind:
        assert _device(SimpleNamespace(device="auto")) == torch.device("cuda:1")
        bind.assert_called_once_with(torch.device("cuda:1"))


def test_both_stages_resume_after_sampler_reshuffle():
    cfg = fake_config(**{"guidance.message_threshold": 0.0, "generation.max_new_tokens": 4,
                         "generation.block_length": 4, "generation.stop_mode": "none",
                         "training.rollouts_per_update": 1, "training.states_per_rollout": 1})
    for stage in ("readout", "guidance"):
        with tempfile.TemporaryDirectory() as d:
            gm, tok, _ = build(cfg)
            states_path = os.path.join(d, "states.pt")
            if stage == "readout":
                _collect(gm, tok, cfg, states_path, limit=1)
            def trainer(model, tokenizer, name):
                if stage == "readout":
                    return ReadoutTrainer(model, cfg, os.path.join(d, name), [states_path])
                return GuidanceTrainer(model, cfg, os.path.join(d, name), load_split(cfg, tokenizer), tokenizer)
            first = trainer(gm, tok, "first")
            # More updates than data items -> checkpoint is inside a reshuffled epoch.
            steps = len(first.order) + 1
            run_training(first, steps, save_every=0)
            saved_order = list(first.order)
            first.update()
            expected = copy.deepcopy(gm.module_state_dict())
            gm2, tok2, _ = build(cfg)
            resumed = trainer(gm2, tok2, "resumed")
            resumed.resume(os.path.join(d, "first", f"checkpoint-step-{steps}"))
            assert resumed.order == saved_order
            resumed.update()
            for key, value in gm2.module_state_dict().items():
                assert torch.equal(value, expected[key]), (stage, key)


def test_eval_repeats_keep_trace_and_sampling_and_threshold_order_consistent():
    cfg = fake_config(**{"generation.temperature": 3.0, "generation.stop_mode": "none",
                         "generation.max_new_tokens": 8, "generation.block_length": 8,
                         "evaluation.repeats": 2, "evaluation.batch_size": 2})
    with tempfile.TemporaryDirectory() as d:
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)[:3]  # final batch is shorter
        first_dir = os.path.join(d, "first")
        out = evaluate(gm, tok, examples, cfg, "off", first_dir, thresholds=[0.5, 0.9], trace_level="state_only")
        for thr in (0.5, 0.9):
            for roll in load_rollouts(os.path.join(first_dir, f"rollouts_thr{thr}.pt")):
                replay_check(roll)
                assert roll["meta"]["generation"]["commit_threshold"] == thr
            rows = [json.loads(line) for line in open(os.path.join(first_dir, f"rows_thr{thr}.jsonl"))]
            batch_times = {r["batch_id"]: r["latency_s"] for r in rows}
            assert out["summaries"][f"thr{thr}"]["latency_total_s"] == sum(batch_times.values())
        # Timing repeats and an earlier threshold must not change the quality sample.
        cfg.evaluation.repeats = 1
        evaluate(gm, tok, examples, cfg, "off", os.path.join(d, "second"), thresholds=[0.9])
        read = lambda folder: [json.loads(line)["generation"] for line in open(os.path.join(folder, "rows_thr0.9.jsonl"))]
        assert read(first_dir) == read(os.path.join(d, "second"))


def test_batch_metrics_are_counted_once_with_a_short_last_batch():
    rows = []
    for batch_id, size, latency, gates in ((0, 2, 10.0, 6), (1, 1, 4.0, 1)):
        for _ in range(size):
            rows.append(dict(batch_id=batch_id, batch_size=size, latency_s=latency, is_correct=True,
                             row_steps=3, effective_tokens=4, receiver_steps=8, gate_open_steps=gates,
                             gate_source_committed=gates, gate_source_identity_changed=0,
                             batch_nfe=3, stop_reason="eos"))
    rep = summarize_rows(rows)
    assert rep["latency_total_s"] == 14.0 and rep["batches"] == 2
    assert rep["gate_open_per_row"] == 7 / 3 and rep["gate_coverage_receiver_steps"] == 7 / 16


def test_induced_fidelity_uses_off_continuation_not_guided_siblings():
    def logits(ids):
        out = torch.empty(*ids.shape, V)
        for row in range(ids.shape[0]):
            for pos in range(ids.shape[1]):
                token, p = 6, 0.99
                if pos == 2:
                    # Guided C=8 reinforces B=7. The off branch instead writes C=9, then D=10,
                    # and only on the third forward commits B=4.
                    token, p = ((7, 0.95) if int(ids[row, 3]) == 8 else
                                ((4, 0.95) if int(ids[row, 4]) == 10 else (4, 0.85)))
                if pos == 3:
                    token, p = (8 if int(ids[row, 2]) == 7 else 9), 0.95
                if pos == 4:
                    token, p = 10, (0.96 if int(ids[row, 3]) == 9 else 0.80)
                probs = torch.full((V,), (1 - p) / (V - 1)); probs[token] = p
                out[row, pos] = probs.log()
        return out
    gm, _, cfg = controlled(logits)
    step = dict(state=torch.tensor([MASK] * 4), eligible_positions=torch.tensor([1, 2, 3, 4]),
                eligible_confidences=torch.tensor([0.99, 0.95, 0.95, 0.8]),
                committed_positions=torch.tensor([1, 2, 3]), committed_tokens=torch.tensor([6, 7, 8]),
                gate_open=True, source_was_committed=True, source_pos=1, block_start=1, block_end=5, step_id=0)
    roll = {"meta": dict(prompt_ids=[9], prompt_attention_mask=[1], prompt_len=1, guidance_mode="single_soft",
                         generation=asdict(cfg.generation), confidence_policy="raw_model_probability"), "steps": [step]}
    rep = consistency_report(gm, [roll], cfg)
    assert rep["regret"] == 0  # guided siblings are self-consistent
    assert rep["induced_commits"] == 1 and rep["induced_fidelity"] == 0
    assert rep["induced_baseline_wait_steps"] == 2
    roll["meta"]["generation"]["temperature"] = 1.0
    rep = consistency_report(gm, [roll], cfg)
    assert rep["induced_commits"] == 0
    assert rep["induced_fidelity_status"] == {"unsupported_sampling": 1}


def _distributed_worker(rank, store_path, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store_path, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        cfg = fake_config(**{"training.grad_clip": 1e9})
        gm, tok, _ = build(cfg, seed=rank + 10)  # intentionally different initial modules
        trainer = GuidanceTrainer(gm, cfg, out_dir, load_split(cfg, tok), tok, DistCtx(gm.device))
        expected = [torch.nn.Parameter(p.detach().clone()) for p in trainer.params]
        optimizer = torch.optim.AdamW(expected, lr=cfg.training.learning_rate_guidance, weight_decay=0.0)
        for counts in ((2, 0), (1, 2)):
            total = sum(counts)
            local_n = counts[rank]
            if local_n:
                loss = sum(p.sum() for p in trainer.params) * (rank + 1) * local_n / total
                loss.backward()
            trainer._apply_update(0.0, local_n, total)
            mean_grad = sum(n * (r + 1) for r, n in enumerate(counts)) / total
            for param in expected:
                param.grad = torch.full_like(param, mean_grad)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for param, reference in zip(trainer.params, expected):
                assert torch.allclose(param, reference, atol=1e-7)
                gathered = [torch.empty_like(param) for _ in range(2)]
                dist.all_gather(gathered, param)
                assert torch.equal(gathered[0], gathered[1])
        for _ in range(len(trainer.order) + rank + 1):
            trainer._next_example()
        path = trainer.save()
        for r in range(2):
            assert os.path.exists(os.path.join(path, f"rng_state_rank-{r}.pt"))
            assert os.path.exists(os.path.join(path, f"sampler_state_rank-{r}.pt"))
        assert os.path.exists(os.path.join(path, "optimizer.pt"))
        expected_example = trainer._next_example().example_id
        expected_random = torch.rand(4, generator=trainer.gen_data)
        trainer.resume(path)
        assert trainer._next_example().example_id == expected_example
        assert torch.equal(torch.rand(4, generator=trainer.gen_data), expected_random)
        assert trainer.state["states_seen"] == (3 if rank == 0 else 2)
    finally:
        dist.destroy_process_group()


def test_distributed_sync_empty_rank_and_checkpoint_resume():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_distributed_worker, args=(os.path.join(d, "store"), os.path.join(d, "training")), nprocs=2, join=True)


if __name__ == "__main__":
    run_all(globals())


def test_stage2_skips_states_whose_source_reselection_differs():
    cfg = fake_config(**{"guidance.message_threshold": 0.0, "training.rollouts_per_update": 1,
                         "training.states_per_rollout": 2, "training.max_consecutive_empty_updates": 5})
    with tempfile.TemporaryDirectory() as d:
        gm, tok, _ = build(cfg)
        t = GuidanceTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        t.refresh()
        assert len(t.buffer) == 2
        bad = copy.copy(t.buffer[0]); bad.step = dict(bad.step)
        bad.step["source_pos"] = int(bad.step["source_pos"]) + 1  # recorded A no longer matches
        t.buffer = [bad, t.buffer[1]]
        t.cfg.training.refresh_updates = 10 ** 6  # keep the hand-made buffer for this update
        t.state["optimizer_step"] = 1
        rec = t.update()
        assert rec["local_states"] == 1 and rec["reselect_mismatch_total"] == 1
        assert os.path.exists(os.path.join(d, "reselect_mismatch_rank0.jsonl"))


def test_training_rollouts_sample_and_retry_until_correct_like_thu():
    # fake answers are never correct -> every problem uses all attempts and is dropped (THU: loss * 0)
    cfg = fake_config(**{"training.rollout_temperature": 0.9, "training.max_rollout_attempts": 8,
                         "training.filter_correct_only": True, "training.rollouts_per_update": 2,
                         "guidance.message_threshold": 0.0})
    with tempfile.TemporaryDirectory() as d:
        gm, tok, _ = build(cfg)
        t = GuidanceTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        stats = t.refresh()
        assert stats["rollouts"] == 16 and stats["attempts_per_problem"] == 8 and stats["kept"] == 0
        assert t.buffer == []
        rolls = load_rollouts(os.path.join(d, "rollouts", "round-00000_rank0.pt"))
        assert [r["meta"]["attempt"] for r in rolls[:8]] == list(range(8))
        assert all(r["meta"]["generation"]["temperature"] == 0.9 for r in rolls)
        assert cfg.generation.temperature == 0.0  # evaluation / collection stay greedy
        texts = {r["generation"] for r in rolls[:8]}
        assert len(texts) > 1, "sampled retries must differ"
    # a correct first attempt stops the retries
    cfg.training.filter_correct_only = False
    with tempfile.TemporaryDirectory() as d:
        gm, tok, _ = build(cfg)
        t = GuidanceTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        assert t.refresh()["attempts_per_problem"] == 1


def test_retry_validation_uses_the_rollout_temperature():
    for over, ok in (({"training.max_rollout_attempts": 8}, False),
                     ({"training.max_rollout_attempts": 8, "training.rollout_temperature": 0.9}, True),
                     ({"training.rollout_temperature": -1.0}, False)):
        try:
            fake_config(**over)
            assert ok, over
        except ValueError:
            assert not ok, over


def test_stage2_trains_under_fixed_budget_decoding():
    cfg = fake_config(**{"generation.mode": "fixed_budget", "generation.steps": 8, "guidance.message_threshold": 0.0,
                         "training.rollout_temperature": 0.9, "training.max_rollout_attempts": 2})
    with tempfile.TemporaryDirectory() as d:
        gm, tok, _ = build(cfg)
        t = GuidanceTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        rec = t.update()
        assert rec.get("local_states", 0) > 0 and rec["grad_norm"] > 0
        assert float(gm.injector.up.weight.abs().sum()) > 0
