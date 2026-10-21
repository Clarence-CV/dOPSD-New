"""CPU contract tests for group communication, same-state continuation, sibling PI and training."""
import copy
import os
import tempfile
from unittest.mock import patch

import torch

from _fake_helpers import MASK, V, build, controlled, fake_config, make_ctx, run_all
from selective_draft.checkpoint import load_snapshot
from selective_draft.data import load_split
from selective_draft.generation import generate
from selective_draft.joint import JointTrainer, certify_tuple, select_reference
from selective_draft.model import GuidanceSpec
from selective_draft.trace import StateRecorder, replay_check
from selective_draft.training import run_training


def _fixed_grade(value):
    return lambda *a, **k: {"parsed_answer": None, "is_correct": value, "scorer": "legacy", "verdicts": {"legacy": value}}


def joint_config(**over):
    base = {"guidance.mode": '"group_soft"', "guidance.group_size": 2,
            "teacher.mode": '"joint"', "training.rollout_temperature": 0.9,
            "training.max_rollout_attempts": 8, "training.rollouts_per_update": 1,
            "training.states_per_rollout": 1, "generation.max_new_tokens": 4,
            "generation.block_length": 4, "generation.stop_mode": '"none"'}
    return fake_config(**dict(base, **over))


def test_group_parity_selection_and_peer_message_gradients():
    cfg = joint_config(**{"guidance.group_size": 3})
    gm, _, _ = build(cfg)
    x = torch.tensor([[8, MASK, MASK, MASK, MASK], [9, MASK, 7, 7, 7]])
    ctx = make_ctx(x, 1, 5)
    gm.set_stage("joint")
    off = gm(x, mode="off").logits
    out = gm(x, context=ctx, mode="group_soft")
    assert torch.equal(off, out.logits), "zero group injection must preserve baseline exactly"
    gi = out.guidance
    assert gi.gate_open.tolist() == [True, False]
    assert len(gi.group_pos[0].unique()) == 3 and bool((gi.group_pos[1] == -1).all())
    loss = out.logits[0, 1:, 6].sum()
    loss.backward()
    assert gm.injector.up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in gm.base.parameters())
    assert all(p.grad is None for p in gm.readout.parameters())
    with torch.no_grad():
        gm.injector.up.weight.normal_(std=0.1)
    natural = GuidanceSpec(gi.source_pos, gi.gate_open, gi.source_probs,
                           group_pos=gi.group_pos, group_probs=gi.group_probs)
    changed = copy.deepcopy(natural)
    changed.group_probs[0, 0].zero_()
    changed.group_probs[0, 0, 10] = 1
    a = gm(x, context=ctx, mode="group_soft", spec=natural).logits
    b = gm(x, context=ctx, mode="group_soft", spec=changed).logits
    peer = int(gi.group_pos[0, 1])
    assert not torch.allclose(a[0, peer], b[0, peer]), "native tail must carry a peer's message"
    assert torch.equal(a[1], off[1]), "singleton row must bypass group injection"


def test_group_ties_choose_earliest_positions():
    gm, _, _ = build(joint_config())
    x = torch.tensor([[8, MASK, MASK, MASK, MASK]])
    with patch.object(gm, "readout_log_probs", side_effect=lambda h: torch.zeros(len(h), V).log_softmax(-1)):
        gi = gm(x, context=make_ctx(x, 1, 5), mode="group_readout").guidance
    assert gi.group_pos.tolist() == [[1, 2]]


def test_continuation_preserves_known_tokens_and_replays_the_same_decoder():
    for window in ("fixed", "growing"):
        cfg = joint_config(**{"generation.max_new_tokens": 8, "generation.block_length": 4,
                              "generation.window_mode": f'"{window}"'})
        gm, _, _ = build(cfg)
        prompt = torch.tensor([[8, 9]])
        pmask = torch.ones_like(prompt)
        rec = StateRecorder("state_only", MASK, gm.backend.eos_token_id, 2,
                            [{"prompt_ids": [8, 9], "prompt_attention_mask": [1, 1]}])
        result = generate(gm, prompt, pmask, cfg.generation, guidance_mode="group_soft", recorder=rec)
        roll = rec.finalize(result)[0]
        replay_check(roll)
        # Exercise both a partial current block and a completed previous block.
        for step in roll["steps"]:
            initial = torch.cat([prompt[0], step["state"].long()])[None]
            saved = initial.clone()
            resumed = generate(gm, prompt, pmask, cfg.generation, guidance_mode="group_soft",
                               initial_state=initial, start_block=step["block_id"], initial_step=step["step_id"])
            assert torch.equal(initial, saved), "branch must not mutate the recorded state"
            assert torch.equal(resumed.sequences, result.sequences)
            known = initial != MASK
            assert torch.equal(resumed.sequences[:, :initial.shape[1]][known], initial[known])


def test_leave_one_out_reveals_only_peers_and_batches_independent_inputs():
    def logits(ids):
        out = torch.zeros(*ids.shape, V)
        out[:, 1, 4] = 4
        out[:, 2, 5] = 4
        return out
    gm, _, _ = controlled(logits)
    x = torch.tensor([9, MASK, MASK, MASK, 10])
    positions, tokens = torch.tensor([1, 2]), torch.tensor([4, 5])
    for batch in (1, 2):
        gm.base.calls.clear()
        lq, stats = certify_tuple(gm, x, torch.ones_like(x), positions, tokens, batch)
        assert lq.argmax(-1).tolist() == [4, 5]
        actual = torch.cat([c[0] for c in gm.base.calls])
        assert actual.tolist() == [[9, MASK, 5, MASK, 10], [9, 4, MASK, MASK, 10]]
        assert stats["teacher_inputs"] == 2
        assert stats["teacher_forwards"] == (2 if batch == 1 else 1)
    bad, _ = certify_tuple(gm, x, torch.ones_like(x), positions, torch.tensor([4, 6]))
    assert bad is None


def test_whole_tuple_projection_deduplicates_and_rejects_incompatible_witnesses():
    def logits(ids):
        out = torch.zeros(*ids.shape, V)
        out[:, 1, 4], out[:, 2, 5] = 5, 5
        return out
    gm, _, _ = controlled(logits)
    x = torch.tensor([9, MASK, MASK, MASK, 10])
    def candidate(a, b, known=10, correct=True):
        return {"meta": {"prompt_len": 1, "prompt_ids": [9]},
                "final_state": torch.tensor([a, b, 8, known]), "computed_length": 4, "is_correct": correct}
    # Draft 4,4 is nearest but fails certification. The valid 4,5 candidate must win as a whole.
    probs = torch.zeros(2, V)
    probs[:, 4] = 1
    bank = [candidate(4, 5), candidate(4, 4), candidate(4, 5), candidate(4, 4, known=11),
            candidate(4, 4, correct=False)]
    ref, stats, audit = select_reference(gm, x, torch.ones_like(x), torch.tensor([1, 2]), probs, bank)
    assert ref.tokens.tolist() == [4, 5] and ref.witness_index == 0
    assert stats["unique_candidates"] == 2 and stats["compatible_candidates"] == 3
    assert [a["accepted"] for a in audit] == [False, True]
    assert audit[0]["distance"] == 0
    empty, _, _ = select_reference(gm, x, torch.ones_like(x), torch.tensor([1, 2]), probs, [bank[-1]])
    assert empty is None


def _peaked_model(cfg):
    gm, tok, _ = build(cfg)
    original = gm.base.forward
    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        # Make same-token continuations overwhelmingly likely, while retaining attention and gradients.
        out.logits = out.logits * 0.05
        out.logits[..., 7] = out.logits[..., 7] + 8
        return out
    gm.base.forward = forward
    return gm, tok


def test_joint_training_eight_branches_frozen_modules_and_exact_resume():
    cfg = joint_config(**{"training.max_steps": 4, "training.warmup_ratio": 0.25,
                          "training.learning_rate_guidance": 0.001})
    with tempfile.TemporaryDirectory() as d, \
            patch("selective_draft.training.grade", side_effect=_fixed_grade(False)), \
            patch("selective_draft.joint.grade", side_effect=_fixed_grade(True)):
        gm, tok = _peaked_model(cfg)
        trainer = JointTrainer(gm, cfg, os.path.join(d, "first"), load_split(cfg, tok), tok)
        frozen = {n: p.clone() for n, p in gm.named_parameters() if not p.requires_grad}
        run_training(trainer, 2, 0)
        assert gm.injector.up.weight.abs().sum() > 0
        assert all(torch.equal(dict(gm.named_parameters())[n], p) for n, p in frozen.items())
        files = sorted(os.listdir(trainer.reference_dir))
        record = torch.load(os.path.join(trainer.reference_dir, files[0]), weights_only=False)
        assert len(record["candidates"]) == 8 and record["selected_tokens"] is not None
        assert len(record["group_positions"]) == 2
        assert all(r["meta"]["generation"]["temperature"] == 0.9 for r in record["candidates"])
        # Natural states were kept even though their original rollout failed task verification.
        assert trainer.state["states_seen"] >= 2
        ckpt = os.path.join(d, "first", "checkpoint-step-2")
        run_training(trainer, 3, 0)
        expected = copy.deepcopy(gm.module_state_dict())
        gm2, tok2 = _peaked_model(cfg)
        resumed = JointTrainer(gm2, cfg, os.path.join(d, "resumed"), load_split(cfg, tok2), tok2)
        run_training(resumed, 3, 0, resume=ckpt)
        for k, value in expected.items():
            assert torch.equal(gm2.module_state_dict()[k], value), f"resume mismatch: {k}"
        assert trainer.state["reference_states"] == resumed.state["reference_states"]
        bad = cfg.copy()
        bad.guidance.group_size = 3
        try:
            load_snapshot(os.path.join(d, "first", "snapshot"), gm2, bad)
            assert False, "a joint snapshot must retain its group size"
        except ValueError:
            pass


def test_no_valid_reference_skips_update_and_retains_failures():
    cfg = joint_config(**{"training.max_consecutive_empty_updates": 2})
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.joint.grade", side_effect=_fixed_grade(False)):
        gm, tok = _peaked_model(cfg)
        trainer = JointTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        before = copy.deepcopy(gm.module_state_dict())
        result = trainer.update()
        assert result["skipped"] == 1 and trainer.state["optimizer_step"] == 0
        record = torch.load(os.path.join(trainer.reference_dir, os.listdir(trainer.reference_dir)[0]), weights_only=False)
        assert len(record["candidates"]) == 8 and record["selected_tokens"] is None
        assert record["stats"]["teacher_forwards"] == 0
        assert all(torch.equal(before[k], v) for k, v in gm.module_state_dict().items())
        try:
            trainer.update()
            assert False, "repeated certification failures must stop, not resample forever"
        except RuntimeError as exc:
            assert "consecutive" in str(exc)


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all(globals())
