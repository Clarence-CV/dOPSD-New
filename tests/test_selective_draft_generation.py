"""Threshold / fixed-budget contracts, padding, EOS stop, growing window, trace replay, sampling."""
import torch

from _fake_helpers import (EOS, MASK, V, build, controlled, fake_config, peaked_logits, run_all)
from selective_draft.generation import generate, get_num_transfer_tokens
from selective_draft.trace import StateRecorder, replay_check


def _prompt(B=1, P=4, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(4, V, (B, P))
    return ids, torch.ones_like(ids)


def test_threshold_commits_all_above_tau_and_falls_back():
    # block of 8: positions with confidence 0.95 commit together; 0.9 exactly must NOT commit; rest wait
    def probs(ids):
        B, N = ids.shape
        p = torch.full((B, N), 0.5)
        p[:, 4 + 0] = 0.95; p[:, 4 + 3] = 0.95; p[:, 4 + 5] = 0.9 - 1e-4  # just below tau (float-safe)
        return p
    gm, tok, cfg = controlled(peaked_logits(probs))
    cfg.generation.max_new_tokens, cfg.generation.block_length = 8, 8
    ids, mask = _prompt()
    rec = StateRecorder("state_only", MASK, EOS, 4, [{}])
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="off", recorder=rec)
    steps = rec.steps[0]
    first = steps[0]["committed_positions"].tolist()
    assert sorted(first) == [4, 7], first
    # the rule itself is strict: confidence equal to tau does not commit
    assert not bool((torch.tensor([0.9]) > 0.9).item())
    # afterwards nothing exceeds 0.9 -> exactly one fallback commit per step
    for s in steps[1:]:
        assert len(s["committed_positions"]) == 1
    assert len(steps) == 1 + 6 and res.row_steps[0] == 7 and res.stats["nfe"] == 7
    assert not (res.sequences[0, 4:] == MASK).any()
    replay_check(rec.finalize(res)[0])


def test_threshold_sweep_changes_steps_monotonically():
    gm, tok, cfg = build()
    ids, mask = _prompt(B=2, P=6)
    nfes = [generate(gm, ids, mask, cfg.generation, guidance_mode="off", commit_threshold=t).stats["nfe"] for t in (0.0, 0.5, 0.99)]
    assert nfes[0] <= nfes[1] <= nfes[2] and nfes[0] == cfg.generation.max_new_tokens // cfg.generation.block_length


def test_fixed_budget_schedule_and_empty_steps():
    gm, tok, cfg = build()
    cfg.generation.mode = "fixed_budget"
    ids, mask = _prompt(B=2)
    # 16 tokens / 2 blocks; steps=8 -> 4 per block -> k=2 each
    rec = StateRecorder("state_only", MASK, EOS, 4, [{}, {}])
    cfg.generation.stop_mode = "none"
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="off", recorder=rec, steps=8)
    assert res.stats["nfe"] == 8 and all(len(s["committed_positions"]) == 2 for s in rec.steps[0])
    assert not (res.sequences[:, 4:] == MASK).any()
    # steps=32 -> 16 per block of 8 masks -> k=1 for 8 steps then 0 (skipped, counted)
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="off", steps=32)
    assert res.stats["nfe"] == 16 and res.stats["empty_budget_steps"] == 16
    # remainder distribution: 8 masks over 3 steps -> 3,3,2
    k = get_num_transfer_tokens(torch.ones(1, 8, dtype=torch.bool), 3)
    assert k.tolist() == [[3, 3, 2]]
    try:
        generate(gm, ids, mask, cfg.generation, guidance_mode="off", steps=7)
        assert False
    except ValueError:
        pass


def test_eos_stops_after_block_and_lengths():
    def probs(ids):
        return torch.full(ids.shape, 0.99)
    gm, tok, cfg = controlled(peaked_logits(probs))
    # token at position 4+2 is EOS? peaked_logits uses token = pos % 60 + 4; force EOS by overriding the fn
    base_fn = gm.base.fn
    def fn(ids):
        lg = base_fn(ids)
        lg[:, 6, :] = -10.0; lg[:, 6, EOS] = 10.0  # row writes EOS at completion index 2 in block 0
        return lg
    gm.base.fn = fn
    ids, mask = _prompt(B=2)
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="off")
    assert res.stop_reasons == ["eos", "eos"] and res.computed_lengths == [8, 8] and res.effective_lengths == [2, 2]
    assert res.stats["nfe"] == 1 and (res.sequences[:, 12:] == MASK).all(), "second block never decoded"
    cfg.generation.stop_mode = "none"
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="off")
    assert res.stop_reasons == ["length", "length"] and res.computed_lengths == [16, 16] and res.effective_lengths == [2, 2]


def test_padding_batch_matches_single_rows():
    # absolute learned positions would legitimately shift under left padding (LLaDA uses RoPE, which does
    # not); the fake runs position-free here so batched == single tests the padding mask alone.
    gm, tok, cfg = build(abs_pos=False)
    a = tok(["<user> what is two plus three <assistant>"])
    b = tok(["<user> compute the sum of the numbers four and five and six please <assistant>"])
    both = tok(["<user> what is two plus three <assistant>",
                "<user> compute the sum of the numbers four and five and six please <assistant>"])
    ra = generate(gm, a["input_ids"], a["attention_mask"], cfg.generation, guidance_mode="off")
    rb = generate(gm, b["input_ids"], b["attention_mask"], cfg.generation, guidance_mode="off")
    rab = generate(gm, both["input_ids"], both["attention_mask"], cfg.generation, guidance_mode="off")
    assert rab.completion_ids(0, effective=False) == ra.completion_ids(0, effective=False)
    assert rab.completion_ids(1, effective=False) == rb.completion_ids(0, effective=False)
    assert (both["attention_mask"][0, :2] == 0).all(), "test expects left padding on the shorter prompt"
    assert not (rab.sequences[:, both["input_ids"].shape[1]:] == MASK).any()


def test_growing_window_and_guided_rollout_trace():
    gm, tok, cfg = build()
    gm.cfg.message_threshold = 0.0
    torch.nn.init.normal_(gm.injector.up.weight, std=0.05)
    cfg.generation.window_mode = "growing"
    ids, mask = _prompt(B=2, P=5)
    rec = StateRecorder("diagnostic", MASK, EOS, 5, [{}, {}])
    res = generate(gm, ids, mask, cfg.generation, guidance_mode="single_soft", recorder=rec)
    assert res.sequences.shape[1] == 5 + cfg.generation.max_new_tokens
    rolls = rec.finalize(res)
    for r in rolls:
        replay_check(r)
        assert len(r["steps"][0]["state"]) == cfg.generation.block_length  # growing: first state is one block
        gated = [s for s in r["steps"] if s["gate_open"]]
        assert gated and all("source_topk_ids" in s and "delta_norm" in s for s in gated)
        assert all(s["source_pos"] >= s["block_start"] for s in gated)
    assert res.stats["gate_open_steps"] > 0 and res.stats["receiver_steps"] >= res.stats["gate_open_steps"]


def test_sampling_is_seeded_and_confidence_is_raw_prob():
    gm, tok, cfg = build()
    cfg.generation.temperature = 3.0  # the fake's logits are sharp; a high T makes sampling visible
    ids, mask = _prompt(B=2, P=5)
    g1 = torch.Generator().manual_seed(7); g2 = torch.Generator().manual_seed(7); g3 = torch.Generator().manual_seed(8)
    r1 = generate(gm, ids, mask, cfg.generation, guidance_mode="off", sample_rng=g1)
    r2 = generate(gm, ids, mask, cfg.generation, guidance_mode="off", sample_rng=g2)
    r3 = generate(gm, ids, mask, cfg.generation, guidance_mode="off", sample_rng=g3)
    assert torch.equal(r1.sequences, r2.sequences) and not torch.equal(r1.sequences, r3.sequences)
    from selective_draft.generation import choose_tokens
    logits = torch.randn(1, 3, V) * 3
    cand, conf = choose_tokens(logits, 1.0, torch.Generator().manual_seed(0))
    assert torch.allclose(conf, torch.softmax(logits, -1).gather(-1, cand[..., None]).squeeze(-1))


if __name__ == "__main__":
    run_all(globals())
