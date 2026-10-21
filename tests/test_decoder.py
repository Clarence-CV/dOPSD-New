"""Block-causal backend and the DMax/dInfer decoder: attention structure, MoE gradients, commit rule, length, stop,
completion view, NFE accounting, prefix KV cache, communication-group rule."""
import math
import types

import torch

from _helpers import BL, EOS, MASK, V, build, conf_logits, controlled, mini_config, prompt, run_all
from selective_draft.decoding import choose_tokens, dinfer_block_forwards, dinfer_commits
from selective_draft.generation import generate
from selective_draft.model import StepContext
from selective_draft.trace import StateRecorder, replay_check


# ----------------------------------------------------------------------------- backend
def test_block_mask_hides_future_blocks_and_mixes_inside_a_block():
    gm, _, _ = build()
    ids = prompt(3 * BL, 1)
    out = gm.backend.forward_logits(ids)
    for pos, same_block in ((2 * BL + 3, range(2 * BL, 3 * BL)), (BL + 1, range(BL, 2 * BL))):
        alt = ids.clone()
        alt[0, pos] = (alt[0, pos] + 1 - 4) % (V - 4) + 4
        out2 = gm.backend.forward_logits(alt)
        past = slice(0, (pos // BL) * BL)
        assert (out[0, past] - out2[0, past]).abs().max() < 1e-5, "a later block changed an earlier block"
        inside = [p for p in same_block if p != pos]
        assert (out[0, inside] - out2[0, inside]).abs().max() > 1e-4, "same-block peers must see each other"
    m = gm.backend.block_mask(2 * BL + 3, ids.device, torch.float32)[0, 0]
    assert m[BL, BL - 1] == 0 and m[BL - 1, BL] == -math.inf and m[2 * BL + 2, 0] == 0


def test_prefix_forward_matches_the_same_prefix_inside_a_longer_window():
    gm, _, _ = build()
    ids = prompt(4 * BL, 2)
    full = gm.backend.forward_logits(ids)
    pre = gm.backend.forward_logits(ids[:, :2 * BL])
    assert (full[:, :2 * BL] - pre).abs().max() < 1e-5


def test_moe_autograd_path_is_bit_identical_and_routes_input_gradients():
    gm, _, _ = build()
    layer = gm.backend.blocks[gm.backend.moe_layers[0]].mlp
    official = type(layer).moe_infer  # the decorated no_grad class method
    torch.manual_seed(3)
    h = torch.randn(1, 10, gm.backend.d_model)
    idx, w, _ = layer.gate(h)
    flat = h.view(-1, h.shape[-1])
    with torch.no_grad():
        assert torch.equal(official(layer, flat, idx, w), layer.moe_infer(flat, idx, w))
    # Routed-expert Jacobian by directional finite difference (routing is piecewise constant). The router casts to
    # fp32 internally, so the comparison runs in fp64 with an fp32-sized step and tolerance.
    x = torch.randn(1, 6, gm.backend.d_model, dtype=torch.float64)
    layer64 = layer.double()
    try:
        x.requires_grad_(True)
        y, _ = layer64(x)
        direction = torch.randn_like(x)
        (y * direction).sum().backward()
        g = x.grad.clone()
        v = torch.randn_like(x)
        eps = 1e-3
        with torch.no_grad():
            yp, _ = layer64(x + eps * v)
            ym, _ = layer64(x - eps * v)
            assert torch.equal(layer64.gate(x + eps * v)[0], layer64.gate(x - eps * v)[0]), "routing switched; change seed"
        fd = ((yp - ym) * direction).sum() / (2 * eps)
        assert abs(float(fd) - float((g * v).sum())) < 1e-3 * max(1.0, abs(float(fd)))
        # the official no_grad path drops the routed part, so its gradient is wrong
        x2 = x.detach().clone().requires_grad_(True)
        layer64.moe_infer = types.MethodType(official, layer64)
        y2, _ = layer64(x2)
        (y2 * direction).sum().backward()
        assert abs(float((x2.grad * v).sum()) - float(fd)) > 1e-4
    finally:
        layer.float()


# ----------------------------------------------------------------------------- dInfer commit rule and accounting
def test_dinfer_commit_rule_ge_tie_tolerance_and_empty_rows():
    f = lambda xs: torch.tensor([xs], dtype=torch.float32)
    el = torch.ones(1, 4, dtype=torch.bool)
    # conf == tau passes (>=), and the best is always in; below-tau positions stay out
    assert dinfer_commits(el, f([0.95, 0.90, 0.96, 0.94]), 0.95).tolist() == [[True, False, True, False]]
    # nobody reaches tau: the best and everything within 1e-5 of it are committed together
    assert dinfer_commits(el, f([0.5, 0.7, 0.700005, 0.2]), 0.95).tolist() == [[False, True, True, False]]
    assert dinfer_commits(el, f([0.5, 0.7, 0.70002, 0.2]), 0.95).tolist() == [[False, False, True, False]]
    # ineligible positions never commit; an all -inf row commits nothing
    mask = torch.tensor([[True, False, True, False]])
    assert dinfer_commits(mask, f([0.99, 0.99, 0.3, 0.99]), 0.95).tolist() == [[True, False, False, False]]
    assert not bool(dinfer_commits(el, f([-math.inf] * 4), 0.95).any())


def test_dinfer_forward_accounting_hand_cases():
    fw = dinfer_block_forwards
    assert fw([8], 8) == 2  # k=2 unrolled forwards, the second one commits nothing
    assert fw([32], 32) == 4  # k=4
    assert fw([10, 10, 12], 32) == 4  # block done after 3 real steps, the 4th forward of the batch is still paid
    assert fw([1, 1, 1], 3) == 3  # fewer than expected_tpf masks: one forward at a time
    assert fw([2] * 16, 32) == 16  # parallelism below expected_tpf never over-runs
    assert fw([], 0) == 0
    assert fw([2, 1], 8) == 2  # a block cut short is charged only for the forwards that ran (and must terminate)


# ----------------------------------------------------------------------------- decoder semantics
def test_greedy_ignores_rng_floored_length_and_absolute_grid():
    gm, _, cfg = build()
    ids = prompt(13, 7)
    a = generate(gm, ids, cfg.generation, guidance_mode="off", sample_rng=torch.Generator().manual_seed(0))
    b = generate(gm, ids, cfg.generation, guidance_mode="off", sample_rng=torch.Generator().manual_seed(99))
    assert torch.equal(a.sequences, b.sequences)
    # P=13, block 8, G=20: total length floor((13+20)/8)*8 = 32 -> blocks [8,16) [16,24) [24,32) -> 19 positions
    assert a.stats["first_block"] == 1 and a.stats["end"] == 32 and a.stats["max_new_tokens_effective"] == 19
    assert a.computed_length in (3, 11, 19) and a.steps == a.stats["nfe"]  # fewer only if an EOS ended a block early
    rec = StateRecorder(MASK, EOS, 13, {"prompt_ids": ids[0].tolist()})
    c = generate(gm, ids, cfg.generation, guidance_mode="off", recorder=rec)
    roll = rec.finalize(c, {})
    replay_check(roll)
    assert roll["steps"][0]["block_start"] == 8 and int(roll["steps"][0]["eligible_positions"].min()) == 13


def test_floor_length_policy_matches_the_eval_harness_bucketing():
    ids = torch.full((1, 8), 30)  # P=8, G=18: floor((8+18)/8)*8 = 24 -> 16 generated, not 18
    cfg = mini_config(**{"generation.max_new_tokens": 18, "generation.early_stop": "false"})
    gm, _, cfg, _ = controlled(conf_logits({}), cfg)  # everything confident: one step per block
    r = generate(gm, ids, cfg.generation, guidance_mode="off")
    assert r.stats["max_new_tokens_effective"] == 16 and r.stats["end"] == 24 and r.steps == 2
    assert len(r.completion) == 16 and r.stop_reason == "length"


def test_block_end_stop_drop_eos_view_and_dinfer_nfe():
    # P=8: block [8,16) commits [8,9] first, then one position per step by descending confidence; EOS lands at 10
    plan = {8: (20, .99), 9: (21, .98), 10: (EOS, .50), 11: (22, .40), 12: (23, .30), 13: (24, .20),
            14: (25, .10), 15: (26, .05)}
    ids = torch.full((1, 8), 30)
    cfg = mini_config(**{"generation.max_new_tokens": 16})
    gm, _, cfg, calls = controlled(conf_logits(plan), cfg)
    r = generate(gm, ids, cfg.generation, guidance_mode="off")
    # the block that contains the EOS is decoded to its end, then generation stops
    assert r.stop_reason == "eos" and r.steps == 7 and r.computed_length == 8 and r.eos_position == 2
    assert len(calls) == 7 and r.stats["commit_counts"] == [2, 1, 1, 1, 1, 1, 1]
    # dInfer's completion keeps every non-EOS token of the decoded block, including those after the EOS
    assert r.completion == [20, 21, 22, 23, 24, 25, 26] and r.stats["output_tokens"] == 7
    # NFE like the eval script: prefill + (k=2 unrolled batch, then one per step) = 1 + 7; TPF = 7 / 8
    assert r.stats["nfe_dinfer"] == 8 and abs(r.stats["tpf_dinfer"] - 7 / 8) < 1e-9
    # without early_stop the next block is decoded as well (one step: everything there is confident)
    cfg2 = mini_config(**{"generation.max_new_tokens": 16, "generation.early_stop": "false"})
    gm2, _, cfg2, _ = controlled(conf_logits(plan), cfg2)
    r2 = generate(gm2, ids, cfg2.generation, guidance_mode="off")
    assert r2.stop_reason == "length" and r2.steps == 8 and len(r2.completion) == 7 + 8 and r2.eos_position == 2


def test_argmax_mask_positions_are_excluded_and_a_stuck_block_falls_back():
    # position 9 predicts MASK with 0.99: dInfer never commits it while other positions are available
    ids = torch.full((1, 8), 30)
    cfg = mini_config(**{"generation.max_new_tokens": 8, "generation.early_stop": "false"})

    def fn(i):
        out = conf_logits({8: (20, .5)}, default=(21, .6))(i)
        out[..., 9, MASK] = math.log(.99 * 62 / .01)
        return out
    gm, _, cfg, _ = controlled(fn, cfg)
    r = generate(gm, ids, cfg.generation, guidance_mode="off")
    # step 1 takes 10..15 (conf .6), step 2 takes 8, then only position 9 (argmax MASK) is left: counted fallback
    assert r.stats["commit_counts"] == [6, 1, 1] and r.stats["mask_argmax_fallbacks"] == 1
    assert bool((r.sequences[0, 8:16] != MASK).all())


# ----------------------------------------------------------------------------- prefix KV cache
def test_prefix_cache_reproduces_the_uncached_decoder_token_for_token():
    base = {"generation.max_new_tokens": 40}
    cfg_n = mini_config(**base)
    cfg_c = mini_config(**dict(base, **{"generation.cache_mode": '"prefix"'}))
    gm, _, _ = build(cfg_n)
    n = 0
    for P in (8, 13, 16, 21, 27):
        ids = prompt(P, P)
        for tau in (0.95, 0.5, 0.2):
            a = generate(gm, ids, cfg_n.generation, guidance_mode="off", commit_threshold=tau)
            b = generate(gm, ids, cfg_c.generation, guidance_mode="off", commit_threshold=tau)
            assert b.stats["cache_used"] and not a.stats["cache_used"]
            assert torch.equal(a.sequences, b.sequences), (P, tau)
            assert a.stats["commit_counts"] == b.stats["commit_counts"] and a.completion == b.completion, (P, tau)
            assert a.stats["nfe_dinfer"] == b.stats["nfe_dinfer"]
            n += 1
    assert n == 15
    # a prompt shorter than one block cannot be prefilled by whole blocks: documented fallback, recorded in the stats
    short = prompt(5, 1)
    assert not generate(gm, short, cfg_c.generation, guidance_mode="off").stats["cache_used"]
    # guidance keeps the recompute path (the injection hook indexes absolute positions)
    cfg_g = mini_config(**dict(base, **{"generation.cache_mode": '"prefix"', "guidance.mode": '"group_soft"'}))
    gmg, _, _ = build(cfg_g)
    assert not generate(gmg, prompt(13, 3), cfg_g.generation, guidance_mode="group_soft").stats["cache_used"]


# ----------------------------------------------------------------------------- communication group rule
def _group_inputs(P=8, seed=6):
    torch.manual_seed(seed)
    x = torch.cat([torch.randint(4, V, (1, P)), torch.full((1, 8), MASK)], 1)
    elig = torch.zeros_like(x, dtype=torch.bool)
    elig[:, P:] = True
    return x, elig


def _ctx(elig, tau, guidance_mask=None):
    return StepContext(elig, torch.ones(1, dtype=torch.bool), torch.tensor([8]), torch.tensor([16]),
                       torch.zeros(1, dtype=torch.long), commit_threshold=tau, guidance_mask=guidance_mask)


def test_group_takes_every_position_at_or_above_tau_and_nothing_else():
    cfg = mini_config(**{"guidance.mode": '"group_readout"', "guidance.early_layer": 4})  # last layer: readout == native
    gm, _, _ = build(cfg)
    x, elig = _group_inputs()
    cand, conf = choose_tokens(gm.base_forward(x)[0, 8:], 0.0, None)
    conf = conf.masked_fill(cand == MASK, 0.0)
    values = sorted(set(conf.tolist()))
    taus = [values[0] - 0.01] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 0.01]
    for tau in taus:
        gi = gm(x, context=_ctx(elig, tau)).guidance
        expect = {8 + i for i, c in enumerate(conf.tolist()) if c >= tau}
        got = set(gi.group_pos[0][gi.group_pos[0] >= 0].tolist())
        assert int(gi.n_high[0]) == len(expect)
        assert got == (expect if len(expect) >= 2 else set()) and bool(gi.gate_open[0]) == (len(expect) >= 2)
        assert gi.skip_reason[0] == ("open" if len(expect) >= 2 else "single_high" if expect else "none_high")


def test_group_never_includes_positions_outside_the_guidance_mask():
    cfg = mini_config(**{"guidance.mode": '"group_readout"', "guidance.early_layer": 4})
    gm, _, _ = build(cfg)
    x, elig = _group_inputs()
    allowed = elig & (torch.arange(16) < 11)[None]  # positions 8..10 only
    gi = gm(x, context=_ctx(elig, 0.0, allowed)).guidance
    cand, _ = choose_tokens(gm.base_forward(x)[0, 8:11], 0.0, None)
    expect = {8 + i for i, c in enumerate(cand.tolist()) if c != MASK}
    assert set(gi.group_pos[0][gi.group_pos[0] >= 0].tolist()) == (expect if len(expect) >= 2 else set())


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all(globals())
