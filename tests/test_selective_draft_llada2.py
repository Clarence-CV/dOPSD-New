"""CPU contract tests for the LLaDA2.0-mini path, run against a tiny model built from the vendored official
remote code (needs Transformers 4.57): block-causal attention, autograd-equivalent MoE, decoder parity with the
official HF generate, EOS/cap semantics, the threshold group rule and grading."""
import math
import os
import tempfile
import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import _fake_helpers  # noqa: F401  (puts the repo root on sys.path)
from _fake_helpers import ROOT, run_all
from selective_draft.config import Config
from selective_draft.decoding import choose_tokens
from selective_draft.generation import generate
from selective_draft.loading import build_model_and_tokenizer
from selective_draft.model import StepContext
from selective_draft.trace import StateRecorder, replay_check

MASK, EOS, V, BL = 3, 2, 64, 8


def mini_config(**over) -> Config:
    cfg = Config.load(os.path.join(ROOT, "configs", "selective_draft", "fake_llada2_cpu.json"))
    return cfg.apply_overrides([f"{k}={v}" for k, v in over.items()]).validate()


def build(cfg=None, **fake_kw):
    cfg = cfg or mini_config()
    cfg.model.fake_kwargs = dict(cfg.model.fake_kwargs, **fake_kw)
    gm, tok = build_model_and_tokenizer(cfg, torch.device("cpu"))
    return gm, tok, cfg


def controlled(fn, cfg=None):
    """Real backend/decoder around logits fn(input_ids) -> [B, N, V]."""
    gm, tok, cfg = build(cfg)
    calls = []

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        calls.append((input_ids.clone(), attention_mask))
        return SimpleNamespace(logits=fn(input_ids))
    gm.base.forward = types.MethodType(forward, gm.base)
    return gm, tok, cfg, calls


def test_block_mask_hides_future_blocks_and_mixes_inside_a_block():
    gm, _, _ = build()
    torch.manual_seed(1)
    ids = torch.randint(4, V, (1, 3 * BL))
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
    try:
        gm.backend.forward_logits(ids, torch.cat([torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 3 * BL - 1, dtype=torch.long)], 1))
        raise AssertionError("padding must be rejected")
    except ValueError:
        pass


def test_prefix_forward_matches_the_same_prefix_inside_a_longer_window():
    gm, _, _ = build()
    torch.manual_seed(2)
    ids = torch.randint(4, V, (1, 4 * BL))
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
        a = official(layer, flat, idx, w)
        b = layer.moe_infer(flat, idx, w)
    assert torch.equal(a, b)
    # Routed-expert Jacobian: directional finite difference vs autograd (routing is piecewise constant). The
    # router casts to fp32 internally, so the comparison runs in fp64 with an fp32-sized step and tolerance.
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
            idx_p, _, _ = layer64.gate(x + eps * v)
            idx_m, _, _ = layer64.gate(x - eps * v)
        assert torch.equal(idx_p, idx_m), "routing switched inside the finite-difference step; change the seed"
        fd = ((yp - ym) * direction).sum() / (2 * eps)
        assert abs(float(fd) - float((g * v).sum())) < 1e-3 * max(1.0, abs(float(fd))), (float(fd), float((g * v).sum()))
        # The official no_grad path drops the routed part: its gradient differs from the finite difference.
        x2 = x.detach().clone().requires_grad_(True)
        layer64.moe_infer = types.MethodType(official, layer64)
        y2, _ = layer64(x2)
        (y2 * direction).sum().backward()
        assert abs(float((x2.grad * v).sum()) - float(fd)) > 1e-4
    finally:
        layer.float()


def test_decoder_matches_official_generate_with_the_argmax_fix():
    cfg = mini_config(**{"generation.stop_mode": '"block_end"', "generation.forbid_mask_candidate": "false",
                         "generation.max_new_tokens": 40})
    gm, _, cfg = build(cfg)
    base = gm.base.to(torch.bfloat16)  # official generate builds a bf16 mask
    gm.backend._mask_cache.clear()

    def argmax_sampler(self, logits, temperature=0.0, top_k=0, top_p=1.0):  # LLaDA2.1's T=0 branch
        probs = F.softmax(logits.float(), -1)
        tok = probs.argmax(-1)
        return tok, probs.gather(-1, tok[..., None]).squeeze(-1)
    base._sample_with_temperature_topk_topp = types.MethodType(argmax_sampler, base)
    for P in (5, 8, 13, 16):
        torch.manual_seed(P)
        ids = torch.randint(4, V, (1, P))
        for tau in (0.95, 0.4):
            ref = base.generate(ids, temperature=0.0, block_length=BL, steps=BL, gen_length=40, threshold=tau,
                                eos_id=EOS, mask_id=MASK, eos_early_stop=False)
            r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off", commit_threshold=tau)
            ours = r.sequences[0, P:P + r.effective_lengths[0] + (r.stop_reasons[0] == "eos")]
            assert torch.equal(ref[0], ours), (P, tau, ref[0].tolist(), ours.tolist())


def test_greedy_ignores_rng_and_grid_is_absolute():
    gm, _, cfg = build()
    ids = torch.randint(4, V, (1, 13))
    a = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off",
                 sample_rng=torch.Generator().manual_seed(0))
    b = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off",
                 sample_rng=torch.Generator().manual_seed(99))
    assert torch.equal(a.sequences, b.sequences)
    # P=13, B=8, G=20 -> blocks [8,16) [16,24) [24,32) [32,40): 3 + 8 + 8 + 8 positions decoded
    assert a.stats["first_block"] == 1 and a.stats["physical_end"] == 40 and a.computed_lengths == [27]
    assert a.effective_lengths == [20] and a.stop_reasons == ["length"]
    rec = StateRecorder("state_only", MASK, EOS, 13, [{}])
    c = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off", recorder=rec)
    roll = rec.finalize(c)[0]
    replay_check(roll)
    assert roll["steps"][0]["block_start"] == 8 and int(roll["steps"][0]["eligible_positions"].min()) == 13


def _scripted(plan):
    """plan: absolute position -> (token, forward call from which it is confident). Others: p=0.25 ties."""
    state = {"calls": 0}

    def fn(ids):
        B, N = ids.shape
        logits = torch.full((B, N, V), -9.0)
        logits[..., 10:14] = 0.0  # unconfident default: four tied tokens
        for pos, (tok, when) in plan.items():
            if pos < N and state["calls"] >= when:
                logits[:, pos, :] = -9.0
                logits[:, pos, tok] = 9.0
        logits[..., MASK] = -30.0
        state["calls"] += 1
        return logits
    return fn


def test_prefix_complete_eos_stops_when_the_hole_is_filled_without_a_new_eos():
    # P=6, B=8: call 0 fills 6..7; in block [8,16) EOS commits at 10 on call 1, 8 and 9 only on calls 2 and 3:
    # no new EOS arrives when the left side completes, the stop must still happen (HF would miss it).
    plan = {6: (20, 0), 7: (21, 0), 10: (EOS, 1), 8: (22, 2), 9: (23, 3)}
    cfg = mini_config(**{"generation.max_new_tokens": 30})
    gm, tok, cfg, calls = controlled(_scripted(plan), cfg)
    ids = torch.full((1, 6), 30)
    r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
    assert r.stop_reasons == ["eos"] and r.effective_lengths == [4] and r.stats["eos_position"] == 4
    assert r.completion_ids(0) == [20, 21, 22, 23]
    # the forward that would decode the rest of block [8,16) never happens
    assert len(calls) == r.row_steps[0] == 4
    # block_end keeps decoding the block, then stops at the same EOS
    cfg2 = mini_config(**{"generation.max_new_tokens": 30, "generation.stop_mode": '"block_end"'})
    gm2, _, cfg2, calls2 = controlled(_scripted(plan), cfg2)
    r2 = generate(gm2, ids, torch.ones_like(ids), cfg2.generation, guidance_mode="off")
    assert r2.stop_reasons == ["eos"] and r2.completion_ids(0) == [20, 21, 22, 23]
    assert r2.row_steps[0] > r.row_steps[0]


def test_eos_beyond_the_cap_is_ignored_and_groups_skip_the_rounded_tail():
    # P=6, cap 4 -> semantic end 10, physical end 16. EOS at 12 (outside the cap) must not stop decoding.
    plan = {12: (EOS, 0)}
    cfg = mini_config(**{"generation.max_new_tokens": 4})
    gm, tok, cfg, calls = controlled(_scripted(plan), cfg)
    ids = torch.full((1, 6), 30)
    r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
    assert r.stop_reasons == ["length"] and r.effective_lengths == [4] and r.stats["physical_end"] == 16
    # threshold group rule: readout = native lens at the last layer; positions >= 10 never join G
    cfg = mini_config(**{"generation.max_new_tokens": 4, "guidance.mode": '"group_readout"', "guidance.early_layer": 4})
    gm, tok, cfg = build(cfg)
    torch.manual_seed(5)
    x = torch.cat([torch.randint(4, V, (1, 6)), torch.full((1, 10), MASK)], 1)
    elig = torch.zeros_like(x, dtype=torch.bool)
    elig[:, 8:16] = True
    ctx = StepContext(elig, torch.ones(1, dtype=torch.bool), torch.tensor([8]), torch.tensor([16]),
                      torch.zeros(1, dtype=torch.long), commit_threshold=0.0,
                      guidance_mask=elig & (torch.arange(16) < 10)[None])
    out = gm(x, torch.ones_like(x), context=ctx)
    gi = out.guidance
    assert set(gi.group_pos[0][gi.group_pos[0] >= 0].tolist()) == {8, 9} and int(gi.n_high[0]) == 2


def test_threshold_group_takes_every_confident_position_and_nothing_else():
    cfg = mini_config(**{"guidance.mode": '"group_readout"', "guidance.early_layer": 4})
    gm, tok, cfg = build(cfg)
    torch.manual_seed(6)
    x = torch.cat([torch.randint(4, V, (1, 8)), torch.full((1, 8), MASK)], 1)
    elig = torch.zeros_like(x, dtype=torch.bool)
    elig[:, 8:] = True
    logits = gm.base_forward(x)  # early_layer = last layer: readout at init equals the native distribution
    _, conf = choose_tokens(logits[0, 8:], 0.0, None, MASK)
    for tau in sorted(conf.tolist()):
        ctx = StepContext(elig, torch.ones(1, dtype=torch.bool), torch.tensor([8]), torch.tensor([16]),
                          torch.zeros(1, dtype=torch.long), commit_threshold=tau - 1e-6)
        gi = gm(x, torch.ones_like(x), context=ctx).guidance
        expect = {8 + i for i, c in enumerate(conf.tolist()) if c > tau - 1e-6}
        got = set(gi.group_pos[0][gi.group_pos[0] >= 0].tolist())
        assert int(gi.n_high[0]) == len(expect)
        assert got == (expect if len(expect) >= 2 else set()) and bool(gi.gate_open[0]) == (len(expect) >= 2)
        assert gi.skip_reason[0] == ("open" if len(expect) >= 2 else "single_high" if expect else "none_high")


def test_zero_injector_group_soft_matches_off_and_klass_grading():
    cfg = mini_config(**{"guidance.mode": '"group_soft"', "guidance.early_layer": 2,
                         "generation.commit_threshold": 0.3})
    gm, tok, cfg = build(cfg)
    ids = torch.randint(4, V, (1, 11))
    a = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
    b = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="group_soft")
    assert torch.equal(a.sequences, b.sequences) and b.stats["gate_open_steps"] > 0
    from selective_draft.scoring import grade
    g = grade("Compute 1/2 + 1/4.", "\\frac{3}{4}", "so the final answer is $0.75$.", "math", "klass")
    assert g["is_correct"] and g["verdicts"] == {"legacy": False, "klass": True}
    g = grade("Compute 1/2 + 1/4.", "\\frac{3}{4}", "\\boxed{\\frac{3}{4}}", "math", "legacy")
    assert g["is_correct"] and g["verdicts"]["klass"]


def test_sharded_resumable_evaluation_merges_to_the_unsharded_rows():
    from selective_draft.data import load_split
    from selective_draft.evaluation import evaluate, merge_shards
    import json
    gm, tok, cfg = build(mini_config(**{"generation.max_new_tokens": 12}))
    examples = load_split(cfg, tok)[:3]
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full")
        evaluate(gm, tok, examples, cfg, "off", full)
        sh = os.path.join(d, "sh")
        evaluate(gm, tok, examples, cfg, "off", sh, shard=(1, 2))
        evaluate(gm, tok, examples[:1], cfg, "off", sh, shard=(0, 2))
        evaluate(gm, tok, examples, cfg, "off", sh, shard=(0, 2), resume=True)  # skips the done example
        merged = merge_shards(sh, [e.example_id for e in examples])
        read = lambda p: [(r["example_id"], r["generation"], r["row_steps"]) for r in map(json.loads, open(p))]
        assert read(os.path.join(full, "rows_thr0.95.jsonl")) == read(os.path.join(sh, "rows_thr0.95.jsonl"))
        assert merged["summaries"]["thr0.95"]["n"] == 3
        # a 2-GPU partial run resumed with 3 GPUs: done examples are skipped, merge covers every example once
        mix = os.path.join(d, "mix")
        evaluate(gm, tok, examples, cfg, "off", mix, shard=(1, 2))
        for i in range(3):
            evaluate(gm, tok, examples, cfg, "off", mix, shard=(i, 3), resume=True)
        merge_shards(mix, [e.example_id for e in examples])
        assert read(os.path.join(full, "rows_thr0.95.jsonl")) == read(os.path.join(mix, "rows_thr0.95.jsonl"))


def _root_cfg(**over):
    base = {"guidance.mode": '"group_soft"', "guidance.early_layer": 3, "generation.commit_threshold": 0.3,
            "training.sampling_protocol": '"root_success_v2"', "training.rollout_temperature": 0.9,
            "training.max_rollout_attempts": 3, "training.max_steps": 50, "training.warmup_updates": 2,
            "training.learning_rate_guidance": 0.01, "training.max_consecutive_empty_updates": 50,
            "data.split": '"train"'}
    return mini_config(**dict(base, **over))


def _text_grade(*args, **kw):
    """Deterministic in the generated text only, so a resumed run sees the same verdicts."""
    import hashlib
    ok = int(hashlib.md5(args[2].encode()).hexdigest(), 16) % 3 == 0
    return {"parsed_answer": None, "is_correct": ok, "scorer": "legacy", "verdicts": {"legacy": ok}}


def test_root_success_v2_first_success_one_state_no_redraw_and_exact_resume():
    import json
    from unittest.mock import patch
    from selective_draft.data import load_split
    from selective_draft.root_joint import RootSuccessTrainer
    from selective_draft.training import run_training
    cfg = _root_cfg()
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.training.grade", side_effect=_text_grade):
        gm, tok, _ = build(cfg)
        examples = load_split(cfg, tok)
        for e in examples:  # freeze the fake tokenizer's lazily built id->word map before any decoding
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

        # The fake tokenizer learns id->word lazily, so the resumed trainers share `tok` (decoding is grading).
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
        # compare decisions, not text: the fake tokenizer's lazily built vocabulary changes what ids decode to
        strip = lambda rows: [(r["example_id"], r["outcome"], r.get("state_step_id"),
                               [(a["is_correct"], a["nfe"], a["completion_length"]) for a in r.get("attempts", [])])
                              for r in rows]
        assert strip(ledger) == strip(ledger_b), [(a, b) for a, b in zip(strip(ledger), strip(ledger_b)) if a != b]
        for k, v in gm3.module_state_dict().items():
            assert torch.allclose(v, final_a[k], atol=1e-6), k


def test_root_success_v2_exhausted_budget_never_trains():
    from unittest.mock import patch
    from selective_draft.data import load_split
    from selective_draft.root_joint import RootSuccessTrainer
    cfg = _root_cfg(**{"training.max_rollout_attempts": 2})
    never = lambda *a, **k: {"parsed_answer": None, "is_correct": False, "scorer": "legacy", "verdicts": {"legacy": False}}
    with tempfile.TemporaryDirectory() as d, patch("selective_draft.training.grade", side_effect=never):
        gm, tok, _ = build(cfg)
        tr = RootSuccessTrainer(gm, cfg, d, load_split(cfg, tok), tok)
        rec = tr.update()
        assert rec.get("skipped") == 1 and rec["outcome"] == "root_budget_exhausted"
        assert tr.state["root_attempts"] == 2 and tr.state["states_drawn"] == 0
        assert float(gm.injector.up.weight.abs().sum()) == 0.0

# ----------------------------------------------------------------------------- dInfer (DMax) evaluation semantics
def _conf_logits(plan, default=(10, 0.99)):
    """Static logits: position -> (token, confidence). Others share the remaining mass; MASK is impossible."""
    def fn(ids):
        B, N = ids.shape
        logits = torch.zeros(B, N, V)
        for pos in range(N):
            tok, c = plan.get(pos, default)
            logits[:, pos, tok] = math.log(c * (V - 2) / (1 - c))
        logits[..., MASK] = -30.0
        return logits
    return fn


def _dinfer_cfg(**over):
    base = {"generation.commit_rule": '"dinfer"', "generation.stop_mode": '"block_end"',
            "generation.output_view": '"drop_eos"', "generation.max_new_tokens": 16}
    return mini_config(**dict(base, **over))


def test_dinfer_commit_rule_ge_tie_tolerance_and_empty_rows():
    from selective_draft.decoding import dinfer_commits
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
    from selective_draft.decoding import dinfer_block_forwards as fw
    assert fw([8], 8) == 2  # k=2 unrolled forwards, the second one commits nothing
    assert fw([32], 32) == 4  # k=4
    assert fw([10, 10, 12], 32) == 4  # block done after 3 real steps, the 4th forward of the batch is still paid
    assert fw([1, 1, 1], 3) == 3  # fewer than expected_tpf masks: one forward at a time
    assert fw([2] * 16, 32) == 16  # parallelism below expected_tpf never over-runs
    assert fw([], 0) == 0


def test_block_end_stop_drop_eos_view_and_dinfer_nfe():
    # P=8, block 8: block [8,16) commits [8,9] first, then one position per step by descending confidence
    plan = {8: (20, .99), 9: (21, .98), 10: (EOS, .50), 11: (22, .40), 12: (23, .30), 13: (24, .20),
            14: (25, .10), 15: (26, .05)}
    ids = torch.full((1, 8), 30)
    gm, tok, cfg, calls = controlled(_conf_logits(plan), _dinfer_cfg())
    r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
    assert r.stop_reasons == ["eos"] and r.row_steps == [7] and r.computed_lengths == [8]
    assert r.effective_lengths == [2] and r.stats["eos_position"] == 2  # cut-at-EOS meaning is kept
    # dInfer's output keeps every non-EOS token of the decoded block, including those after the EOS
    assert r.completion_ids(0) == [20, 21, 22, 23, 24, 25, 26] and r.stats["output_tokens"] == 7
    assert r.stats["nfe_decode"] == 7 and r.stats["nfe_dinfer"] == 8  # 7 steps + (2 + 5) unrolled - 7 + prefill
    assert abs(r.stats["tpf_dinfer"] - 7 / 8) < 1e-9 and r.stats["commit_counts"] == [2, 1, 1, 1, 1, 1, 1]
    # the same text, but stopping as soon as the EOS has no MASK on its left, needs only 2 steps
    cfg2 = _dinfer_cfg(**{"generation.stop_mode": '"prefix_complete_eos"'})
    gm2, _, cfg2, _ = controlled(_conf_logits(plan), cfg2)
    r2 = generate(gm2, ids, torch.ones_like(ids), cfg2.generation, guidance_mode="off")
    assert r2.row_steps == [2] and r2.effective_lengths == [2]
    # without drop_eos the completion is cut at the EOS
    gm3, _, cfg3, _ = controlled(_conf_logits(plan), _dinfer_cfg(**{"generation.output_view": '"cut_at_eos"'}))
    r3 = generate(gm3, ids, torch.ones_like(ids), cfg3.generation, guidance_mode="off")
    assert r3.completion_ids(0) == [20, 21]


def test_floor_length_policy_matches_the_eval_harness_bucketing():
    ids = torch.full((1, 8), 30)
    for policy, effective, physical in (("dinfer_floor32", 16, 24), ("cap", 18, 32)):
        cfg = _dinfer_cfg(**{"generation.max_new_tokens": 18, "generation.length_policy": f'"{policy}"',
                             "generation.stop_mode": '"none"'})
        gm, tok, cfg, calls = controlled(_conf_logits({}), cfg)  # everything confident: one step per block
        r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
        assert r.stats["max_new_tokens_effective"] == effective and r.stats["physical_end"] == physical, policy
        assert len(r.completion_ids(0)) == (16 if policy == "dinfer_floor32" else 24)  # drop_eos shows every token


def test_argmax_mask_positions_are_excluded_and_a_stuck_block_falls_back():
    # position 9 predicts MASK with 0.99: dInfer never commits it while other positions are available
    plan = {8: (20, .5), 9: (MASK, .99)}
    ids = torch.full((1, 8), 30)
    cfg = _dinfer_cfg(**{"generation.max_new_tokens": 8, "generation.stop_mode": '"none"'})

    def fn(i):
        out = _conf_logits(plan, default=(21, .6))(i)
        out[..., 9, MASK] = math.log(.99 * 62 / .01)  # the one position whose argmax is MASK
        return out
    gm, _, cfg, _ = controlled(fn, cfg)
    r = generate(gm, ids, torch.ones_like(ids), cfg.generation, guidance_mode="off")
    # position 9 is excluded from the first step; the block finishes once the other positions are in, and the
    # last remaining position (argmax MASK everywhere) triggers the counted fallback to its best non-MASK token
    assert r.stats["mask_argmax_fallbacks"] >= 1
    assert bool((r.sequences[0, 8:16] != MASK).all())


def test_prefix_cache_reproduces_the_uncached_decoder_token_for_token():
    base = {"generation.commit_rule": '"dinfer"', "generation.stop_mode": '"block_end"',
            "generation.output_view": '"drop_eos"', "generation.max_new_tokens": 40}
    cfg_n = mini_config(**base)
    cfg_c = mini_config(**dict(base, **{"generation.cache_mode": '"prefix"'}))
    gm, _, _ = build(cfg_n)
    n_checked = 0
    for P in (8, 13, 16, 21, 27):
        torch.manual_seed(P)
        ids = torch.randint(4, V, (1, P))
        for tau in (0.95, 0.5, 0.2):
            a = generate(gm, ids, torch.ones_like(ids), cfg_n.generation, guidance_mode="off", commit_threshold=tau)
            b = generate(gm, ids, torch.ones_like(ids), cfg_c.generation, guidance_mode="off", commit_threshold=tau)
            assert b.stats["cache_used"] and not a.stats["cache_used"]
            assert torch.equal(a.sequences, b.sequences), (P, tau)
            assert a.stats["commit_counts"] == b.stats["commit_counts"], (P, tau)
            assert a.stats["nfe_dinfer"] == b.stats["nfe_dinfer"] and a.completion_ids(0) == b.completion_ids(0)
            n_checked += 1
    assert n_checked == 15
    # a prompt shorter than one block cannot be prefilled by whole blocks: documented fallback, recorded in the row
    short = torch.randint(4, V, (1, 5))
    r = generate(gm, short, torch.ones_like(short), cfg_c.generation, guidance_mode="off")
    assert not r.stats["cache_used"]
    # guidance modes keep the recompute path (the injection hook indexes absolute positions)
    cfg_g = mini_config(**dict(base, **{"generation.cache_mode": '"prefix"', "guidance.mode": '"group_soft"',
                                         "guidance.early_layer": 2}))
    gmg, _, _ = build(cfg_g)
    ids = torch.randint(4, V, (1, 13))
    rg = generate(gmg, ids, torch.ones_like(ids), cfg_g.generation, guidance_mode="group_soft")
    assert not rg.stats["cache_used"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all(globals())
