"""Boundary, parity, source selection, gating, hook lifecycle, gradient paths, soft message."""
import torch
import torch.nn.functional as F

from _fake_helpers import (D, EOS, MASK, V, build, fake_config, make_ctx, run_all)
from selective_draft.modules import soft_message
from selective_draft.model import GuidanceSpec


def _state(gm, B=1, P=5, G=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(4, V, (B, P + G))
    x[:, P:] = MASK
    attn = torch.ones_like(x)
    return x, attn, P, P + G


def test_boundary_is_block_output():
    gm, _, cfg = build()
    x, attn, bs, be = _state(gm)
    out = gm(x, attn, context=make_ctx(x, bs, be), mode="off", capture_hidden=True)
    inner = gm.base.model
    h = inner.transformer["wte"](x) + inner.transformer["wpe"](torch.arange(x.shape[1]))[None]
    for blk in inner.transformer["blocks"][: cfg.guidance.early_layer]:
        h, _ = blk(h)
    assert torch.allclose(out.hidden_early, h), "captured hidden != output of blocks[:early_layer]"
    assert gm.backend.boundary_module(cfg.guidance.early_layer) is inner.transformer["blocks"][cfg.guidance.early_layer - 1]


def test_parity_off_readout_scale0_and_gate_closed():
    gm, _, cfg = build()
    x, attn, bs, be = _state(gm)
    ctx = make_ctx(x, bs, be)
    ref = gm(x, attn, mode="off").logits
    assert torch.equal(gm(x, attn, context=ctx, mode="readout_only").logits, ref)
    # zero-initialised injector -> single_soft identical even when the gate opens
    gm.cfg.message_threshold = 0.0
    out = gm(x, attn, context=ctx, mode="single_soft")
    assert out.guidance.gate_open.all() and torch.equal(out.logits, ref)
    # non-zero injector but scale 0 -> parity
    torch.nn.init.normal_(gm.injector.up.weight, std=0.1)
    gm.cfg.scale = 0.0
    assert torch.equal(gm(x, attn, context=ctx, mode="single_soft").logits, ref)
    gm.cfg.scale = 1.0
    assert not torch.equal(gm(x, attn, context=ctx, mode="single_soft").logits, ref)
    # gate closed (threshold above any confidence) -> parity again
    gm.cfg.message_threshold = 1.0
    out = gm(x, attn, context=ctx, mode="single_soft")
    assert not out.guidance.gate_open.any() and torch.equal(out.logits, ref)
    assert set(out.guidance.skip_reason) == {"below_threshold"}


def test_single_mask_has_no_receiver_and_prompt_is_never_source():
    gm, _, cfg = build()
    gm.cfg.message_threshold = 0.0
    x, attn, bs, be = _state(gm)
    x[:, bs + 1:] = 7  # only one MASK left in the block
    out = gm(x, attn, context=make_ctx(x, bs, be), mode="single_soft")
    assert out.guidance.skip_reason == ["no_receiver"] and not out.guidance.gate_open.any()
    assert int(out.guidance.source_pos[0]) == bs
    # future-block MASKs and prompt positions are not eligible even if present in x
    x, attn, bs, be = _state(gm, G=16)
    ctx = make_ctx(x, bs, bs + 8)
    out = gm(x, attn, context=ctx, mode="readout_only")
    assert bs <= int(out.guidance.source_pos[0]) < bs + 8


def test_tie_breaks_to_smallest_position_and_threshold_boundary():
    gm, _, cfg = build()
    x, attn, bs, be = _state(gm)
    h = torch.randn(1, x.shape[1], D) * 0.01  # near-uniform readouts everywhere ...
    W = gm.backend.ff_out.weight.detach()
    h[0, bs + 1] = W[17] * 50  # ... except two identical positions aligned with one vocabulary row
    h[0, bs + 4] = h[0, bs + 1]
    ctx = make_ctx(x, bs, be)
    info = gm._select_source(h, ctx)
    assert int(info.source_pos[0]) == bs + 1
    conf = float(info.source_confidence[0])
    gm.cfg.message_threshold = conf
    assert gm._select_source(h, ctx).gate_open.all(), "max r == threshold must open the gate (>=)"
    gm.cfg.message_threshold = min(1.0, conf + 1e-6)
    assert not gm._select_source(h, ctx).gate_open.any()
    # chunking must not change the choice
    gm.cfg.readout_position_chunk = 1
    assert int(gm._select_source(h, ctx).source_pos[0]) == bs + 1


def test_inactive_rows_and_mixed_batch():
    gm, _, cfg = build()
    gm.cfg.message_threshold = 0.0
    x, attn, bs, be = _state(gm, B=3)
    x[1, bs:] = 9  # finished row: no masks
    ctx = make_ctx(x, bs, be, active=torch.tensor([True, True, False]))
    out = gm(x, attn, context=ctx, mode="single_soft")
    assert out.guidance.skip_reason == ["open", "no_eligible", "inactive"]
    assert out.guidance.gate_open.tolist() == [True, False, False]
    ref = gm(x, attn, mode="off").logits
    assert torch.equal(out.logits[1:], ref[1:]), "rows without a message must be untouched"


def test_hook_lifecycle_and_nesting():
    gm, _, cfg = build()
    x, attn, bs, be = _state(gm)
    mod = gm.backend.boundary_module(cfg.guidance.early_layer)
    gm(x, attn, context=make_ctx(x, bs, be), mode="readout_only")
    assert len(mod._forward_hooks) == 0 and gm._scratch is None
    # exception inside the forward still removes the hook
    bad_ctx = make_ctx(x, bs, be)
    bad_ctx.eligible_mask = bad_ctx.eligible_mask[:, :3]  # wrong shape -> raises inside the hook
    try:
        gm(x, attn, context=bad_ctx, mode="readout_only")
        assert False, "expected an error"
    except Exception:
        pass
    assert len(mod._forward_hooks) == 0 and gm._scratch is None
    assert torch.equal(gm(x, attn, mode="off").logits, gm.base_forward(x, attn))


def test_gradient_paths():
    gm, _, cfg = build()
    x, attn, bs, be = _state(gm)
    ctx = make_ctx(x, bs, be)
    # stage 1: only readout grads
    gm.set_stage("readout")
    with torch.no_grad():
        out = gm(x, attn, context=ctx, mode="off", capture_hidden=True)
    pos = torch.arange(bs, be)
    log_q = F.log_softmax(out.logits[0, pos].float(), -1)
    log_r = gm.readout_log_probs(out.hidden_early[0, pos])
    loss = (log_q.exp() * (log_q - log_r)).sum(-1).mean()
    loss.backward()
    assert gm.readout.up.weight.grad is not None and gm.readout.up.weight.grad.abs().sum() > 0
    assert gm.readout.down.weight.grad is not None  # zero up -> down grad is zero at init, but exists
    assert all(p.grad is None for p in gm.injector.parameters())
    assert all(p.grad is None for p in gm.base.parameters())
    # stage 2: only injector grads, non-zero through the frozen tail despite zero init
    gm.zero_grad(set_to_none=True)
    gm.set_stage("guidance")
    gm.cfg.message_threshold = 0.0
    out = gm(x, attn, context=ctx, mode="single_soft")
    assert out.logits.requires_grad
    A = int(out.guidance.source_pos[0])
    recv = pos[pos != A]
    target = F.softmax(torch.randn(len(recv), V), -1)
    loss = (target * (target.log() - F.log_softmax(out.logits[0, recv].float(), -1))).sum(-1).mean()
    loss.backward()
    assert gm.injector.up.weight.grad.abs().sum() > 0, "zero-initialised up projection must receive gradient"
    assert all(p.grad is None for p in gm.readout.parameters())
    assert all(p.grad is None for p in gm.base.parameters())
    gm.assert_base_frozen()


def test_soft_message_matches_enumeration_and_depends_on_probs():
    E = torch.randn(V, D)
    p = F.softmax(torch.randn(2, V), -1)
    ref = torch.stack([sum(p[b, v] * E[v] for v in range(V)) for b in range(2)])
    assert torch.allclose(soft_message(p, E, vocab_chunk=7), ref, atol=1e-5)
    assert torch.allclose(soft_message(p, E, vocab_chunk=V), ref, atol=1e-5)
    q = F.softmax(torch.randn(2, V), -1)
    assert not torch.allclose(soft_message(q, E), ref)
    # the message through the injector changes the output once the injector is non-zero
    gm, _, cfg = build()
    torch.nn.init.normal_(gm.injector.up.weight, std=0.1)
    x, attn, bs, be = _state(gm)
    ctx = make_ctx(x, bs, be)
    spec = GuidanceSpec(torch.tensor([bs]), torch.tensor([True]), p[:1])
    spec2 = GuidanceSpec(torch.tensor([bs]), torch.tensor([True]), q[:1])
    o1 = gm(x, attn, context=ctx, mode="single_soft", spec=spec).logits
    o2 = gm(x, attn, context=ctx, mode="single_soft", spec=spec2).logits
    assert not torch.equal(o1, o2)


def test_train_mode_keeps_base_eval_and_module_state_roundtrip():
    gm, _, cfg = build()
    gm.train()
    assert not gm.base.training and gm.readout.training
    sd = gm.module_state_dict()
    gm2, _, _ = build(seed=1)
    gm2.load_module_state_dict(sd)
    for k, v in gm2.module_state_dict().items():
        assert torch.equal(v, sd[k])
    bad = {k: (torch.zeros(3, 3) if k == "readout.up.weight" else v) for k, v in sd.items()}
    try:
        gm2.load_module_state_dict(bad)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    run_all(globals())
