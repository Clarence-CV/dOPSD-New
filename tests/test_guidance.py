"""Guided forward: boundary, parity, hook lifecycle, gradient paths, soft message, module state, forward KL."""
import torch
import torch.nn.functional as F

from _helpers import MASK, V, build, mini_config, run_all
from selective_draft.losses import forward_kl
from selective_draft.model import GuidanceSpec, StepContext
from selective_draft.modules import soft_message


def _state(P=8, G=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(4, V, (1, P + G))
    x[:, P:] = MASK
    elig = torch.zeros_like(x, dtype=torch.bool)
    elig[:, P:] = True
    return x, elig, P, P + G


def _ctx(elig, bs, be, tau):
    return StepContext(elig, torch.ones(1, dtype=torch.bool), torch.tensor([bs]), torch.tensor([be]),
                       torch.zeros(1, dtype=torch.long), commit_threshold=tau)


def _model(**over):
    gm, _, cfg = build(mini_config(**over))
    return gm, cfg


def test_boundary_is_the_early_layer_output():
    gm, cfg = _model()
    x, elig, bs, be = _state()
    captured = {}
    handle = gm.backend.boundary_module(cfg.guidance.early_layer).register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", gm.backend.block_hidden(o).detach().clone()))
    try:
        out = gm(x, context=_ctx(elig, bs, be, 0.5), mode="off", capture_hidden=True)
    finally:
        handle.remove()
    assert torch.equal(out.hidden_early, captured["h"])
    assert gm.backend.boundary_module(cfg.guidance.early_layer) is gm.backend.blocks[cfg.guidance.early_layer - 1]


def test_parity_off_readout_zero_injector_scale0_and_closed_group():
    gm, cfg = _model()
    x, elig, bs, be = _state()
    ctx = _ctx(elig, bs, be, 0.0)  # tau 0: every non-MASK-argmax position joins the group
    ref = gm(x, mode="off").logits
    assert torch.equal(gm(x, context=ctx, mode="group_readout").logits, ref)
    out = gm(x, context=ctx, mode="group_soft")  # zero-initialised injector: identical even though a group formed
    assert out.guidance.gate_open.all() and torch.equal(out.logits, ref)
    torch.nn.init.normal_(gm.injector.up.weight, std=0.1)
    gm.cfg.scale = 0.0
    assert torch.equal(gm(x, context=ctx, mode="group_soft").logits, ref)  # non-zero injector but scale 0
    gm.cfg.scale = 1.0
    assert not torch.equal(gm(x, context=ctx, mode="group_soft").logits, ref)
    closed = gm(x, context=_ctx(elig, bs, be, 1.1), mode="group_soft")  # tau above any confidence: no message
    assert not closed.guidance.gate_open.any() and torch.equal(closed.logits, ref)
    assert closed.guidance.skip_reason == ["none_high"]


def test_hook_lifecycle_and_nesting():
    gm, cfg = _model()
    x, elig, bs, be = _state()
    mod = gm.backend.boundary_module(cfg.guidance.early_layer)
    gm(x, context=_ctx(elig, bs, be, 0.0), mode="group_readout")
    assert len(mod._forward_hooks) == 0 and gm._scratch is None
    bad = _ctx(elig, bs, be, 0.0)
    bad.eligible_mask = bad.eligible_mask[:, :3]  # wrong shape -> raises inside the hook
    try:
        gm(x, context=bad, mode="group_readout")
        assert False, "expected an error"
    except Exception:
        pass
    assert len(mod._forward_hooks) == 0 and gm._scratch is None  # the hook is removed after an exception
    assert torch.equal(gm(x, mode="off").logits, gm.base_forward(x))
    try:
        with gm._scoped_hook(type("S", (), {})()):
            gm.base_forward(x)  # a guided forward is in progress
        assert False, "base_forward inside a guided forward must raise"
    except RuntimeError:
        pass


def test_gradient_paths():
    gm, cfg = _model()
    x, elig, bs, be = _state()
    ctx = _ctx(elig, bs, be, 0.0)
    pos = torch.arange(bs, be)
    gm.set_stage("readout")  # stage 1: only the readout learns
    with torch.no_grad():
        out = gm(x, context=ctx, mode="off", capture_hidden=True)
    log_q = F.log_softmax(out.logits[0, pos].float(), -1)
    loss = (log_q.exp() * (log_q - gm.readout_log_probs(out.hidden_early[0, pos]))).sum(-1).mean()
    loss.backward()
    assert gm.readout.up.weight.grad is not None and gm.readout.up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in gm.injector.parameters()) and all(p.grad is None for p in gm.base.parameters())
    gm.zero_grad(set_to_none=True)
    gm.set_stage("joint")  # stage 2: only the injector learns, through the frozen tail and its routed experts
    out = gm(x, context=ctx, mode="group_soft")
    assert out.logits.requires_grad and out.guidance.gate_open.all()
    group = out.guidance.group_pos[0][out.guidance.group_pos[0] >= 0]
    target = F.softmax(torch.randn(len(group), V), -1)
    kl = (target * (target.log() - F.log_softmax(out.logits[0, group].float(), -1))).sum(-1).mean()
    kl.backward()
    assert gm.injector.up.weight.grad.abs().sum() > 0, "zero-initialised up projection must receive gradient"
    assert all(p.grad is None for p in gm.readout.parameters()) and all(p.grad is None for p in gm.base.parameters())
    gm.assert_base_frozen()


def test_soft_message_matches_enumeration_and_the_message_matters():
    E = torch.randn(V, 32)
    p = F.softmax(torch.randn(2, V), -1)
    ref = torch.stack([sum(p[b, v] * E[v] for v in range(V)) for b in range(2)])
    assert torch.allclose(soft_message(p, E, vocab_chunk=7), ref, atol=1e-5)
    assert torch.allclose(soft_message(p, E, vocab_chunk=V), ref, atol=1e-5)
    assert not torch.allclose(soft_message(F.softmax(torch.randn(2, V), -1), E), ref)
    gm, cfg = _model()
    torch.nn.init.normal_(gm.injector.up.weight, std=0.1)
    x, elig, bs, be = _state()
    ctx = _ctx(elig, bs, be, 0.0)
    pos = torch.tensor([[bs, bs + 1]])
    probs = lambda: F.softmax(torch.randn(1, 2, V), -1)
    o1 = gm(x, context=ctx, mode="group_soft", spec=GuidanceSpec(pos, probs())).logits
    o2 = gm(x, context=ctx, mode="group_soft", spec=GuidanceSpec(pos, probs())).logits
    assert not torch.equal(o1, o2)


def test_train_mode_keeps_base_eval_and_module_state_roundtrip():
    gm, cfg = _model()
    gm.train()
    assert gm.base.training is False and gm.training is True
    torch.nn.init.normal_(gm.injector.up.weight, std=0.1)
    torch.nn.init.normal_(gm.readout.up.weight, std=0.1)
    sd = {k: v.clone() for k, v in gm.module_state_dict().items()}
    gm2, _ = _model()
    gm2.load_module_state_dict(sd)
    assert all(torch.equal(v, gm2.module_state_dict()[k]) for k, v in sd.items())
    bad = dict(sd)
    bad["readout.down.weight"] = torch.zeros(3, 3)
    try:
        gm2.load_module_state_dict(bad)
        assert False, "shape mismatch must be rejected"
    except ValueError:
        pass


def test_forward_kl_properties():
    logits = torch.randn(5, V)
    log_q = F.log_softmax(torch.randn(5, V), -1)
    assert torch.allclose(forward_kl(F.log_softmax(logits, -1), logits), torch.zeros(5), atol=1e-6)
    kl = forward_kl(log_q, logits)
    assert bool((kl >= -1e-7).all())
    leaf = logits.clone().requires_grad_(True)
    forward_kl(log_q, leaf).sum().backward()
    assert leaf.grad is not None and leaf.grad.abs().sum() > 0
    try:
        forward_kl(log_q, logits[:, :-1])
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_all(globals())
