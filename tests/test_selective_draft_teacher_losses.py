"""Teacher input isolation, mixtures by enumeration, KL, oracle patch."""
import torch
import torch.nn.functional as F

from _fake_helpers import EOS, MASK, V, build, fake_config, make_ctx, run_all
from selective_draft.config import TeacherConfig
from selective_draft.losses import forward_kl, kl_between_logprobs
from selective_draft.teacher import (candidate_set, log_probs_at, oracle_patch_log_probs, reveal_view,
                                     teacher_targets, unconditional_log_probs)


def _setup():
    gm, tok, cfg = build()
    torch.manual_seed(1)
    P, G = 5, 8
    x = torch.randint(4, V, (P + G,)); x[P:] = MASK; x[P + 2] = 11
    attn = torch.ones_like(x)
    A = P + 4
    recv = torch.tensor([p for p in range(P, P + G) if p != A and x[p] == MASK])
    r_A = F.softmax(torch.randn(V) * 2, -1)
    return gm, x, attn, P, A, recv, r_A


def test_reveal_only_changes_A_and_teacher_sees_exactly_that():
    gm, x, attn, P, A, recv, r_A = _setup()
    calls = []
    orig = gm.backend.forward_logits
    gm.backend.forward_logits = lambda ids, am=None: (calls.append((ids.clone(), am.clone())), orig(ids, am))[1]
    tcfg = TeacherConfig(mode="realized")
    tgt = teacher_targets(gm, x, attn, A, recv, r_A, realized_token=17, cfg=tcfg)
    assert len(calls) == 1 and tgt.info["teacher_forwards"] == 1
    ids, am = calls[0]
    diff = (ids[0] != x).nonzero().flatten().tolist()
    assert diff == [A] and int(ids[0, A]) == 17 and torch.equal(am[0], attn)
    assert x[A] == MASK, "original state must not be modified in place"
    assert tgt.log_q.shape == (len(recv), V) and torch.allclose(tgt.log_q.exp().sum(-1), torch.ones(len(recv)), atol=1e-5)


def test_mixtures_match_enumeration_and_mix_probabilities():
    gm, x, attn, P, A, recv, r_A = _setup()
    tcfg = TeacherConfig(mode="topk", topk=V, mc_samples=6)
    # top-K with K = V is the exact full mixture
    full = teacher_targets(gm, x, attn, A, recv, r_A, None, tcfg, mode="topk")
    brute = torch.zeros(len(recv), V)
    for v in range(V):
        lq = log_probs_at(gm, reveal_view(x, A, v)[None], attn[None], recv)[0]
        brute += float(r_A[v]) * lq.exp()
    assert torch.allclose(full.log_q.exp(), brute, atol=1e-5)
    assert abs(full.info["retained_mass"] - 1.0) < 1e-5
    # top-3: weights renormalised by retained mass
    t3 = teacher_targets(gm, x, attn, A, recv, r_A, None, TeacherConfig(mode="topk", topk=3), mode="topk")
    p, idx = r_A.topk(3)
    ref = sum(float(p[i] / p.sum()) * log_probs_at(gm, reveal_view(x, A, int(idx[i]))[None], attn[None], recv)[0].exp() for i in range(3))
    assert torch.allclose(t3.log_q.exp(), ref, atol=1e-5) and abs(t3.info["retained_mass"] - float(p.sum())) < 1e-6
    # MC: duplicate candidates merge into frequency weights, one forward per unique candidate
    g = torch.Generator().manual_seed(3)
    cands, w, info = candidate_set("mc", r_A, None, TeacherConfig(mode="mc", mc_samples=6), g)
    assert abs(float(w.sum()) - 1.0) < 1e-6 and len(cands) == info["unique_candidates"] and all(float(x_) * 6 % 1 < 1e-6 for x_ in w)
    # probability mixing, not logit averaging: mixture of two one-hot-ish conditionals has two modes
    la = torch.log(torch.tensor([[0.98, 0.01, 0.01]])); lb = torch.log(torch.tensor([[0.01, 0.98, 0.01]]))
    mix = torch.logsumexp(torch.stack([la, lb]) + torch.log(torch.tensor([0.5, 0.5])).view(2, 1, 1), 0).exp()
    assert torch.allclose(mix, torch.tensor([[0.495, 0.495, 0.01]]), atol=1e-6)
    # MASK candidate = no reveal -> identical to unconditional
    tm = teacher_targets(gm, x, attn, A, recv, None, realized_token=MASK, cfg=TeacherConfig(mode="realized"))
    assert tm.info["mask_candidate_events"] == 1
    assert torch.allclose(tm.log_q, unconditional_log_probs(gm, x, attn, recv), atol=1e-6)


def test_kl_properties_and_gradient():
    log_q = F.log_softmax(torch.randn(4, V), -1)
    assert torch.allclose(forward_kl(log_q, log_q.clone()), torch.zeros(4), atol=1e-5)
    logits = (torch.randn(4, V)).requires_grad_(True)
    kl = forward_kl(log_q, logits)
    manual = (log_q.exp() * (log_q - F.log_softmax(logits, -1))).sum(-1)
    assert torch.allclose(kl, manual) and (kl > -1e-6).all()
    kl.mean().backward()
    # d KL / d logits = (p - q) / n
    assert torch.allclose(logits.grad, (F.softmax(logits, -1) - log_q.exp()) / 4, atol=1e-6)
    assert torch.allclose(kl_between_logprobs(log_q, log_q), torch.zeros(4), atol=1e-6)


def test_oracle_patch_with_no_reveal_is_identity_and_with_reveal_moves():
    gm, x, attn, P, A, recv, r_A = _setup()
    ctx = make_ctx(x[None], P, P + 8)
    lp_mask, delta = oracle_patch_log_probs(gm, x, attn, ctx, A, MASK, recv)
    assert float(delta.abs().max()) == 0.0
    assert torch.allclose(lp_mask, unconditional_log_probs(gm, x, attn, recv), atol=1e-5)
    lp_rev, delta = oracle_patch_log_probs(gm, x, attn, ctx, A, 17, recv)
    assert float(delta.abs().max()) > 0 and not torch.allclose(lp_rev, lp_mask)
    # the spec path must not leave hooks or scratch behind
    assert gm._scratch is None and len(gm.backend.boundary_module(gm.cfg.early_layer)._forward_hooks) == 0


if __name__ == "__main__":
    run_all(globals())
