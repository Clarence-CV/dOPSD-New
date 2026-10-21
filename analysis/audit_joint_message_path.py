"""Audit the production group-soft hook using a supplied GuidedModel and exact state.

The CLI runs only a tiny synthetic CPU model; it never loads pretrained weights.
On a compute node, call audit_message_path(gm, ids, attention_mask, context)
after loading a real snapshot. This diagnoses a causal path, not task utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selective_draft.model import GuidanceSpec, GuidedModel, StepContext


def audit_message_path(gm, ids, attention_mask, context, peer_probs=None):
    """B=1, no-cache audit; keep state, group, and every non-peer message fixed.

    peer_probs optionally supplies a [vocab] counterfactual for group slot 0.
    By default use a different token's one-hot distribution (an OOD stress test).
    No parameters or .grad buffers are changed; cfg.scale is restored on exit.
    Layer hooks record only group positions, not the complete residual stream.
    """
    if ids.shape[0] != 1:
        raise ValueError("message audit requires a single exact state")
    early = gm.cfg.early_layer
    if early >= gm.backend.n_layers:
        raise ValueError("peer propagation requires at least one later layer")
    if gm.base.training:
        raise ValueError("run the audit with base.eval() and deterministic dropout")
    with torch.no_grad():
        gi = gm(ids, attention_mask, context, mode="group_readout").guidance
    positions = gi.group_pos[0]
    valid = positions >= 0
    if int(valid.sum()) < 2:
        raise ValueError("at least two group positions are needed")
    slots = valid.nonzero(as_tuple=True)[0]
    peer_slot, receiver_slot = int(slots[0]), int(slots[1])
    peer, receiver = int(positions[peer_slot]), int(positions[receiver_slot])
    watch = positions[valid]
    peer_watch, receiver_watch = 0, 1
    natural_probs = gi.group_probs.detach().clone()

    def spec(probs):
        return GuidanceSpec(gi.source_pos, gi.gate_open, probs[:, 0],
                            group_pos=gi.group_pos, group_probs=probs)

    changed_probs = natural_probs.clone()
    if peer_probs is None:
        banned = {gm.backend.mask_token_id, gm.backend.eos_token_id,
                  gm.backend.pad_token_id, int(natural_probs[0, peer_slot].argmax())}
        token = next(v for v in range(gm.backend.vocab_size) if v not in banned)
        peer_probs = torch.zeros_like(natural_probs[0, peer_slot])
        peer_probs[token] = 1
    else:
        peer_probs = peer_probs.to(natural_probs)
    if (peer_probs.shape != natural_probs[0, peer_slot].shape or
            not bool(torch.isfinite(peer_probs).all()) or bool((peer_probs < 0).any()) or
            not torch.isclose(peer_probs.sum(), peer_probs.new_tensor(1.0), atol=1e-5)):
        raise ValueError("counterfactual must be a normalized finite vocabulary distribution")
    changed_probs[0, peer_slot] = peer_probs

    def run(mode, probs):
        captured, handles = {}, []

        def pre_hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
            if h is None:
                raise RuntimeError("cannot locate next-block input hidden")
            if "next_input" in captured:
                raise RuntimeError("next block executed more than once")
            captured["next_input"] = h[0, watch].detach().float().cpu()

        def output_hook(layer):
            def hook(module, args, output):
                key = f"L{layer}_output"
                if key in captured:
                    raise RuntimeError(f"{key} executed more than once")
                captured[key] = gm.backend.block_hidden(output)[0, watch].detach().float().cpu()
            return hook

        try:
            handles.append(gm.backend.blocks[early].register_forward_pre_hook(pre_hook, with_kwargs=True))
            for index in range(early, gm.backend.n_layers):
                handles.append(gm.backend.blocks[index].register_forward_hook(output_hook(index + 1)))
            with torch.no_grad():
                out = gm(ids, attention_mask, context, mode=mode,
                         spec=spec(probs) if mode == "group_soft" else None)
            captured["logits"] = out.logits[0, watch].detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        if "next_input" not in captured:
            raise RuntimeError("next-block input was not observed")
        return captured

    off = run("off", natural_probs)
    natural = run("group_soft", natural_probs)
    repeat = run("group_soft", natural_probs)
    changed = run("group_soft", changed_probs)
    old_scale = gm.cfg.scale
    try:
        gm.cfg.scale = 0.0
        zero_scale = run("group_soft", natural_probs)
    finally:
        gm.cfg.scale = old_scale

    def max_delta(a, b, key, row=None):
        diff = a[key] - b[key]
        if row is not None:
            diff = diff[row]
        return float(diff.abs().max())

    # This backward checks dependence on probabilities, rather than merely on h.
    # autograd.grad leaves parameter .grad buffers unchanged. Probe a fixed logit.
    with torch.enable_grad():
        differentiable_probs = natural_probs.clone().requires_grad_(True)
        out = gm(ids, attention_mask, context, mode="group_soft", spec=spec(differentiable_probs))
        target = int(natural["logits"][receiver_watch].argmax())
        grad, = torch.autograd.grad(out.logits[0, receiver, target], differentiable_probs, allow_unused=True)
    peer_grad = None if grad is None else float(grad[0, peer_slot].float().norm())
    return {
        "model_class": type(gm.base).__name__, "early_layer": early,
        "peer_position": peer, "receiver_position": receiver, "group_positions": watch.tolist(),
        "receiver_message_unchanged": torch.equal(natural_probs[0, receiver_slot], changed_probs[0, receiver_slot]),
        "repeat_logits_max_abs": max_delta(natural, repeat, "logits"),
        "scale_zero_vs_off_logits_max_abs": max_delta(zero_scale, off, "logits"),
        "natural_vs_off_logits_max_abs": max_delta(natural, off, "logits"),
        "peer_edit_next_input_peer_max_abs": max_delta(natural, changed, "next_input", peer_watch),
        "peer_edit_next_input_receiver_max_abs": max_delta(natural, changed, "next_input", receiver_watch),
        "peer_edit_receiver_hidden_max_abs_by_layer": {
            key: max_delta(natural, changed, key, receiver_watch)
            for key in natural if key.startswith("L")
        },
        "peer_edit_receiver_logits_max_abs": max_delta(natural, changed, "logits", receiver_watch),
        "receiver_logit_gradient_wrt_peer_probs_l2": peer_grad,
        "gradient_probe_token": target,
    }


def fake_demo():
    from selective_draft.backends import resolve_backend
    from selective_draft.config import Config
    from selective_draft.testing import make_fake

    torch.set_num_threads(1)
    cfg = Config.load(str(ROOT / "configs/selective_draft/fake_cpu.json"))
    cfg.guidance.mode, cfg.guidance.group_size = "group_soft", 3
    cfg.guidance.early_layer = 2
    cfg.model.fake_kwargs["n_layers"] = 6  # Four later blocks, like L28 -> L32.
    base, _ = make_fake(cfg.model.fake_kwargs)
    torch.manual_seed(0)
    gm = GuidedModel(base, resolve_backend("fake_llada", base), cfg.guidance)
    gm.set_stage("joint")
    mask = gm.backend.mask_token_id
    ids = torch.tensor([[8, mask, mask, mask, mask]])
    context = StepContext(ids == mask, torch.tensor([True]), torch.tensor([1]),
                          torch.tensor([5]), torch.tensor([0]))
    attn = torch.ones_like(ids)
    zero = audit_message_path(gm, ids, attn, context)
    # Exercise the route without claiming to use learned/pretrained weights.
    with torch.no_grad():
        gm.injector.up.weight.normal_(std=0.1)
    nonzero = audit_message_path(gm, ids, attn, context)
    assert zero["natural_vs_off_logits_max_abs"] == 0
    assert zero["peer_edit_receiver_logits_max_abs"] == 0
    assert zero["receiver_logit_gradient_wrt_peer_probs_l2"] == 0
    assert nonzero["repeat_logits_max_abs"] == 0
    assert nonzero["scale_zero_vs_off_logits_max_abs"] == 0
    assert nonzero["peer_edit_next_input_receiver_max_abs"] == 0
    assert nonzero["peer_edit_next_input_peer_max_abs"] > 0
    assert nonzero["peer_edit_receiver_logits_max_abs"] > 0
    assert nonzero["receiver_logit_gradient_wrt_peer_probs_l2"] > 0
    return {
        "scope": "Synthetic CPU path audit through production GuidedModel; no pretrained weights or task evaluation.",
        "torch_version": torch.__version__, "seed": 0,
        "zero_initialized_injector": zero, "synthetic_nonzero_injector": nonzero,
        "limitation": "Nonzero effects show connectivity, not learned peer use, useful reasoning, or mini MoE gradient correctness.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = fake_demo()
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
