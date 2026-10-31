"""Peak-memory probe for one Joint training step at the worst-case state lengths (no data, no rollouts).

Memory depends on the sequence length N and the group size, not on the token values, so a synthetic state is enough:
ordinary random tokens, the last block MASK, a full-block group with random natural drafts. The step is the same
sequence as RootSuccessTrainer.state_loss: group_readout capture (no grad, block rows kept), one Teacher forward of the
leave-one-out certification (no grad), the Student group_soft forward with grad, forward KL, backward. An OOM is
reported per length instead of aborting the probe.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import Config
from .losses import forward_kl
from .model import GuidanceSpec, GuidedModel, StepContext
from .root_joint import certify_tuple

GIB = 2 ** 30


def probe(gm: GuidedModel, cfg: Config, lengths: List[int], group: int = 32, seed: int = 0) -> List[Dict[str, object]]:
    be, dev, V, BL = gm.backend, gm.device, gm.backend.vocab_size, gm.backend.block_length
    if dev.type != "cuda":
        raise RuntimeError("the memory probe needs a CUDA device")
    gm.set_stage("joint")
    g = torch.Generator().manual_seed(seed)
    results = []
    for n_req in lengths:
        n = n_req // BL * BL
        gm.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rec: Dict[str, object] = {"length": n, "group": min(group, BL), "weights_gib": round(torch.cuda.memory_allocated() / GIB, 2)}
        t0 = time.perf_counter()
        try:
            bs = n - BL
            x = torch.randint(1000, 100000, (1, n), generator=g).to(dev)  # ordinary ids, far below the special range
            x[0, bs:] = be.mask_token_id
            elig = torch.zeros(1, n, dtype=torch.bool, device=dev)
            elig[0, bs:] = True
            ctx = StepContext(eligible_mask=elig, active_rows=torch.ones(1, dtype=torch.bool, device=dev),
                              block_start=torch.tensor([bs], device=dev), block_end=torch.tensor([n], device=dev),
                              step_id=torch.tensor([0], device=dev), commit_threshold=cfg.generation.commit_threshold,
                              guidance_mask=elig, group_threshold=cfg.guidance.group_threshold)
            k = rec["group"]
            pos = torch.arange(bs, bs + k, device=dev)
            probs = torch.softmax(torch.randn(1, k, V, generator=g) * 3.0, -1).to(dev)
            spec = GuidanceSpec(pos[None], probs)
            with torch.no_grad():
                cap = gm(x, context=ctx, mode="group_readout")
                native = cap.logits[0, pos].clone()
                del cap
                tokens = torch.randint(1000, 100000, (k,), generator=g).to(dev)
                certify_tuple(gm, x, pos, tokens, cfg.training.loo_batch_size)  # random tokens: stops after 1 forward
            out = gm(x, context=ctx, mode="group_soft", spec=spec)
            log_q = F.log_softmax(torch.randn(k, V, generator=g).to(dev), -1)
            loss = forward_kl(log_q, out.logits[0, pos]).mean()
            loss.backward()
            grad = sum(float(p.grad.float().abs().sum()) for p in gm.injector.parameters() if p.grad is not None)
            rec.update(oom=False, finite=bool(torch.isfinite(loss)), injector_grad_abs_sum=grad,
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2))
            del out, loss, log_q, native
        except torch.cuda.OutOfMemoryError:
            rec.update(oom=True, peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GIB, 2),
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 2))
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        rec["gpu_total_gib"] = round(torch.cuda.get_device_properties(dev).total_memory / GIB, 2)
        results.append(rec)
        print(rec, flush=True)
    return results
