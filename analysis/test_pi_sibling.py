"""CPU mock test for pi_sibling_analysis on traces produced by the real eval sampler.

    python analysis/test_pi_sibling.py

Checks view construction (sibling view reveals exactly C_t minus i; PI view only touches
positions from the next block on, with int(n*ratio) of them, never C_t) and metric
sanity (ratio=0 -> g=0, JS_pi=0, cos=0; |C_t|=1 -> delta=0; JS >= 0; |cos| <= 1).
"""

import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import pi_sibling_analysis as A  # noqa: E402
import test_llada_trace as T  # noqa: E402
from llada_trace import LLaDATraceRecorder  # noqa: E402

A.MASK_ID = T.MASK
A.SPECIAL_IDS = (T.EOS,)


class ContextModel(torch.nn.Module):
    """Logits depend on the whole input (embedding mean + own token), so views differ."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.randn(T.V, 16, generator=g)
        self.out = torch.randn(16, T.V, generator=g)

    def forward(self, x):
        h = self.emb[x] + self.emb[x].mean(1, keepdim=True)
        logits = h @ self.out
        logits[..., T.MASK] = -1e4
        return type("O", (), {"logits": logits})()


def make_trace(ev, steps, L=32, block=8, P=5):
    prompt = torch.randint(10, T.V, (1, P))
    rec = LLaDATraceRecorder(T.MASK, P, topk=5)
    out, _, _ = ev.generate(T.FakeLLaDA(), prompt, None, steps=steps, gen_length=L, block_length=block,
                            temperature=0.0, mask_id=T.MASK, eos_token_id=T.EOS, recorder=rec)
    tr = rec.finalize(out)[0]
    tr["prompt_ids"] = prompt[0].int()
    tr["meta"] = {"is_correct": True, "block_length": block, "steps": steps, "temperature": 0.0}
    return tr


def main():
    os.environ.update(LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29535")
    dist.init_process_group("gloo", rank=0, world_size=1)
    torch.manual_seed(0)
    ev = T.load("eval_generate", os.path.join(ROOT, "eval", "generate.py"), stub_utils=True)
    model = ContextModel()

    # --- view construction ---
    tr = make_trace(ev, steps=16)
    S, final = tr["states"][3].long(), tr["states"][-1].long()
    c_t = tr["reveal"]["pos"][tr["reveal"]["step"] == 3].long()
    for i in c_t.tolist():
        v = A.sibling_view(S, final, c_t, i)
        changed = (v != S).nonzero().flatten()
        assert torch.equal(changed.sort().values, c_t[c_t != i].sort().values)
        assert v[i] == T.MASK
    gen = torch.Generator().manual_seed(1)
    blk = int(tr["block"][3])
    v = A.pi_view(S, final, blk, 8, 0.25, gen)
    changed = (v != S).nonzero().flatten()
    n_cand = 32 - (blk + 1) * 8
    assert (changed >= (blk + 1) * 8).all() and changed.numel() <= int(n_cand * 0.25)
    assert not torch.isin(changed, c_t).any()
    print(f"ok  views: sibling reveals C_t\\{{i}}; PI touches {changed.numel()} future slots (<= {int(n_cand * 0.25)})")

    # --- matched controls: current block, not in C_t, masked at S_t, revealed later ---
    t = 4  # first step of block 1 (the last step of a block has no candidates left by design)
    S = tr["states"][t].long()
    c_t = tr["reveal"]["pos"][tr["reveal"]["step"] == t].long()
    blk = int(tr["block"][t])
    bpos = torch.arange(blk * 8, (blk + 1) * 8)
    cands = bpos[(S[bpos] == T.MASK) & ~torch.isin(bpos, c_t)].tolist()
    conf = {j: -float(j % 5) for j in range(32)}
    for i in c_t.tolist():
        sibs = [k for k in c_t.tolist() if k != i]
        cpos, diag = A.match_controls(i, sibs, cands, conf, tr["reveal_step"], t, 8)
        assert cpos is not None and len(cpos) == len(sibs)
        for j in cpos:
            assert j in cands and j not in c_t.tolist() and S[j] == T.MASK and int(tr["reveal_step"][j]) > t
        assert diag[2] >= 1  # controls are revealed strictly later than the siblings
    assert A.match_controls(0, [1, 2], [5], conf, tr["reveal_step"], t, 8) == (None, None)
    print(f"ok  controls: {len(cands)} candidates at step {t}, matched ones are same-block future tokens")

    # --- metrics ---
    for steps, ratio in ((16, 0.25), (16, 0.0), (32, 0.25), (8, 0.25)):
        tr = make_trace(ev, steps=steps)
        tr["meta"]["teacher_retain_ratio"] = ratio
        an = A.Analyzer(model, False, "fixed", pi_samples=2, topk=5, batch_size=3, steps_per_chunk=3, device="cpu")
        r = an.analyze(tr, seed=0)
        npar = 32 // steps
        n_steps = steps - steps // 4  # last of 4 blocks excluded
        assert r["g"].numel() == n_steps * npar, (r["g"].numel(), n_steps, npar)
        assert (r["n_parallel"] == npar).all() and (r["block"] < 3).all()
        assert (r["js_sib"] >= -1e-6).all() and (r["js_pi"] >= -1e-6).all()
        assert (r["cos_pi_sib"].abs() <= 1 + 1e-5).all()
        assert torch.allclose(r["g"], r["logp_T"] - r["logp_S"])
        ok = r["ctrl_ok"]
        assert torch.allclose(r["coord"], r["D_S"] - r["D_T"])
        assert torch.equal(r["D_S"], r["delta"])
        multi = ok & (r["n_parallel"] > 1)
        assert (r["ctrl_delay"][multi] >= 1).all() and torch.isnan(r["D_S_ctrl"][~ok]).all()
        if ratio == 0.0:  # no PI: teacher view == student view
            assert r["g"].abs().max() < 1e-5 and r["js_pi"].abs().max() < 1e-6 and (r["cos_pi_sib"] == 0).all()
            assert r["coord"].abs().max() < 1e-4 and r["coord_ctrl"][ok].abs().max() < 1e-4
        if npar == 1:  # no siblings: nothing to add, control is empty
            assert r["delta"].abs().max() < 1e-5 and r["js_sib"].abs().max() < 1e-6
            assert r["D_T"].abs().max() < 1e-5 and ok.all() and r["D_S_ctrl"].abs().max() < 1e-5
        else:
            assert r["delta"].abs().max() > 0  # context model: siblings do shift p
            assert r["D_S_ctrl"][ok].abs().max() > 0
        assert r["topk_S_ids"].shape == (r["g"].numel(), 5)
        # reproducible PI sampling
        assert torch.equal(an.analyze(tr, seed=0)["g"], r["g"])
        print(f"ok  steps={steps} |C_t|={npar} ratio={ratio}: tokens={r['g'].numel()}  ctrl_ok={ok.float().mean():.2f}  "
              f"g={r['g'].mean():+.3f}  D_S={r['D_S'].mean():+.3f}  D_T={r['D_T'].mean():+.3f}  "
              f"coord={r['coord'].mean():+.3f}  coord_ctrl={r['coord_ctrl'][ok].mean():+.3f}")
    dist.destroy_process_group()
    print("all passed")


if __name__ == "__main__":
    main()
