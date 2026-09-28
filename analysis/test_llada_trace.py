"""CPU mock test: LLaDATraceRecorder driven by the repo's REAL samplers.

    python analysis/test_llada_trace.py

Runs `eval/generate.py::generate` and `d-opsd/utils.py::generate` with a
random-logit fake model and checks the recorded trace against the sampler
output (S_t transitions, |C_t|, block locality, greedy consistency, and for the
training sampler, agreement with its returned trajectory).
"""

import importlib.util
import os
import sys
import types

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from llada_trace import LLaDATraceRecorder  # noqa: E402

MASK, EOS, V = 7, 3, 50


def load(name, path, stub_utils=False):
    if stub_utils:  # eval/utils.py pulls tiktoken; generate only needs main_print
        sys.modules["utils"] = types.SimpleNamespace(main_print=print)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLLaDA:
    dtype = torch.float32
    device = torch.device("cpu")

    def __call__(self, x):
        logits = torch.randn(x.shape[0], x.shape[1], V)
        logits[..., MASK] = -1e4
        logits[..., EOS] = -1e4  # keep the whole window "content" so every step is traced
        return types.SimpleNamespace(logits=logits)


def check(rows, final, P, L, block_len, steps, greedy, traj=None):
    tpb = L // steps  # tokens per step
    spb = block_len // tpb  # steps per block
    for b, tr in enumerate(rows):
        S = tr["states"]
        assert S.shape == (steps + 1, L), S.shape
        assert (S[0] == MASK).all() and not (S[-1] == MASK).any()
        assert torch.equal(S[-1], final[b, P:].int())
        assert (tr["n_revealed"] == tpb).all(), tr["n_revealed"]
        assert torch.equal(tr["block"].long(), torch.arange(steps) // spb)
        rv, mk = tr["reveal"], tr["masked"]
        for t in range(steps):
            changed = (S[t] != S[t + 1]).nonzero().flatten()
            sel_r = rv["step"] == t
            c_t = rv["pos"][sel_r].long()
            assert torch.equal(changed, c_t.sort().values), (t, changed, c_t)
            assert torch.equal(rv["token"][sel_r], S[t + 1, c_t])
            blk = int(tr["block"][t])
            assert ((c_t >= blk * block_len) & (c_t < (blk + 1) * block_len)).all()
            sel = mk["step"] == t
            assert int(sel.sum()) == int((S[t] == MASK).sum())
            assert int(mk["revealed"][sel].sum()) == len(c_t)
            if greedy:
                assert (rv["rank"][sel_r] == 0).all()
                pos = mk["pos"][sel].long()
                in_blk = (pos >= blk * block_len) & (pos < (blk + 1) * block_len)
                conf = mk["cand_prob"][sel][in_blk]
                thr = conf.topk(len(c_t)).values.min()
                assert (mk["cand_prob"][sel][mk["revealed"][sel]] >= thr - 1e-6).all()
        if traj is not None:  # training sampler: trajectory[k] must equal S_k
            for k, xk in enumerate(traj):
                assert torch.equal(xk[b, P:].int(), S[k]), k
    return torch.cat([tr["n_revealed"] for tr in rows])


def main():
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("gloo", rank=0, world_size=1)  # eval generate calls dist.get_rank()
    torch.manual_seed(0)
    model = FakeLLaDA()

    ev = load("eval_generate", os.path.join(ROOT, "eval", "generate.py"), stub_utils=True)
    tr_utils = load("dopsd_utils", os.path.join(ROOT, "d-opsd", "utils.py"))

    B, P, L, block_len, steps = 3, 5, 32, 8, 16  # 2 tokens / step, like the paper
    for temp in (0.0, 1.0):
        prompt = torch.randint(10, V, (B, P))
        rec = LLaDATraceRecorder(MASK, P, topk=5, chunk=7)
        out, _, _ = ev.generate(model, prompt, None, steps=steps, gen_length=L, block_length=block_len,
                                temperature=temp, mask_id=MASK, eos_token_id=EOS, recorder=rec)
        sizes = check(rec.finalize(out), out, P, L, block_len, steps, greedy=temp == 0.0)
        print(f"ok  eval/generate.py      temp={temp}  |C_t| counts={torch.bincount(sizes.long()).tolist()}")

        prompt1 = prompt[:1]  # training sampler runs batch size 1
        rec = LLaDATraceRecorder(MASK, P, topk=5, chunk=7)
        out, traj = tr_utils.generate(model, prompt1, steps=steps, gen_length=L, block_length=block_len,
                                      temperature=temp, mask_id=MASK, eos_token_id=EOS, recorder=rec)
        assert len(traj) == steps - steps // (L // block_len)  # last block never appended
        check(rec.finalize(out), out, P, L, block_len, steps, greedy=temp == 0.0, traj=traj)
        print(f"ok  d-opsd/utils.py       temp={temp}  trajectory len={len(traj)} matches S_t")

    # recorder=None must leave the samplers bit-identical
    for gen, kw in ((ev.generate, dict(tokenizer=None)), (tr_utils.generate, {})):
        outs = []
        for use in (False, True):
            torch.manual_seed(1)
            r = LLaDATraceRecorder(MASK, P) if use else None
            o = gen(model, prompt[:1], steps=steps, gen_length=L, block_length=block_len,
                    temperature=1.0, mask_id=MASK, eos_token_id=EOS, recorder=r, **kw)
            outs.append(o[0])
        assert torch.equal(outs[0], outs[1])
    print("ok  recorder does not change sampler output")
    dist.destroy_process_group()
    print("all passed")


if __name__ == "__main__":
    main()
