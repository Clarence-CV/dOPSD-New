"""PI-gain and same-step sibling-dependency analysis over LLaDA rollout traces.

Reads trace files written by `eval.py --trace_dir` (or trainer `traces/`), which hold
every state S_t, the tokens revealed at each step (C_t), the final rollout y and its
correctness. For every step t (outside the last block) and every i in C_t it runs
analysis-only forwards on three views of the same state:

  student  S_t                                   -> p_S(x_i | S_t)
  teacher  S_t + PI (d-OPSD construction: a random
           teacher_retain_ratio of the positions from
           the NEXT block to the end, filled with y)  -> p_T(x_i | S_t, PI)
  sibling  S_t + y at C_t \\ {i}, i still masked      -> p_sib(x_i | S_t, Y_{C_t\\i})

and records per token i:
  log p_S(y_i), log p_T(y_i), log p_sib(y_i)
  g_i = log p_T(y_i) - log p_S(y_i)            (PI gain)
  delta_i = log p_sib(y_i) - log p_S(y_i)      (sibling gain)
  JS(p_S, p_sib), JS(p_S, p_T)                 (full vocabulary)
  cos(p_T - p_S, p_sib - p_S)                  (PI shift vs sibling shift, full vocabulary)
  entropy / top-1 / top-k of p_S, p_T, p_sib
  context: step, block, step-in-block, |C_t|, sibling distance, special/after-answer flags.

PI never covers C_t itself (C_t lies in the current block, PI starts at the next one),
so g_i measures help from the future, not answer leakage. Teacher weights follow
training's fixed_teacher: with --adapter, `--teacher fixed` (default) runs the teacher
view with the LoRA disabled (= base model); `--teacher self` keeps the adapter.

    python analysis/pi_sibling_analysis.py --trace_dir T --out_dir O [--adapter A] \\
        [--pi_samples K] [--shard_id k --num_shards n]
Outputs one records file per rollout in O/records/ (skips existing -> resumable).
"""

import argparse
import glob
import os
import zlib

import torch
import torch.nn.functional as F

MASK_ID = 126336
SPECIAL_IDS = (126081, 126348)  # <|endoftext|>, <|eot_id|>


def js_divergence(logp, logq):
    """Jensen-Shannon divergence (nats) between rows of log-prob tensors [N, V]."""
    m = torch.logsumexp(torch.stack([logp, logq]), dim=0) - torch.log(torch.tensor(2.0, device=logp.device))
    kl_pm = (logp.exp() * (logp - m)).sum(-1)
    kl_qm = (logq.exp() * (logq - m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def cosine(a, b, zero=1e-6):
    """Row cosine; 0 where either shift is numerically nil (||.||_2 < zero, e.g. no PI change)."""
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb).clamp_min(1e-12)
    return torch.where((na < zero) | (nb < zero), torch.zeros_like(c), c)


def pi_view(state, final, block, block_length, ratio, gen):
    """d-OPSD teacher view: fill a random `ratio` of positions from the next block on with y.

    Mirrors dOPSDTrainer._generate_and_score_completions (candidates run to the end of the
    window, num_replace = int(num_candidates * ratio)).
    """
    view = state.clone()
    start = (block + 1) * block_length
    cand = torch.arange(start, state.numel())
    n = int(cand.numel() * ratio)
    if n > 0:
        sel = cand[torch.randperm(cand.numel(), generator=gen)[:n]]
        view[sel] = final[sel]
    return view


def sibling_view(state, final, c_t, i):
    """S_t with every other token of C_t revealed to its actual value; i stays masked."""
    view = state.clone()
    others = c_t[c_t != i]
    view[others] = final[others]
    return view


class Analyzer:
    def __init__(self, model, has_adapter, teacher, pi_samples, topk, batch_size, steps_per_chunk, device):
        self.model = model
        self.has_adapter = has_adapter
        self.teacher = teacher
        self.pi_samples = pi_samples
        self.topk = topk
        self.bs = batch_size
        self.steps_per_chunk = steps_per_chunk
        self.device = device

    @torch.no_grad()
    def _logprobs(self, prompt, items, teacher_weights):
        """items: [(view [L], [pos, ...])] -> log-softmax rows [sum(len(pos)), V] float32, in order."""
        out = []
        for s in range(0, len(items), self.bs):
            chunk = items[s:s + self.bs]
            ids = torch.stack([torch.cat([prompt, v]) for v, _ in chunk]).to(self.device)
            rows = torch.tensor([b for b, (_, ps) in enumerate(chunk) for _ in ps], device=self.device)
            cols = torch.tensor([p for _, ps in chunk for p in ps], device=self.device) + prompt.numel()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if teacher_weights and self.has_adapter and self.teacher == "fixed":
                    with self.model.disable_adapter():
                        logits = self.model(ids).logits
                else:
                    logits = self.model(ids).logits
            out.append(F.log_softmax(logits[rows, cols].float(), dim=-1))
            del logits
        return torch.cat(out)

    def _topk(self, logp):
        p, i = logp.exp().topk(self.topk, dim=-1)
        return i.int().cpu(), p.half().cpu()

    def analyze(self, tr, seed):
        meta = tr["meta"]
        prompt = tr["prompt_ids"].long()
        states = tr["states"].long()
        final = states[-1]
        L = final.numel()
        bl = int(meta["block_length"])
        num_blocks = L // bl
        ratio = float(meta.get("teacher_retain_ratio", 0.25))
        rv_step, rv_pos = tr["reveal"]["step"].long(), tr["reveal"]["pos"].long()
        blocks = tr["block"].long()
        special = torch.isin(final, torch.tensor(SPECIAL_IDS))
        answer_end = int(special.nonzero()[0]) if special.any() else L
        gen = torch.Generator().manual_seed(seed)
        first_step_of_block = {}
        for t in range(len(blocks)):
            first_step_of_block.setdefault(int(blocks[t]), t)

        # Steps with a future block (PI is empty in the last block, as in training).
        work = []
        for t in range(len(blocks)):
            c_t = rv_pos[rv_step == t].sort().values
            if int(blocks[t]) < num_blocks - 1 and c_t.numel() > 0:
                work.append((t, c_t))

        rec, K = {}, self.pi_samples
        add = lambda k, v: rec.setdefault(k, []).append(v)
        for c0 in range(0, len(work), self.steps_per_chunk):
            chunk = work[c0:c0 + self.steps_per_chunk]
            s_items, t_items = [], []
            for t, c_t in chunk:
                S, blk, ps = states[t], int(blocks[t]), c_t.tolist()
                s_items.append((S, ps))                                        # student: 1 forward, n rows
                s_items += [(sibling_view(S, final, c_t, i), [i]) for i in ps]  # sibling: 1 forward per i
                t_items += [(pi_view(S, final, blk, bl, ratio, gen), ps) for _ in range(K)]
            s_lp = self._logprobs(prompt, s_items, teacher_weights=False)
            t_lp = self._logprobs(prompt, t_items, teacher_weights=True)

            so = to = 0
            for t, c_t in chunk:
                n, blk = c_t.numel(), int(blocks[t])
                lp_S, lp_sib = s_lp[so:so + n], s_lp[so + n:so + 2 * n]
                so += 2 * n
                tk_ = t_lp[to:to + K * n].view(K, n, -1)
                to += K * n
                lp_T = torch.logsumexp(tk_, dim=0) - torch.log(torch.tensor(float(K), device=tk_.device))

                y = final[c_t].to(lp_S.device)
                gy = lambda lp: lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                ent = lambda lp: -(lp.exp() * lp).sum(-1)
                pS, pT, pSib = lp_S.exp(), lp_T.exp(), lp_sib.exp()
                d = (c_t.unsqueeze(0) - c_t.unsqueeze(1)).abs().float()
                d.fill_diagonal_(float("inf"))

                add("step", torch.full((n,), t))
                add("block", torch.full((n,), blk))
                add("step_in_block", torch.full((n,), t - first_step_of_block[blk]))
                add("pos", c_t)
                add("token", final[c_t])
                add("n_parallel", torch.full((n,), n))
                add("sib_dist", d.min(1).values if n > 1 else torch.full((1,), float("nan")))
                add("is_special", special[c_t])
                add("after_answer", c_t > answer_end)
                for name, val in (
                    ("logp_S", gy(lp_S)), ("logp_T", gy(lp_T)), ("logp_sib", gy(lp_sib)),
                    ("g", gy(lp_T) - gy(lp_S)), ("delta", gy(lp_sib) - gy(lp_S)),
                    ("js_sib", js_divergence(lp_S, lp_sib)), ("js_pi", js_divergence(lp_S, lp_T)),
                    ("cos_pi_sib", cosine(pT - pS, pSib - pS)),
                    ("ent_S", ent(lp_S)), ("ent_T", ent(lp_T)), ("ent_sib", ent(lp_sib)),
                    ("top1_S", lp_S.argmax(-1)), ("top1_T", lp_T.argmax(-1)), ("top1_sib", lp_sib.argmax(-1)),
                ):
                    add(name, val.cpu())
                for tag, lp in (("S", lp_S), ("T", lp_T), ("sib", lp_sib)):
                    ids, p = self._topk(lp)
                    add(f"topk_{tag}_ids", ids)
                    add(f"topk_{tag}_p", p)

        out = {k: torch.cat(v) for k, v in rec.items()}
        ok = meta.get("is_correct")
        out["meta"] = {
            "correct": bool(ok >= 1.0) if isinstance(ok, float) else bool(ok),
            "steps": int(meta.get("steps", len(blocks))), "gen_length": L, "block_length": bl,
            "temperature": meta.get("temperature"), "teacher_retain_ratio": ratio,
            "pi_samples": K, "teacher": self.teacher if self.has_adapter else "same-as-student",
            "answer_end": answer_end, "source": meta,
        }
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA dir (student = base+LoRA).")
    ap.add_argument("--teacher", default="fixed", choices=["fixed", "self"])
    ap.add_argument("--pi_samples", type=int, default=1, help="PI views per step; p_T is their mixture.")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8, help="Sequences per forward.")
    ap.add_argument("--steps_per_chunk", type=int, default=8, help="Denoising steps whose views are batched together.")
    ap.add_argument("--max_rollouts", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModel

    files = sorted(glob.glob(os.path.join(args.trace_dir, "*.pt")))
    if args.max_rollouts:
        files = files[:args.max_rollouts]
    files = files[args.shard_id::args.num_shards]
    rec_dir = os.path.join(args.out_dir, "records")
    os.makedirs(rec_dir, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(rec_dir, os.path.basename(f)))]
    print(f"[pi_sib] shard {args.shard_id}/{args.num_shards}: {len(todo)}/{len(files)} rollouts to analyze")
    if not todo:
        return

    device = "cuda"
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    an = Analyzer(model, bool(args.adapter), args.teacher, args.pi_samples, args.topk, args.batch_size,
                  args.steps_per_chunk, device)

    for k, f in enumerate(todo):
        tr = torch.load(f, weights_only=False)
        if "prompt_ids" not in tr:
            raise SystemExit(f"{f} has no prompt_ids; regenerate traces with the current eval.py")
        # Per-rollout PI seed: reproducible across runs/processes (unlike hash()).
        out = an.analyze(tr, seed=args.seed * 1_000_003 + zlib.crc32(os.path.basename(f).encode()))
        out["meta"]["adapter"] = args.adapter
        torch.save(out, os.path.join(rec_dir, os.path.basename(f)))
        print(f"[pi_sib] {k + 1}/{len(todo)} {os.path.basename(f)}  tokens={out['g'].numel()}  "
              f"correct={out['meta']['correct']}", flush=True)
    print(f"[pi_sib] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (batch_size={args.batch_size})")


if __name__ == "__main__":
    main()
