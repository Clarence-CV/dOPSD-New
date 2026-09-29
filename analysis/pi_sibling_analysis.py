"""PI gain, intra-step coordination gap, and matched-control analysis over LLaDA traces.

Question: does the successful self-future (d-OPSD's privileged information, PI) carry
information that coordinates tokens committed in the SAME denoising step?

Reads trace files written by `eval.py --trace_dir` (or trainer `traces/`): every state
S_t, the tokens revealed at each step (C_t), the final rollout y and its correctness.
For every step t outside the last block and every i in C_t it forwards

  student  S_t                        -> p_S(.)            teacher  S_t+PI            -> p_T(.)
  student  S_t + y[C_t\\i]  (siblings) -> p_S(.|sib)        teacher  S_t+PI+y[C_t\\i]  -> p_T(.|sib,PI)
  student  S_t + y[M_i]     (control)  -> p_S(.|ctrl)       teacher  S_t+PI+y[M_i]     -> p_T(.|ctrl,PI)

PI = d-OPSD construction: a random teacher_retain_ratio of positions from the NEXT block
to the end, filled with y. PI never touches the current block, so it never covers C_t
or the controls. Teacher weights follow training's fixed_teacher: with --adapter,
`--teacher fixed` (default) disables the LoRA for teacher views (= base model).

Matched control M_i: for each sibling k of i, one position j in the CURRENT block (so PI
can never pre-fill it), not in C_t, still masked at S_t (it is revealed later), chosen to
match |j-i| ~ |k-i|, student confidence log p_S(y_j) ~ log p_S(y_k), and earliest reveal.
Missing when the block has too few candidates (ctrl_ok=False).

Per token i (y_i = its value in the final rollout):
  g        = log p_T(y_i) - log p_S(y_i)                       PI gain
  D_S      = log p_S(y_i|sib) - log p_S(y_i)                   what siblings add for the student
  D_T      = log p_T(y_i|sib,PI) - log p_T(y_i|PI)             what siblings still add once PI is known
  coord    = D_S - D_T                                         coordination gap PI already fills
  D_S_ctrl, D_T_ctrl, coord_ctrl                               same with the matched control
  JS(p_S,p_sib), JS(p_S,p_ctrl), JS(p_S,p_T); cos(p_T-p_S, p_sib-p_S), cos(p_T-p_S, p_ctrl-p_S)
  entropy / top-1 / top-k of p_S, p_T, p_sib; step, block, |C_t|, sibling distance,
  control match quality (distance gap, confidence gap, reveal delay), special/after-answer flags.
  Decoding order (over ALL masked positions of the current block at S_t, confidence = top-1 prob):
  rank_S / rank_T of i (1 = most confident), rank_T_norm in [0,1], n_block_masked, and per step
  order_overlap = |C_t & teacher's top-|C_t|| / |C_t| and order_spearman(conf_S, conf_T).
CoGain_t (fraction of C_t with g>0) is formed at plotting time from `step` + `g`.

    python analysis/pi_sibling_analysis.py --trace_dir T --out_dir O [--adapter A] \\
        [--pi_samples K] [--shard_id k --num_shards n]
Writes one records file per rollout to O/records/ (existing ones are skipped -> resumable).
"""

import argparse
import glob
import os
import zlib

import torch
import torch.nn.functional as F

MASK_ID = 126336
SPECIAL_IDS = (126081, 126348)  # <|endoftext|>, <|eot_id|>
CONF_FLOOR = -20.0  # clamp log-probs used for control matching


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


def reveal(view, final, positions):
    """Copy of `view` with `positions` set to their final-rollout values."""
    out = view.clone()
    if len(positions):
        idx = torch.as_tensor(positions, dtype=torch.long)
        out[idx] = final[idx]
    return out


def sibling_view(state, final, c_t, i):
    """S_t with every other token of C_t revealed to its actual value; i stays masked."""
    return reveal(state, final, c_t[c_t != i].tolist())


def match_controls(i, siblings, cands, conf, reveal_step, t, block_length):
    """One matched non-sibling future position per sibling of i (greedy, without reuse).

    siblings: list of positions k; cands: candidate positions j (current block, masked, not
    in C_t); conf: dict pos -> clamped log p_S(y_pos) at S_t. Returns (positions, diagnostics)
    or (None, None) if there are not enough candidates.
    """
    avail = [j for j in cands if j != i]
    if len(avail) < len(siblings):
        return None, None
    chosen, dgap, cgap, delay = [], [], [], []
    for k in siblings:
        d_k = abs(k - i)
        best = min(avail, key=lambda j: abs(abs(j - i) - d_k) / block_length
                   + abs(conf[j] - conf[k]) + 0.05 * (int(reveal_step[j]) - t - 1))
        avail.remove(best)
        chosen.append(best)
        dgap.append(abs(abs(best - i) - d_k))
        cgap.append(abs(conf[best] - conf[k]))
        delay.append(int(reveal_step[best]) - t)
    n = len(siblings)
    nan = float("nan")  # no siblings -> no control needed, match quality undefined
    diag = (sum(dgap) / n, sum(cgap) / n, sum(delay) / n) if n else (nan, nan, nan)
    return chosen, diag


def order_metrics(conf_S, conf_T, n):
    """Decoding-order agreement at one step. conf_*: [M] top-1 probs over the block's masked
    positions, the first n being C_t. Returns ranks of C_t (1 = most confident), the teacher's
    top-n overlap with C_t, and the Spearman correlation of the two confidence vectors."""
    M = conf_S.numel()
    rank = lambda c: 1 + (c.unsqueeze(0) > c[:n].unsqueeze(1)).sum(1)
    rank_S, rank_T = rank(conf_S), rank(conf_T)
    overlap = float((conf_T.argsort(descending=True)[:n] < n).sum()) / n
    if M >= 3:
        rs = conf_S.argsort().argsort().float()
        rt = conf_T.argsort().argsort().float()
        rs, rt = rs - rs.mean(), rt - rt.mean()
        rho = float((rs * rt).sum() / (rs.norm() * rt.norm()).clamp_min(1e-12))
    else:
        rho = float("nan")
    return rank_S, rank_T, overlap, rho, M


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
        return torch.cat(out) if out else torch.empty(0, device=self.device)

    def _topk(self, logp):
        p, i = logp.exp().topk(self.topk, dim=-1)
        return i.int().cpu(), p.half().cpu()

    def _mix(self, lp):
        """[K, n, V] log-probs of K PI samples -> log of their mixture [n, V]."""
        return torch.logsumexp(lp, dim=0) - torch.log(torch.tensor(float(lp.shape[0]), device=lp.device))

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
        reveal_step = tr["reveal_step"].long()
        blocks = tr["block"].long()
        special = torch.isin(final, torch.tensor(SPECIAL_IDS))
        answer_end = int(special.nonzero()[0]) if special.any() else L
        gen = torch.Generator().manual_seed(seed)
        K = self.pi_samples
        first_step_of_block = {}
        for t in range(len(blocks)):
            first_step_of_block.setdefault(int(blocks[t]), t)

        # Steps with a future block (PI is empty in the last block, as in training).
        work = []
        for t in range(len(blocks)):
            c_t = rv_pos[rv_step == t].sort().values
            blk = int(blocks[t])
            if blk < num_blocks - 1 and c_t.numel() > 0:
                S = states[t]
                bpos = torch.arange(blk * bl, (blk + 1) * bl)
                cands = bpos[(S[bpos] == MASK_ID) & ~torch.isin(bpos, c_t)]
                work.append((t, blk, c_t, cands))

        rec = {}
        add = lambda k, v: rec.setdefault(k, []).append(v)
        for c0 in range(0, len(work), self.steps_per_chunk):
            chunk = work[c0:c0 + self.steps_per_chunk]

            # Phase 1: p_S at C_t and at control candidates; p_T at C_t for each PI sample.
            s1, t1, pis = [], [], {}
            for t, blk, c_t, cands in chunk:
                s1.append((states[t], c_t.tolist() + cands.tolist()))
                for k in range(K):
                    pis[t, k] = pi_view(states[t], final, blk, bl, ratio, gen)
                    t1.append((pis[t, k], c_t.tolist() + cands.tolist()))
            s1_lp = self._logprobs(prompt, s1, teacher_weights=False)
            t1_lp = self._logprobs(prompt, t1, teacher_weights=True)

            # Match controls, then Phase 2: sibling / control views under both weights.
            info, s2, t2 = [], [], []
            so = to = 0
            for t, blk, c_t, cands in chunk:
                n, m = c_t.numel(), cands.numel()
                rows = s1_lp[so:so + n + m]
                so += n + m
                lp_Tall = self._mix(t1_lp[to:to + K * (n + m)].view(K, n + m, -1))
                to += K * (n + m)
                lp_T = lp_Tall[:n]
                order = order_metrics(rows.max(-1).values.exp(), lp_Tall.max(-1).values.exp(), n)
                pos_all = c_t.tolist() + cands.tolist()
                y_all = final[torch.tensor(pos_all)].to(rows.device)
                conf_all = rows.gather(-1, y_all.unsqueeze(-1)).squeeze(-1).clamp_min(CONF_FLOOR).tolist()
                conf = dict(zip(pos_all, conf_all))
                ctrls = []
                for i in c_t.tolist():
                    sibs = [k for k in c_t.tolist() if k != i]
                    ctrls.append(match_controls(i, sibs, cands.tolist(), conf, reveal_step, t, bl))
                info.append((t, blk, c_t, rows[:n], lp_T, ctrls, order))
                S = states[t]
                for i, (cpos, _) in zip(c_t.tolist(), ctrls):
                    s2.append((sibling_view(S, final, c_t, i), [i]))
                    if cpos is not None:
                        s2.append((reveal(S, final, cpos), [i]))
                    for k in range(K):
                        t2.append((sibling_view(pis[t, k], final, c_t, i), [i]))
                        if cpos is not None:
                            t2.append((reveal(pis[t, k], final, cpos), [i]))
            s2_lp = self._logprobs(prompt, s2, teacher_weights=False)
            t2_lp = self._logprobs(prompt, t2, teacher_weights=True)

            so = to = 0
            for t, blk, c_t, lp_S, lp_T, ctrls, order in info:
                n = c_t.numel()
                V = lp_S.shape[-1]
                nan_row = torch.full((V,), float("nan"), device=lp_S.device)
                lp_Ssib, lp_Sctrl, lp_Tsib, lp_Tctrl = [], [], [], []
                for cpos, _ in ctrls:
                    ok = cpos is not None
                    lp_Ssib.append(s2_lp[so]); so += 1
                    if ok:
                        lp_Sctrl.append(s2_lp[so]); so += 1
                    else:
                        lp_Sctrl.append(nan_row)
                    tsib, tctrl = [], []
                    for _ in range(K):
                        tsib.append(t2_lp[to]); to += 1
                        if ok:
                            tctrl.append(t2_lp[to]); to += 1
                    lp_Tsib.append(self._mix(torch.stack(tsib).unsqueeze(1))[0])
                    lp_Tctrl.append(self._mix(torch.stack(tctrl).unsqueeze(1))[0] if ok else nan_row)
                lp_Ssib, lp_Sctrl = torch.stack(lp_Ssib), torch.stack(lp_Sctrl)
                lp_Tsib, lp_Tctrl = torch.stack(lp_Tsib), torch.stack(lp_Tctrl)

                y = final[c_t].to(lp_S.device)
                gy = lambda lp: lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                ent = lambda lp: -(lp.exp() * lp).sum(-1)
                pS = lp_S.exp()
                d = (c_t.unsqueeze(0) - c_t.unsqueeze(1)).abs().float()
                d.fill_diagonal_(float("inf"))
                D_S, D_T = gy(lp_Ssib) - gy(lp_S), gy(lp_Tsib) - gy(lp_T)
                D_Sc, D_Tc = gy(lp_Sctrl) - gy(lp_S), gy(lp_Tctrl) - gy(lp_T)
                diag = torch.tensor([c[1] if c[1] is not None else (float("nan"),) * 3 for c in ctrls])

                add("step", torch.full((n,), t))
                add("block", torch.full((n,), blk))
                add("step_in_block", torch.full((n,), t - first_step_of_block[blk]))
                add("pos", c_t)
                add("token", final[c_t])
                add("n_parallel", torch.full((n,), n))
                add("sib_dist", d.min(1).values if n > 1 else torch.full((1,), float("nan")))
                add("is_special", special[c_t])
                add("after_answer", c_t > answer_end)
                add("ctrl_ok", torch.tensor([c[0] is not None for c in ctrls]))
                rank_S, rank_T, overlap, rho, M = order
                add("rank_S", rank_S.cpu())
                add("rank_T", rank_T.cpu())
                add("rank_T_norm", ((rank_T - 1).float() / max(M - 1, 1)).cpu())
                add("n_block_masked", torch.full((n,), M))
                add("order_overlap", torch.full((n,), overlap))
                add("order_spearman", torch.full((n,), rho))
                add("ctrl_dist_gap", diag[:, 0])
                add("ctrl_conf_gap", diag[:, 1])
                add("ctrl_delay", diag[:, 2])
                for name, val in (
                    ("logp_S", gy(lp_S)), ("logp_T", gy(lp_T)), ("logp_sib", gy(lp_Ssib)),
                    ("logp_Tsib", gy(lp_Tsib)), ("logp_Sctrl", gy(lp_Sctrl)), ("logp_Tctrl", gy(lp_Tctrl)),
                    ("g", gy(lp_T) - gy(lp_S)),
                    ("delta", D_S), ("D_S", D_S), ("D_T", D_T), ("coord", D_S - D_T),
                    ("D_S_ctrl", D_Sc), ("D_T_ctrl", D_Tc), ("coord_ctrl", D_Sc - D_Tc),
                    ("js_sib", js_divergence(lp_S, lp_Ssib)), ("js_ctrl", js_divergence(lp_S, lp_Sctrl)),
                    ("js_pi", js_divergence(lp_S, lp_T)),
                    ("cos_pi_sib", cosine(lp_T.exp() - pS, lp_Ssib.exp() - pS)),
                    ("cos_pi_ctrl", cosine(lp_T.exp() - pS, lp_Sctrl.exp() - pS)),
                    ("ent_S", ent(lp_S)), ("ent_T", ent(lp_T)), ("ent_sib", ent(lp_Ssib)),
                    ("top1_S", lp_S.argmax(-1)), ("top1_T", lp_T.argmax(-1)), ("top1_sib", lp_Ssib.argmax(-1)),
                ):
                    add(name, val.cpu())
                for tag, lp in (("S", lp_S), ("T", lp_T), ("sib", lp_Ssib)):
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
    ap.add_argument("--batch_size", type=int, default=16, help="Sequences per forward.")
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
