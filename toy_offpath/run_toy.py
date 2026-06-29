"""
Toy experiment: "the noise is off the decoding path".

Claim under test
----------------
A diffusion LM (dLLM) unmasks tokens in a confidence-ordered, easy->hard
schedule. So at any masked fraction p, the *on-policy* partial state has a
defining property: every already-decoded position was more confident than
every still-masked one. A uniformly-random mask of the finished rollout breaks
this -- it reveals "hard/late" tokens and hides "easy/early" ones, producing a
partial state the model never visits.

Clean control
-------------
In BOTH the on-policy and the uniform partial state, the *visible token values*
are exactly the final rollout tokens (we re-mask the finished answer). The ONLY
thing that differs is WHICH positions are masked at a given level p. So any
measured gap is attributable purely to the masking pattern.

What this script does
---------------------
1. Rolls out completions with the BASE Dream model using its own
   entropy (confidence) decoding, while LOGGING for every generated position:
       - commit_step : the decode step at which it was unmasked
       - commit_conf : the model's confidence (neg-entropy) at commit time
   This logged trajectory is the model's real decoding path.
2. For a sweep of masked fractions p, builds two partial states per rollout:
       - on-policy mask : the lowest-commit-confidence positions (decoded last)
       - uniform mask   : a random subset of the same size
   Visible values = final rollout tokens in both.
3. Computes three metrics (see README) and dumps everything for visualize.py.

Run with the BASE model (motivation must not use the trained checkpoint):
    python run_toy.py --pretrained Dream-org/Dream-v0-Instruct-7B
"""

import os
import json
import argparse

import numpy as np
import torch
import transformers

from prompts import PROMPTS


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def get_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def load_model(pretrained, dtype, device):
    model = transformers.AutoModel.from_pretrained(
        pretrained, torch_dtype=get_dtype(dtype), trust_remote_code=True,
    ).eval().to(device)
    tok = transformers.AutoTokenizer.from_pretrained(
        pretrained, trust_remote_code=True,
    )
    return model, tok


def get_mask_id(model, tok):
    for obj, attr in [(model.config, "mask_token_id"),
                      (getattr(model, "generation_config", None), "mask_token_id"),
                      (tok, "mask_token_id")]:
        if obj is not None:
            v = getattr(obj, attr, None)
            if v is not None:
                return int(v)
    raise ValueError("Could not resolve mask_token_id")


def build_input_ids(tok, question, device):
    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(device)


def load_questions(dataset, num_prompts, seed):
    """Return a reproducible list of question strings.

    dataset="gsm8k" -> a fixed-seed subsample of the gsm8k test split.
    dataset="none"  -> the built-in PROMPTS (offline fallback).
    """
    if dataset == "none":
        return PROMPTS[:num_prompts]
    try:
        from datasets import load_dataset
        if dataset == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            questions = ds["question"]
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        rng = np.random.default_rng(seed)
        idx = sorted(rng.permutation(len(questions))[:num_prompts].tolist())
        print(f"[data] sampled {len(idx)} questions from {dataset} (seed={seed})")
        return [questions[i] for i in idx]
    except Exception as e:
        print(f"[warn] could not load '{dataset}' ({e}); "
              f"falling back to {len(PROMPTS)} built-in prompts")
        return PROMPTS[:num_prompts]


# --------------------------------------------------------------------------- #
# generation with trajectory logging (entropy / confidence decoding)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_with_trajectory(model, x, prompt_len, mask_id, tokens_per_step):
    """Reimplements Dream's entropy decode loop, logging the unmask schedule.

    Returns:
        final     : [1, L] fully decoded sequence
        commit_step: np.int array  [G]  (G = max_new_tokens), decode step per gen position
        commit_conf: np.float array [G], neg-entropy confidence at commit time
                     (higher / closer to 0 == more confident == decoded earlier)
    """
    L = x.shape[1]
    G = L - prompt_len
    commit_step = np.full(G, -1, dtype=np.int64)
    commit_conf = np.full(G, np.nan, dtype=np.float32)

    step = 0
    while bool((x == mask_id).any()):
        mask_index = (x == mask_id)                       # [1, L]
        logits = model(input_ids=x).logits                # [1, L, V]
        # Dream's shifted-logits convention: logits at pos j predict token j.
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        logp = torch.log_softmax(logits.float(), dim=-1)
        probs = logp.exp()
        neg_ent = (probs * logp).sum(-1)                  # [1, L]  (<=0)
        x0 = logits.argmax(-1)                            # [1, L]  greedy tokens

        # only masked positions are eligible to be committed this step
        conf = torch.where(mask_index, neg_ent,
                           torch.full_like(neg_ent, -1e30))
        n_mask = int(mask_index.sum().item())
        k = min(tokens_per_step, n_mask)
        top_pos = torch.topk(conf[0], k).indices          # highest-confidence masked positions

        for pos in top_pos.tolist():
            x[0, pos] = x0[0, pos]
            gi = pos - prompt_len                         # index into gen region
            commit_step[gi] = step
            commit_conf[gi] = float(neg_ent[0, pos].item())
        step += 1

    return x, commit_step, commit_conf


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_partial_states(model, partials, mask_id):
    """partials: [M, L] long tensor. Returns per-position log-probs after shift."""
    logits = model(input_ids=partials).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return torch.log_softmax(logits.float(), dim=-1)      # [M, L, V]


def compute_for_trajectory(model, traj, p_levels, num_uniform, rng, mask_id,
                           m2_p=0.5):
    """Build on-policy & uniform partial states, run forward, collect metrics.

    Returns a dict of raw per-(p) accumulators for this trajectory.
    """
    device = next(model.parameters()).device
    final = traj["final"].to(device)                      # [1, L]
    prompt_len = traj["prompt_len"]
    commit_conf = traj["commit_conf"]                     # [G]
    content_len = traj["content_len"]

    # maskable region = generated content positions (absolute indices into L)
    region = np.arange(prompt_len, prompt_len + content_len)
    cc = commit_conf[:content_len]                        # commit conf over content
    # order content positions by commit confidence ASCENDING:
    #   lowest confidence == decoded last == masked first under on-policy.
    order_by_conf = np.argsort(cc)                        # indices into region

    # ---- assemble all partial states for this trajectory in one batch ----
    items = []          # (p, scheme, draw, masked_abs_positions)
    for p in p_levels:
        k = int(round(p * content_len))
        if k == 0 or k >= content_len:
            continue
        onpolicy_local = order_by_conf[:k]                # lowest-conf k -> masked
        items.append((p, "onpolicy", 0, region[onpolicy_local]))
        for d in range(num_uniform):
            unif_local = rng.choice(content_len, size=k, replace=False)
            items.append((p, "uniform", d, region[unif_local]))

    if not items:
        return None

    L = final.shape[1]
    batch = final.repeat(len(items), 1).clone()           # [M, L]
    for m, (_, _, _, masked_abs) in enumerate(items):
        batch[m, masked_abs] = mask_id

    # chunk the forward to bound memory
    logp_chunks = []
    CH = 24
    for s in range(0, batch.shape[0], CH):
        logp_chunks.append(forward_partial_states(model, batch[s:s + CH], mask_id).cpu())
    logp = torch.cat(logp_chunks, dim=0)                  # [M, L, V]

    final_cpu = final[0].cpu()

    # per-item: NLL of the true final token and max-prob confidence at masked pos
    per_item = []
    for m, (p, scheme, draw, masked_abs) in enumerate(items):
        pos = torch.as_tensor(masked_abs, dtype=torch.long)
        lp = logp[m, pos]                                 # [k, V]
        gold = final_cpu[pos]                             # [k]
        nll = -lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)        # [k]
        maxprob = lp.max(-1).values.exp()                          # [k]
        per_item.append(dict(p=p, scheme=scheme, draw=draw,
                             masked_local=(masked_abs - prompt_len),
                             nll=nll.numpy(), maxprob=maxprob.numpy()))

    # ---------- M1: matched-position delta-NLL (uniform - onpolicy) ----------
    # For positions masked under BOTH schemes, only the visible context differs.
    delta_by_p = {}
    for p in p_levels:
        op = next((it for it in per_item if it["p"] == p and it["scheme"] == "onpolicy"), None)
        if op is None:
            continue
        op_nll = {loc: v for loc, v in zip(op["masked_local"], op["nll"])}
        diffs = []
        for it in per_item:
            if it["p"] != p or it["scheme"] != "uniform":
                continue
            for loc, v in zip(it["masked_local"], it["nll"]):
                if loc in op_nll:                          # shared masked position
                    diffs.append(v - op_nll[loc])
        if diffs:
            delta_by_p[p] = diffs

    # ---------- M3: off-path fraction (uses only logged commit conf) ----------
    # frontier_p = highest commit-conf among on-policy-masked positions.
    # uniform-masked positions ABOVE the frontier are "easy" tokens left masked.
    offpath_by_p = {}
    for p in p_levels:
        k = int(round(p * content_len))
        if k == 0 or k >= content_len:
            continue
        onpolicy_local = order_by_conf[:k]
        frontier = cc[onpolicy_local].max()
        fracs = []
        for it in per_item:
            if it["p"] != p or it["scheme"] != "uniform":
                continue
            masked_cc = cc[it["masked_local"]]
            fracs.append(float((masked_cc > frontier).mean()))
        if fracs:
            offpath_by_p[p] = fracs

    # ---------- M2: masked-position confidence at p = m2_p ----------
    m2 = {"onpolicy": [], "uniform": []}
    for it in per_item:
        if abs(it["p"] - m2_p) < 1e-6:
            m2[it["scheme"]].extend(it["maxprob"].tolist())

    return dict(delta_by_p=delta_by_p, offpath_by_p=offpath_by_p, m2=m2,
                order_by_conf=order_by_conf, cc=cc, content_len=content_len)


# --------------------------------------------------------------------------- #
# qualitative example
# --------------------------------------------------------------------------- #
def render_partial(tok, final, prompt_len, content_len, masked_local, mask_id):
    toks = final[0].clone()
    region = np.arange(prompt_len, prompt_len + content_len)
    abs_masked = region[masked_local]
    pieces = []
    for j in region:
        if j in set(abs_masked.tolist()):
            pieces.append(" ___")
        else:
            pieces.append(tok.decode([int(toks[j])]))
    return "".join(pieces)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="Dream-org/Dream-v0-Instruct-7B",
                    help="BASE dLLM. Do NOT use the trained checkpoint for the motivation figure.")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "none"],
                    help="'gsm8k' = fixed-seed subsample of the test split; 'none' = built-in prompts")
    ap.add_argument("--num_prompts", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="256 matches the gsm8k_cot eval setting")
    ap.add_argument("--tokens_per_step", type=int, default=2,
                    help="tokens unmasked per decode step (ranking is by commit-conf, so >1 is fine)")
    ap.add_argument("--num_uniform_draws", type=int, default=4)
    ap.add_argument("--m2_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] {args.pretrained}")
    model, tok = load_model(args.pretrained, args.dtype, args.device)
    mask_id = get_mask_id(model, tok)
    eos_id = tok.eos_token_id
    print(f"[info] mask_id={mask_id} eos_id={eos_id}")

    p_levels = [round(x, 2) for x in np.arange(0.1, 0.95, 0.1)]

    questions = load_questions(args.dataset, args.num_prompts, args.seed)

    # ---- generate trajectories ----
    trajectories = []
    for qi, q in enumerate(questions):
        input_ids = build_input_ids(tok, q, args.device)
        prompt_len = input_ids.shape[1]
        L = prompt_len + args.max_new_tokens
        x = torch.full((1, L), mask_id, dtype=torch.long, device=args.device)
        x[0, :prompt_len] = input_ids[0]

        final, commit_step, commit_conf = generate_with_trajectory(
            model, x, prompt_len, mask_id, args.tokens_per_step)

        gen = final[0, prompt_len:]
        eos_hits = (gen == eos_id).nonzero()
        content_len = int(eos_hits[0].item()) if len(eos_hits) else args.max_new_tokens
        content_len = max(content_len, 4)                 # guard tiny outputs

        trajectories.append(dict(
            final=final.cpu(), prompt_len=prompt_len,
            commit_step=commit_step, commit_conf=commit_conf,
            content_len=content_len, question=q))
        print(f"[gen] prompt {qi:2d}  content_len={content_len:3d}  "
              f"answer={tok.decode(gen[:content_len], skip_special_tokens=True)[:60]!r}")

    # ---- compute metrics ----
    agg_delta = {p: [] for p in p_levels}
    agg_offpath = {p: [] for p in p_levels}
    agg_m2 = {"onpolicy": [], "uniform": []}
    example = None

    # pick a legible rollout for the qualitative / cartoon panels:
    # the shortest answer with at least a handful of content tokens.
    content_lens = [t["content_len"] for t in trajectories]
    cand = [i for i, c in enumerate(content_lens) if c >= 8] or list(range(len(trajectories)))
    example_idx = min(cand, key=lambda i: content_lens[i])

    for ti, traj in enumerate(trajectories):
        out = compute_for_trajectory(model, traj, p_levels, args.num_uniform_draws,
                                     rng, mask_id, m2_p=args.m2_p)
        if out is None:
            continue
        for p, v in out["delta_by_p"].items():
            agg_delta[p].extend(v)
        for p, v in out["offpath_by_p"].items():
            agg_offpath[p].extend(v)
        agg_m2["onpolicy"].extend(out["m2"]["onpolicy"])
        agg_m2["uniform"].extend(out["m2"]["uniform"])

        if ti == example_idx:
            # stash an example trajectory for the frontier-cartoon panel
            cl = out["content_len"]
            cc = out["cc"]
            k = int(round(args.m2_p * cl))
            onpolicy_local = out["order_by_conf"][:k]
            unif_local = rng.choice(cl, size=k, replace=False)
            example = dict(
                commit_conf=cc.tolist(),
                content_len=cl,
                m2_p=args.m2_p,
                onpolicy_masked=onpolicy_local.tolist(),
                uniform_masked=unif_local.tolist(),
                text_onpolicy=render_partial(tok, traj["final"], traj["prompt_len"],
                                             cl, onpolicy_local, mask_id),
                text_uniform=render_partial(tok, traj["final"], traj["prompt_len"],
                                            cl, unif_local, mask_id),
                question=traj["question"],
            )

    def mean_se(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if len(xs) == 0:
            return float("nan"), float("nan")
        return float(xs.mean()), float(xs.std() / max(np.sqrt(len(xs)), 1))

    results = dict(
        pretrained=args.pretrained,
        p_levels=p_levels,
        m2_p=args.m2_p,
        delta_nll=[mean_se(agg_delta[p]) for p in p_levels],
        offpath_frac=[mean_se(agg_offpath[p]) for p in p_levels],
        example=example,
    )
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.out_dir, "m2_confidence.npz"),
             onpolicy=np.asarray(agg_m2["onpolicy"]),
             uniform=np.asarray(agg_m2["uniform"]))

    # write the qualitative example as plain text too
    if example is not None:
        with open(os.path.join(args.out_dir, "qualitative_example.txt"), "w") as f:
            f.write(f"QUESTION: {example['question']}\n\n")
            f.write(f"--- ON-POLICY partial state (p={args.m2_p}) ---\n")
            f.write(example["text_onpolicy"] + "\n\n")
            f.write(f"--- UNIFORM partial state (p={args.m2_p}) ---\n")
            f.write(example["text_uniform"] + "\n")

    print("\n[done] wrote results to", args.out_dir)
    print("  delta_nll (uniform - onpolicy), per p:")
    for p, (m, se) in zip(p_levels, results["delta_nll"]):
        print(f"    p={p:.1f}  ΔNLL={m:+.3f} ± {se:.3f}")
    print("  off-path fraction (easy tokens masked by uniform), per p:")
    for p, (m, se) in zip(p_levels, results["offpath_frac"]):
        print(f"    p={p:.1f}  frac={m:.3f} ± {se:.3f}")


if __name__ == "__main__":
    main()
