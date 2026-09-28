"""Per-step denoising trace for the LLaDA semi-AR samplers (pure torch).

Used by both `eval/generate.py::generate` and `d-opsd/utils.py::generate` via an
optional `recorder=` argument (default None -> sampler behaviour is unchanged).
The sampler calls, once per denoising step, BEFORE writing the new tokens:

    recorder.record_step(block, x, logits, x0, transfer)

  x         [B, N]    state S_t (prompt + completion, mask id at undecoded slots)
  logits    [B, N, V] raw model logits at step t (no Gumbel noise)
  x0        [B, N]    the sampler's candidate token per position (argmax of noisy logits)
  transfer  [B, N]    bool, positions revealed at this step (C_t)

and `recorder.finalize(x_final)` once at the end. Per step and per rollout this
keeps: S_t, mask ratio, C_t (+ chosen token, its raw prob and rank), and for EVERY
masked position of the completion (current block and future blocks): entropy,
max prob, the sampler's candidate token and its raw prob (= LLaDA's
low_confidence score for in-block positions), and top-k ids/probs.
Full-vocab logits are not stored (~16 GB per 256x128 rollout); top-k is.
"""

import torch
import torch.nn.functional as F


class LLaDATraceRecorder:
    def __init__(self, mask_id, prompt_len, topk=20, chunk=256):
        self.mask_id = int(mask_id)
        self.P = int(prompt_len)
        self.topk = int(topk)
        self.chunk = int(chunk)
        self.states = []  # S_0 .. S_T, each [B, L] (completion span)
        self.blocks = []  # block index of step t
        self.m = {k: [] for k in ("step", "row", "pos", "entropy", "maxprob",
                                  "cand_token", "cand_prob", "topk_ids", "topk_probs")}
        self.r = {k: [] for k in ("step", "row", "pos", "token", "prob", "rank")}

    @torch.no_grad()
    def record_step(self, block, x, logits, x0, transfer):
        t = len(self.states)
        comp = x[:, self.P:]
        self.states.append(comp.clone())
        self.blocks.append(int(block))
        comp_logits, comp_x0 = logits[:, self.P:], x0[:, self.P:]

        rows, pos = (comp == self.mask_id).nonzero(as_tuple=True)
        for s in range(0, rows.numel(), self.chunk):
            rr, pp = rows[s:s + self.chunk], pos[s:s + self.chunk]
            logp = F.log_softmax(comp_logits[rr, pp].float(), dim=-1)
            p = logp.exp()
            cand = comp_x0[rr, pp]
            tk_p, tk_i = p.topk(self.topk, dim=-1)
            self.m["step"].append(torch.full_like(rr, t))
            self.m["row"].append(rr)
            self.m["pos"].append(pp)
            self.m["entropy"].append(-(p * logp).sum(-1))
            self.m["maxprob"].append(tk_p[:, 0])
            self.m["cand_token"].append(cand)
            self.m["cand_prob"].append(p.gather(-1, cand.unsqueeze(-1)).squeeze(-1))
            self.m["topk_ids"].append(tk_i)
            self.m["topk_probs"].append(tk_p)

        rows, pos = transfer[:, self.P:].nonzero(as_tuple=True)
        if rows.numel():
            tok = comp_x0[rows, pos]
            logp = F.log_softmax(comp_logits[rows, pos].float(), dim=-1)
            chosen = logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            self.r["step"].append(torch.full_like(rows, t))
            self.r["row"].append(rows)
            self.r["pos"].append(pos)
            self.r["token"].append(tok)
            self.r["prob"].append(chosen.exp())
            self.r["rank"].append((logp > chosen.unsqueeze(-1)).sum(-1))

    @staticmethod
    def _cat(parts):
        return torch.cat(parts, dim=0).cpu() if parts else torch.empty(0, dtype=torch.long)

    def finalize(self, x_final):
        """Append the final state and split into one compact dict per batch row (CPU)."""
        self.states.append(x_final[:, self.P:].clone())
        states = torch.stack(self.states, dim=1).cpu()  # [B, T+1, L]
        B, T1, L = states.shape
        m = {k: self._cat(v) for k, v in self.m.items()}
        r = {k: self._cat(v) for k, v in self.r.items()}
        blocks = torch.tensor(self.blocks, dtype=torch.short)

        out = []
        for b in range(B):
            s_b = states[b]
            mb, rb = m["row"] == b, r["row"] == b
            r_step, r_pos = r["step"][rb].long(), r["pos"][rb].long()
            n_revealed = torch.zeros(T1 - 1, dtype=torch.long).index_add_(
                0, r_step, torch.ones_like(r_step)
            )
            reveal_step = torch.full((L,), -1, dtype=torch.long)
            reveal_step[r_pos] = r_step
            m_step, m_pos = m["step"][mb].long(), m["pos"][mb].long()
            out.append({
                "states": s_b.int(),                                    # [T+1, L]  S_t
                "mask_ratio": (s_b == self.mask_id).float().mean(-1),   # [T+1]
                "block": blocks,                                        # [T]  block of step t
                "n_revealed": n_revealed.short(),                       # [T]  |C_t|
                "reveal_step": reveal_step.short(),                     # [L]  t with j in C_t
                "reveal": {                                             # every C_t entry
                    "step": r_step.short(),
                    "pos": r_pos.short(),
                    "token": r["token"][rb].int(),
                    "prob": r["prob"][rb].float(),
                    "rank": r["rank"][rb].int(),
                },
                "masked": {                                             # every (t, masked j)
                    "step": m_step.short(),
                    "pos": m_pos.short(),
                    "entropy": m["entropy"][mb].half(),
                    "maxprob": m["maxprob"][mb].half(),
                    "cand_token": m["cand_token"][mb].int(),
                    "cand_prob": m["cand_prob"][mb].float(),
                    "topk_ids": m["topk_ids"][mb].int(),
                    "topk_probs": m["topk_probs"][mb].half(),
                    "revealed": reveal_step[m_pos] == m_step,
                },
            })
        return out
