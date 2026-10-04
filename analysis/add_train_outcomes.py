"""Add the training-time outcomes that the d-OPSD log recorded to gsm8k_train_order.csv.

dOPSDTrainer prints a rich table of (prompt, completion, reward) for every process whenever a
generation round falls on global_step % completion_logging_steps == 0 (every 40 steps here, i.e. every
5th round of 3 prompts). Row k of the table at step S is rank k's prompt of round S / BATCH_DIVIDE,
i.e. training-order position n_gpus * (S / BATCH_DIVIDE) + k. The question text is cross-checked.

    python analysis/add_train_outcomes.py --log logs/dopsd-train-<job>.out --csv results/gsm8k_train_order.csv
"""

import argparse
import csv
import re

COLS = ["train_logged_step", "train_pred_logged", "train_correct_logged"]
DESC = [
    "仅我们的训练：日志记录了这道题时的训练步（每 40 步记录一次，共 102 道题有记录）；空 = 训练日志未记录",
    "训练时模型最终保留的那条 rollout 给出的答案（生成文本中最后一个 \\boxed{} 的内容）；空 = 未记录或未输出 \\boxed{}",
    "训练时这道题是否最终答对：1 = 在最多 8 次尝试（pass@8）内答对，该题参与了 loss；0 = 8 次都错，loss 乘 0；空 = 未记录",
]


def parse_tables(path):
    tables, cur, row = [], None, None
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        m = re.search(r"─ Step (\d+) ─", ln)
        if m:
            cur, row = {"step": int(m.group(1)), "rows": []}, None
            tables.append(cur)
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith("│ ├") or s.startswith("│ └"):
            row = None
            if s.startswith("│ └"):
                cur = None
            continue
        if s.startswith("│ │"):
            cells = [c.strip() for c in s.split("│")[2:-2]]
            if len(cells) < 3:
                continue
            if row is None:
                row = {"prompt": [], "completion": [], "reward": ""}
                cur["rows"].append(row)
            row["prompt"].append(cells[0])
            row["completion"].append(cells[1])
            row["reward"] = row["reward"] or cells[2]
    return tables


def last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return ""
    j, d = i + 7, 1
    while j < len(t) and d:
        d += (t[j] == "{") - (t[j] == "}")
        j += 1
    return t[i + 7:j - 1].strip() if d == 0 else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_gpus", type=int, default=3)
    ap.add_argument("--batch_divide", type=int, default=8)
    args = ap.parse_args()

    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    rows = list(csv.reader(open(args.csv, encoding="utf-8-sig")))
    head, desc, body = rows[0], rows[1], rows[2:]
    keep = [i for i, c in enumerate(head) if c not in COLS]  # idempotent: drop old outcome columns
    head, desc, body = [head[i] for i in keep], [desc[i] for i in keep], [[r[i] for i in keep] for r in body]
    qcol = head.index("question")

    out = {}
    mismatch = 0
    for t in parse_tables(args.log):
        for k, rw in enumerate(t["rows"]):
            o = args.n_gpus * (t["step"] // args.batch_divide) + k
            if o >= len(body):
                continue
            if norm(body[o][qcol])[:60] not in norm(" ".join(rw["prompt"])):
                mismatch += 1
                continue
            comp = norm(" ".join(rw["completion"]))
            out[o] = [t["step"], last_boxed(comp), int(float(rw["reward"]) >= 1.0)]
    if mismatch:
        raise SystemExit(f"{mismatch} logged rows did not match the reconstructed question; check --n_gpus/--batch_divide")

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head + COLS)
        w.writerow(desc + DESC)
        for k, r in enumerate(body):
            w.writerow(r + (out[k] if k in out else ["", "", ""]))
    n_ok = sum(v[2] for v in out.values())
    print(f"[outcomes] {len(out)} problems with logged outcomes ({n_ok} correct), question cross-check OK -> {args.csv}")


if __name__ == "__main__":
    main()
