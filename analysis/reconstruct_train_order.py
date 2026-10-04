"""Reconstruct which GSM8K training problems d-OPSD consumed, in order.

Output CSV: row 1 = header, row 2 = a description of every column (not data; skip it when parsing).

The order is deterministic given the released code:
  d_opsd_train.py   dataset = get_gsm8k_questions("train").shuffle(seed=42)   (HF datasets, np PCG64)
  trl GRPOTrainer   RepeatRandomSampler(seed=42): torch.randperm(N, generator seeded 42), cut into
                    chunks of n_gpus * per_device_bs / num_generations prompts; each chunk is repeated
                    num_iterations (= BATCH_DIVIDE) times, i.e. one chunk spans BATCH_DIVIDE optimizer steps.
The permutation does not depend on the GPU count or BATCH_DIVIDE; those only set the chunk size and how
many steps a chunk spans. So prompts used up to optimizer step S = first n_gpus * ceil(S / BD) entries.

    python analysis/reconstruct_train_order.py --out train_order.csv [--n 600] \\
        [--verify_traces RUN/traces --verify_gpus 3 --verify_bd 8]

--verify_traces checks the reconstruction against our own training traces (meta.example.question,
meta.gen_round): the traced question must lie in chunk `gen_round` of size `verify_gpus`.
"""

import argparse
import csv
import glob
import math
import os

import torch
from datasets import load_dataset


# Row 2 of the CSV: what each column means (row 1 is the machine-readable header).
DESC = [
    "【说明行，不是数据】取题顺序：第几道被用到的训练题（0 起）。THU 和我们按同一顺序取题",
    "THU 设定（4 卡）下属于第几批；每批 4 道题，每张卡各 1 道",
    "THU 若 BATCH_DIVIDE=4：这批题占用的优化步范围（含两端），如 0-3 表示第 0 到 3 步",
    "THU 若 BATCH_DIVIDE=8：这批题占用的优化步范围（含两端）",
    "我们的设定（3 卡）下属于第几批；每批 3 道题",
    "我们（BATCH_DIVIDE=8）这批题占用的优化步范围（含两端）",
    "1 = THU 若 BATCH_DIVIDE=4，训练到论文最佳的第 425 步时已用过这道题；0 = 尚未用到（共 428 个 1）",
    "1 = THU 若 BATCH_DIVIDE=8，训练到第 425 步时已用过；0 = 尚未用到（共 216 个 1）",
    "1 = 我们选出的最佳 checkpoint step-448 已用过这道题；0 = 尚未用到（共 168 个 1）",
    "1 = 我们训练到最后（step-1344）已用过；0 = 尚未用到（共 504 个 1）",
    "这道题在 HuggingFace openai/gsm8k（main）train 原始数据中的行号（0 起）",
    "题目原文",
    "标准答案（取自 #### 之后）",
]


def order(seed=42):
    data = load_dataset("openai/gsm8k", "main")["train"]
    data = data.map(lambda x, i: {"orig_idx": i}, with_indices=True)  # map keeps the row order
    data = data.shuffle(seed=seed)  # as d_opsd_train.py
    g = torch.Generator()
    g.manual_seed(seed)  # as RepeatRandomSampler(seed=args.seed)
    perm = torch.randperm(len(data), generator=g).tolist()
    return data, perm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600, help="number of prompts to list")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify_traces", default="")
    ap.add_argument("--verify_gpus", type=int, default=3)
    ap.add_argument("--verify_bd", type=int, default=8)
    args = ap.parse_args()

    data, perm = order(args.seed)
    # THU (4 GPUs, paper Table 2: best GSM8K at step 425) under both BATCH_DIVIDE settings; ours (3 GPUs, BD 8).
    cutoffs = {
        "thu_bd4_step425": 4 * math.ceil(425 / 4),
        "thu_bd8_step425": 4 * math.ceil(425 / 8),
        "ours_step448": 3 * math.ceil(448 / 8),
        "ours_step1344": 3 * math.ceil(1344 / 8),
    }
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel shows Chinese correctly
        w = csv.writer(f)
        w.writerow(["order", "thu_chunk(4 prompts)", "thu_steps_if_bd4", "thu_steps_if_bd8", "ours_chunk(3 prompts)",
                    "ours_steps_bd8", *cutoffs, "gsm8k_train_idx", "question", "answer"])
        w.writerow(DESC)  # skip this row when parsing, e.g. pandas.read_csv(..., skiprows=[1])
        for k in range(min(args.n, len(perm))):
            ex = data[perm[k]]
            c4, c3 = k // 4, k // 3
            w.writerow([k, c4, f"{c4 * 4}-{c4 * 4 + 3}", f"{c4 * 8}-{c4 * 8 + 7}", c3, f"{c3 * 8}-{c3 * 8 + 7}",
                        *[int(k < v) for v in cutoffs.values()], ex["orig_idx"], ex["question"],
                        ex["answer"].split("####")[-1].strip()])
    print(f"[order] wrote {min(args.n, len(perm))} rows -> {args.out}; cutoffs {cutoffs}")

    if args.verify_traces:
        pos = {data[p]["question"]: k for k, p in enumerate(perm[:5000])}
        ok = bad = 0
        for fpath in sorted(glob.glob(os.path.join(args.verify_traces, "*.pt"))):
            m = torch.load(fpath, weights_only=False)["meta"]
            q, r = m["example"]["question"], int(m["gen_round"])
            k = pos.get(q)
            hit = k is not None and k // args.verify_gpus == r
            ok += hit
            bad += not hit
            print(f"  {os.path.basename(fpath)}: gen_round={r} reconstructed_order={k} -> {'OK' if hit else 'MISMATCH'}")
        print(f"[verify] {ok} match, {bad} mismatch")


if __name__ == "__main__":
    main()
