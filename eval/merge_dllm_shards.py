"""Merge per-shard JSON outputs from evaluate_aime_dllm.py and recompute aggregate metrics.

Each shard process writes its own JSON via --output_file. This script concatenates the
per-problem `results` from every shard and recomputes pass@n / average@n / majority@n
over the full problem set.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_files", nargs="+", help="Per-shard JSON files to merge.")
    parser.add_argument("-o", "--output", required=True, help="Path for the merged JSON.")
    args = parser.parse_args()

    merged_results = []
    base = None
    for path in args.shard_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if base is None:
            base = {
                k: data[k]
                for k in (
                    "base_model", "checkpoint_dir", "dataset", "generator",
                    "temperature", "top_p", "max_new_tokens", "diffusion_steps",
                    "alg", "alg_temp", "val_n", "batch_size",
                )
                if k in data
            }
        merged_results.extend(data["results"])

    val_n = base["val_n"]
    num_problems = len(merged_results)
    total_generations = sum(len(r["generations"]) for r in merged_results)
    pass_at_n = sum(1 for r in merged_results if r["pass_at_n"])
    total_correct = sum(r["num_correct"] for r in merged_results)
    majority_vote_count = sum(1 for r in merged_results if r["majority_vote_correct"])
    formatted_count = sum(
        sum(1 for g in r["generations"] if g["formatted"]) for r in merged_results
    )

    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct / total_generations * 100
    majority_vote_at_n_pct = majority_vote_count / num_problems * 100
    format_rate = formatted_count / total_generations * 100

    print("=" * 70)
    print(f"MERGED RESULTS ({len(args.shard_files)} shards)")
    print("=" * 70)
    print(f"Dataset: {base.get('dataset', '?').upper()}")
    print(f"Problems: {num_problems}")
    print(f"Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct}/{total_generations})")
    print(f"Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_count}/{num_problems})")
    print(f"Format rate: {format_rate:.2f}% ({formatted_count}/{total_generations})")
    print("=" * 70)

    summary = {
        **base,
        "num_problems": num_problems,
        "total_solutions": total_generations,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": pass_at_n_pct,
        "average_at_n": total_correct,
        "average_at_n_pct": average_at_n_pct,
        "majority_vote_at_n": majority_vote_count,
        "majority_vote_at_n_pct": majority_vote_at_n_pct,
        "formatted_count": formatted_count,
        "format_rate": format_rate,
        "shard_files": [str(p) for p in args.shard_files],
        "results": merged_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Merged file: {output_path}")


if __name__ == "__main__":
    main()
