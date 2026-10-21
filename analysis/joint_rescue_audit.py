"""CPU-only audit of saved Joint rollouts; never loads a model or changes training labels.

Example:
  OMP_NUM_THREADS=1 python analysis/joint_rescue_audit.py outputs/joint_math_v1 \
    --remote-manifest /tmp/joint_rescue_remote_inventory.json --out results/joint/rescue_audit

Only load trusted, self-produced .pt artifacts (torch pickle, weights_only=False).
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def key(path, meta):
    rank = int(re.search(r"_rank(\d+)\.pt$", path.name).group(1))
    return rank, meta["example_id"], meta["policy_version"]


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compatible(candidate, ref):
    meta, step = ref["meta"], ref["state"]
    state = step["state"]
    witness = candidate["final_state"]
    end = candidate["computed_length"]
    pos = ref["group_positions"].long() - meta["prompt_len"]
    known = state != meta["mask_id"]
    return (len(witness) >= len(state) and end <= len(witness) and end > int(pos.max())
            and not bool((witness[:end] == meta["mask_id"]).any())
            and torch.equal(witness[:len(state)][known], state[known])
            and candidate["meta"]["prompt_ids"] == meta["prompt_ids"])


def aggregate(rows):
    count_fields = ["states", "any_correct", "certified", "positions", "candidate_attempts",
                    "correct_candidates", "candidate_nfe", "teacher_nfe", "failed_state_candidate_nfe"]
    result = {field: sum(row[field] for row in rows) for field in count_fields}
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[(row["rank"], row["example_id"], row["policy_version"])].append(row)
    result.update(
        trajectories=len(by_trajectory),
        any_correct_trajectories=sum(any(r["any_correct"] for r in rr) for rr in by_trajectory.values()),
        certified_trajectories=sum(any(r["certified"] for r in rr) for rr in by_trajectory.values()),
        certified_states_per_trajectory=dict(sorted(Counter(
            sum(r["certified"] for r in rr) for rr in by_trajectory.values()).items())),
        sampled_states_per_trajectory=dict(sorted(Counter(map(len, by_trajectory.values())).items())),
        expected_certified_trajectories_one_uniform_saved_state=sum(
            sum(r["certified"] for r in rr) / len(rr) for rr in by_trajectory.values()),
        own_tail_loo_status=dict(Counter(r["own_tail_loo_status"] for r in rows)),
    )
    result["candidate_success_first_k"] = {
        str(k): sum(row[f"correct_first_{k}"] for row in rows) for k in (1, 2, 4, 8)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--remote-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    provenance = {"local_root": str(root.resolve()), "mode": "saved_artifacts_cpu_only"}
    if args.remote_manifest:
        manifest = json.loads(args.remote_manifest.read_text())
        for file in manifest["files"]:
            path = root / file["path"]
            require(path.is_file() and path.stat().st_size == file["bytes"]
                    and sha(path) == file["sha256"], f"Remote/local mismatch: {path}")
        provenance.update({k: v for k, v in manifest.items() if k != "files"})
        provenance.update(remote_manifest_sha256=sha(args.remote_manifest),
                          verified_files=len(manifest["files"]), mismatches=0)
        # Keep the full inventory with the ignored run artifacts, not the result table.
        inventory = root / "meta/rescue_remote_manifest.json"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps(manifest, indent=2) + "\n")

    natural = {}
    for path in sorted((root / "train/rollouts").glob("*.pt")):
        for roll in load(path)["rollouts"]:
            ident = key(path, roll["meta"])
            require(ident not in natural, f"Ambiguous natural rollout key: {ident}")
            natural[ident] = roll
    rows = []
    for path in sorted((root / "train/references").glob("*.pt")):
        ref = load(path)
        ident = key(path, ref["meta"])
        require(ident in natural, f"Missing natural rollout: {ident}")
        roll = natural[ident]
        require(ref["policy_version"] == ident[2], f"Stale policy: {path}")
        matches = [s for s in roll["steps"] if s["step_id"] == ref["state"]["step_id"]]
        require(len(matches) == 1, f"Ambiguous step: {path}")
        for field in ("state", "group_positions"):
            require(torch.equal(matches[0][field], ref["state"][field]), f"State mismatch: {path}/{field}")
        require(torch.equal(ref["group_positions"], matches[0]["group_positions"]), f"Reselection: {path}")
        require(roll["meta"] == ref["meta"], f"Metadata mismatch: {path}")
        candidates, stats = ref["candidates"], ref["stats"]
        correct = [c for c in candidates if c["is_correct"]]
        valid = [c for c in correct if compatible(c, ref)]
        require(stats["candidate_attempts"] == len(candidates) == 8, f"Candidate count: {path}")
        require(stats["correct_candidates"] == len(correct), f"Correct count: {path}")
        require(stats["compatible_candidates"] == len(valid), f"Compatible count: {path}")
        certified = ref["selected_tokens"] is not None
        require(stats["certified"] == int(certified), f"Certification count: {path}")
        require(stats["teacher_forwards"] == sum(a["teacher_forwards"] for a in ref["certification"]),
                f"Teacher count: {path}")
        positions = ref["group_positions"].long() - ref["meta"]["prompt_len"]
        if certified:
            witness = candidates[ref["selected_witness"]]
            require(witness["is_correct"] and compatible(witness, ref), f"Invalid witness: {path}")
            require(torch.equal(witness["final_state"][positions], ref["selected_tokens"]), f"Tuple mismatch: {path}")
            accepted = [a for a in ref["certification"] if a["accepted"]]
            require(len(accepted) == 1 and accepted[0]["tokens"] == ref["selected_tokens"].tolist()
                    and accepted[0]["top1"] == accepted[0]["tokens"], f"LOO mismatch: {path}")
        # An unchanged Teacher sees the same state/tuple, so a previously tested identical
        # tuple also determines the natural tail's result. Untested != rejected.
        own_status = "natural_wrong"
        if roll["is_correct"]:
            require(compatible(roll, ref), f"Own tail incompatible: {path}")
            own_tuple = roll["final_state"][positions].tolist()
            tests = [a for a in ref["certification"] if a["tokens"] == own_tuple]
            own_status = ("known_pass" if tests[0]["accepted"] else "known_fail") if tests else "untested"
        nfe = sum(c["stats"]["nfe"] for c in candidates)
        rows.append(dict(
            file=str(path.relative_to(root)), rank=ident[0], example_id=ident[1], policy_version=ident[2],
            natural_correct=int(roll["is_correct"]), step_id=ref["state"]["step_id"],
            block_id=ref["state"]["block_id"], relative_step=ref["state"]["step_id"] / len(roll["steps"]),
            states=1, any_correct=int(bool(correct)), certified=int(certified),
            positions=len(positions) if certified else 0, candidate_attempts=len(candidates),
            correct_candidates=len(correct), candidate_nfe=nfe, teacher_nfe=stats["teacher_forwards"],
            failed_state_candidate_nfe=0 if certified else nfe, own_tail_loo_status=own_status,
            **{f"correct_first_{k}": int(any(c["is_correct"] for c in candidates[:k])) for k in (1, 2, 4, 8)},
        ))

    require(set((r["rank"], r["example_id"], r["policy_version"]) for r in rows) == set(natural),
            "Some natural rollouts have no saved reference states")
    require(len({(r["rank"], r["example_id"], r["policy_version"], r["step_id"]) for r in rows}) == len(rows),
            "Duplicate saved state")
    groups = {"all": aggregate(rows)}
    for label, flag in (("natural_wrong", 0), ("natural_correct", 1)):
        groups[label] = aggregate([r for r in rows if r["natural_correct"] == flag])
    logs = [json.loads(line) for path in sorted((root / "train").glob("log_rank*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]
    checks = {"local_states": "certified", "joint_states_attempted": "states",
              "joint_certified": "certified", "joint_candidate_attempts": "candidate_attempts",
              "joint_correct_candidates": "correct_candidates", "joint_teacher_forwards": "teacher_nfe"}
    log_sums = {field: sum(r.get(field, 0) for r in logs) for field in checks}
    for field, metric in checks.items():
        require(log_sums[field] == groups["all"][metric], f"Training log mismatch: {field}")
    require(sum(r["refresh_correct"] for r in logs) == sum(r["is_correct"] for r in natural.values()),
            "Natural correctness log mismatch")
    result = dict(
        provenance=provenance, validation="all checks passed",
        label_protocol="Original saved is_correct labels; legacy_math parser/string scorer; no relabeling",
        natural_nfe=sum(r["stats"]["nfe"] for r in natural.values()),
        unique_example_ids=len({r["meta"]["example_id"] for r in natural.values()}),
        groups=groups, training_log_sums=log_sums,
        by_block={str(b): {label: aggregate([r for r in rows if r["block_id"] == b and r["natural_correct"] == flag])
                          for label, flag in (("natural_wrong", 0), ("natural_correct", 1))}
                  for b in sorted({r["block_id"] for r in rows})},
        limitations=[
            "Rescue = a correct compatible continuation passing LOO, not demonstrated downstream model improvement.",
            "Originally wrong final answer does not establish that the sampled state already contained an error.",
            "Four states times eight continuations per trajectory; not one state times eight.",
            "No equal-compute prompt-restart baseline or removal ablation was run.",
            "NFE is recorded forward-call count, not GPU time; excludes group readout, training/backward and other overhead.",
            "First-k bank coverage measures answer correctness only, not exact LOO acceptance at smaller K.",
            "Known own-tail LOO results cover previously tested identical tuples only; untested results remain unknown.",
            "Accepted states entered losses, including initial zero-learning-rate update; not every target changed weights.",
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k != "by_block"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
