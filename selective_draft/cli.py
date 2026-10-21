"""python -m selective_draft.cli <command> --config CONFIG [--set section.field=value ...]

commands: prepare-data | inspect | collect | train-readout | diagnose | train-guidance | train-joint | evaluate |
          merge-eval | layer-profile | readout-sweep
Real-model commands run on a TACC compute node; the fake backend runs everything on CPU."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import torch

from .checkpoint import code_revision, load_snapshot
from .config import Config
from .data import load_split, prepare_manifest, tokenize_prompts
from .scoring import grade
from .generation import generate
from .loading import build_model_and_tokenizer
from .trace import StateRecorder, load_rollouts, replay_check, save_rollouts, summarize_rollouts


def _common(p: argparse.ArgumentParser, output=True):
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], help="section.field=value (JSON value)")
    p.add_argument("--device", default="auto")
    if output:
        p.add_argument("--output-dir", required=True)


def _cfg(args) -> Config:
    cfg = Config.load(args.config).apply_overrides(args.set).validate()
    return cfg


def _device(args) -> torch.device:
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.index is None:
            device = torch.device("cuda", local_rank)
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1 and device.index != local_rank:
            raise ValueError("distributed CUDA runs must use their LOCAL_RANK device")
        torch.cuda.set_device(device)  # before loading weights or creating NCCL collectives
    return device


def _rollout_paths(items: List[str]) -> List[str]:
    out = []
    for it in items:
        if os.path.isdir(it):
            out.extend(sorted(glob.glob(os.path.join(it, "*.pt"))))
        else:
            out.extend(sorted(glob.glob(it)))
    if not out:
        raise FileNotFoundError(f"no rollout shards under {items}")
    return out


def _build(cfg, args, snapshot: Optional[str] = None):
    gm, tok = build_model_and_tokenizer(cfg, _device(args))
    snap_id = "none"
    if snapshot:
        man = load_snapshot(snapshot, gm, cfg)
        snap_id = f"{os.path.abspath(snapshot)}@{man.get('stage')}"
    return gm, tok, snap_id


def _dump_resolved(cfg: Config, out_dir: str, extra: dict) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    cfg.dump(os.path.join(out_dir, "resolved_config.json"))
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        json.dump(dict(extra, code_revision=code_revision(), argv=sys.argv), f, indent=1)


# ----------------------------------------------------------------------------- commands
def cmd_prepare_data(args):
    cfg = _cfg(args)
    path = os.path.join(args.output_dir, "split_manifest.json")
    man = prepare_manifest(cfg.data, path, force=args.force)
    print(json.dumps({"path": path, "sizes": man["sizes"]}, indent=1))


def cmd_inspect(args):
    cfg = _cfg(args)
    gm, tok, _ = _build(cfg, args, args.snapshot)
    gm.assert_base_frozen()
    info = {
        "resolved_config": cfg.to_dict(), "backend": gm.backend.describe(),
        "boundary_module": (type(gm.backend.boundary_module(cfg.guidance.early_layer)).__name__
                            if cfg.guidance.early_layer is not None else None),
        "early_layer_block_index": cfg.guidance.early_layer - 1 if cfg.guidance.early_layer is not None else None,
        "trainable_names_stage_readout": (gm.set_stage("readout"), gm.trainable_parameter_names())[1],
        "trainable_names_stage_guidance": (gm.set_stage("guidance"), gm.trainable_parameter_names())[1],
        "module_params": {n: p.numel() for n, p in gm.named_parameters() if not n.startswith("base.")},
        "tokenizer_eos": getattr(tok, "eos_token_id", None), "tokenizer_pad": getattr(tok, "pad_token_id", None),
        "code_revision": code_revision(),
    }
    gm.set_stage(None)
    print(json.dumps(info, indent=1, default=str))


def cmd_collect(args):
    cfg = _cfg(args)
    level = cfg.trace.level if cfg.trace.level != "none" else "state_only"
    gm, tok, snap_id = _build(cfg, args, args.snapshot)
    examples = load_split(cfg, tok)
    if args.limit:
        examples = examples[: args.limit]
    si, sn = (int(v) for v in args.shard.split("/")) if args.shard else (0, 1)
    if si == 0:
        _dump_resolved(cfg, args.output_dir, {"command": "collect", "snapshot": snap_id, "n_examples": len(examples)})
    os.makedirs(args.output_dir, exist_ok=True)
    device = gm.device
    bs = cfg.evaluation.batch_size
    if sn > 1 and bs != 1:
        raise SystemExit("collect --shard needs evaluation.batch_size=1")
    rng = torch.Generator(device=device).manual_seed(cfg.training.seed)
    all_summ = []
    for s in range(0, len(examples), bs):
        shard = s // bs  # file index = global batch index: shards and resumed runs never collide
        if shard % sn != si:
            continue
        path = os.path.join(args.output_dir, f"rollouts-{shard:05d}.pt")
        if args.resume and os.path.exists(path):
            all_summ.extend(load_rollouts(path))
            continue
        if cfg.generation.temperature > 0 and (sn > 1 or args.resume):  # per-example RNG so shards are reproducible
            rng = torch.Generator(device=device).manual_seed(cfg.training.seed + shard)
        batch = examples[s:s + bs]
        ids, mask = tokenize_prompts(tok, [e.prompt_text for e in batch], device)
        metas = [{"example_id": e.example_id, "prompt_ids": ids[i].tolist(), "prompt_attention_mask": mask[i].tolist(),
                  "gold": e.answer_value, "guidance_mode": cfg.guidance.mode, "snapshot_id": snap_id,
                  "generation": cfg.generation.__dict__, "split": cfg.data.split,
                  "confidence_policy": "raw_model_probability"} for i, e in enumerate(batch)]
        rec = StateRecorder(level, gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1], metas, topk=cfg.trace.topk)
        res = generate(gm, ids, mask, cfg.generation, guidance_mode=cfg.guidance.mode, recorder=rec, sample_rng=rng)
        extras = []
        for i, e in enumerate(batch):
            text = tok.decode(res.completion_ids(i), skip_special_tokens=True)
            g = grade(e.question, e.answer_value, text, e.task, cfg.data.scorer)
            extras.append({"generation": text, "parsed_answer": g["parsed_answer"], "is_correct": g["is_correct"],
                           "verdicts": g["verdicts"],
                           "stats": res.stats})
        rolls = rec.finalize(res, extras)
        for r in rolls:
            replay_check(r)
        save_rollouts(path, rolls)
        all_summ.extend(rolls)
        print(f"[collect] {s + len(batch)}/{len(examples)} -> {path} nfe={res.stats['nfe']} "
              f"correct={sum(r['is_correct'] for r in rolls)}/{len(rolls)}")
    summ = summarize_rollouts(all_summ)
    with open(os.path.join(args.output_dir, "summary.json" if sn == 1 else f"summary.shard{si}of{sn}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


def cmd_train_readout(args):
    from .training import ReadoutTrainer, run_training
    cfg = _cfg(args)
    gm, tok, snap_id = _build(cfg, args, args.snapshot)
    _dump_resolved(cfg, args.output_dir, {"command": "train-readout", "init_snapshot": snap_id, "states": args.states})
    trainer = ReadoutTrainer(gm, cfg, args.output_dir, _rollout_paths(args.states))
    last = run_training(trainer, args.max_steps or cfg.training.max_steps, args.save_every or cfg.training.save_every, args.resume)
    print(json.dumps({"final": last, "snapshot": os.path.join(args.output_dir, "snapshot")}, default=float))


def cmd_train_guidance(args):
    from .training import GuidanceTrainer, run_training
    cfg = _cfg(args)
    if cfg.teacher.mode == "joint":
        raise SystemExit("use train-joint for the joint-reference objective")
    gm, tok, snap_id = _build(cfg, args, args.readout_snapshot)
    if not args.readout_snapshot and not args.resume:
        raise SystemExit("train-guidance needs --readout-snapshot (stage 1 output) or --resume")
    examples = load_split(cfg, tok)
    _dump_resolved(cfg, args.output_dir, {"command": "train-guidance", "readout_snapshot": snap_id, "n_examples": len(examples)})
    trainer = GuidanceTrainer(gm, cfg, args.output_dir, examples, tok)
    last = run_training(trainer, args.max_steps or cfg.training.max_steps, args.save_every or cfg.training.save_every, args.resume)
    print(json.dumps({"final": last, "snapshot": os.path.join(args.output_dir, "snapshot")}, default=float))


def cmd_train_joint(args):
    from .joint import JointTrainer
    from .training import run_training
    cfg = _cfg(args)
    if not args.readout_snapshot and not args.resume:
        raise SystemExit("train-joint needs --readout-snapshot or --resume")
    if args.readout_snapshot and args.resume:
        raise SystemExit("choose either a readout snapshot for a fresh run or a joint checkpoint to resume")
    if cfg.data.split != "train":
        raise SystemExit("train-joint requires data.split=train")
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.save_every is not None:
        cfg.training.save_every = args.save_every
    if args.readout_snapshot:
        with open(os.path.join(args.readout_snapshot, "manifest.json")) as f:
            if json.load(f).get("stage") != "readout":
                raise SystemExit("--readout-snapshot must be a stage-readout snapshot (fresh zero injector)")
    gm, tok, snap_id = _build(cfg, args, args.readout_snapshot)
    examples = load_split(cfg, tok)
    _dump_resolved(cfg, args.output_dir, {"command": "train-joint", "readout_snapshot": snap_id,
                                        "n_examples": len(examples)})
    if cfg.training.sampling_protocol == "root_success_v2":
        from .root_joint import RootSuccessTrainer
        trainer = RootSuccessTrainer(gm, cfg, args.output_dir, examples, tok)
    else:
        trainer = JointTrainer(gm, cfg, args.output_dir, examples, tok)
    last = run_training(trainer, cfg.training.max_steps, cfg.training.save_every, args.resume)
    print(json.dumps({"final": last, "snapshot": os.path.join(args.output_dir, "snapshot")}, default=float))


def cmd_diagnose(args):
    from .diagnostics import consistency_report, readout_report, teacher_report
    cfg = _cfg(args)
    gm, tok, snap_id = _build(cfg, args, args.snapshot)
    rollouts = [r for p in _rollout_paths(args.states) for r in load_rollouts(p)]
    _dump_resolved(cfg, args.output_dir, {"command": f"diagnose:{args.kind}", "snapshot": snap_id, "states": args.states})
    if args.kind == "readout":
        rep = readout_report(gm, rollouts, cfg, max_states=args.max_states)
    elif args.kind == "teacher":
        rep = teacher_report(gm, rollouts, cfg, max_states=args.max_states, oracle=not args.no_oracle)
    elif args.kind == "consistency":
        rep = consistency_report(gm, rollouts, cfg, max_steps=args.max_states)
    else:
        raise SystemExit(args.kind)
    with open(os.path.join(args.output_dir, f"{args.kind}_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


def cmd_layer_profile(args):
    from .layer_profile import layer_profile, pick_states
    cfg = _cfg(args)
    gm, tok, _ = _build(cfg, args)
    rollouts = [r for p in _rollout_paths(args.states) for r in load_rollouts(p)]
    picks = pick_states(rollouts)
    _dump_resolved(cfg, args.output_dir, {"command": "layer-profile", "states": args.states, "n_states": len(picks)})
    rep = layer_profile(gm, picks, cfg)
    with open(os.path.join(args.output_dir, "layer_profile.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(os.path.join(args.output_dir, "layer_decision.json"), "w") as f:
        json.dump(dict(rep["decision"], states=args.states, code_revision=code_revision()), f, indent=1, default=float)
    print(json.dumps(rep["decision"], indent=1, default=float))


def cmd_readout_sweep(args):
    """§7.1.1 metrics for step0 (native lens) and each checkpoint on one fixed state bank; --select applies §7.1.2."""
    from .layer_profile import pick_states, readout_metrics, select_readout
    cfg = _cfg(args)
    gm, tok, _ = _build(cfg, args)
    rollouts = [r for p in _rollout_paths(args.states) for r in load_rollouts(p)]
    picks = pick_states(rollouts)
    _dump_resolved(cfg, args.output_dir, {"command": "readout-sweep", "states": args.states,
                                        "checkpoints": args.checkpoints, "n_states": len(picks)})
    results, steps = {"step0": readout_metrics(gm, picks, cfg)}, {"step0": 0}
    for ck in args.checkpoints:
        snap = os.path.join(ck, "snapshot") if os.path.isdir(os.path.join(ck, "snapshot")) else ck
        load_snapshot(snap, gm, cfg)
        name = os.path.basename(os.path.normpath(ck))
        with open(os.path.join(ck, "trainer_state.json")) as f:
            steps[name] = int(json.load(f)["optimizer_step"])
        results[name] = readout_metrics(gm, picks, cfg)
        print(name, json.dumps(results[name]["summary"], default=float))
    out = {"summaries": {k: v["summary"] for k, v in results.items()}, "steps": steps,
           "checkpoints": {os.path.basename(os.path.normpath(c)): os.path.abspath(c) for c in args.checkpoints}}
    if args.select:
        out["selection"] = select_readout(results, steps)
        print(json.dumps(out["selection"], indent=1, default=float))
    with open(os.path.join(args.output_dir, "readout_sweep.json"), "w") as f:
        json.dump(dict(out, rows={k: v["rows"] for k, v in results.items()}), f, indent=1, default=float)


def cmd_evaluate(args):
    from .evaluation import evaluate
    cfg = _cfg(args)
    gm, tok, snap_id = _build(cfg, args, args.snapshot)
    if cfg.guidance.mode != "off" and not args.snapshot:
        print("[warn] guidance.mode != off without a snapshot: modules are at initialisation (readout = native lens, "
              "injector = zero) -> this is the readout_only/zero-injection parity baseline")
    examples = load_split(cfg, tok)
    limit = args.limit or cfg.evaluation.limit
    if limit:
        examples = examples[:limit]
    shard = (0, 1)
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        shard = (i, n)
    os.makedirs(args.output_dir, exist_ok=True)  # every shard: they start concurrently
    if shard == (0, 1):
        _dump_resolved(cfg, args.output_dir, {"command": "evaluate", "snapshot": snap_id, "n_examples": len(examples)})
    elif shard[0] == 0:
        cfg.dump(os.path.join(args.output_dir, "resolved_config.json"))
    with open(os.path.join(args.output_dir, f"example_ids.shard{shard[0]}of{shard[1]}.json"), "w") as f:
        json.dump([e.example_id for e in examples], f)
    out = evaluate(gm, tok, examples, cfg, cfg.guidance.mode, args.output_dir, snapshot_id=snap_id,
                   trace_level=cfg.trace.level, shard=shard, resume=args.resume)
    print(json.dumps(out["summaries"], indent=1, default=float))


def cmd_merge_eval(args):
    from .evaluation import merge_shards
    lists = sorted(glob.glob(os.path.join(args.output_dir, "example_ids.shard*of*.json")))
    expected = None
    if lists:
        sets = [json.load(open(p)) for p in lists]
        if any(s != sets[0] for s in sets):
            raise SystemExit("shards were launched with different example lists")
        expected = sets[0]
    out = merge_shards(args.output_dir, expected)
    print(json.dumps(out["summaries"], indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="selective_draft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="write the sample-ID split manifest (never overwrites without --force)")
    _common(p); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_prepare_data)

    p = sub.add_parser("inspect", help="print backend layout, token ids, trainable names, resolved config")
    _common(p, output=False); p.add_argument("--snapshot"); p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("collect", help="decode the configured split and save CPU state shards")
    _common(p); p.add_argument("--snapshot"); p.add_argument("--limit", type=int)
    p.add_argument("--shard", help="I/N: every N-th example starting at I (batch size 1)")
    p.add_argument("--resume", action="store_true", help="skip rollout files that already exist")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("train-readout", help="stage 1: train g_phi on recorded base states")
    _common(p); p.add_argument("--states", nargs="+", required=True); p.add_argument("--snapshot")
    p.add_argument("--max-steps", type=int); p.add_argument("--save-every", type=int); p.add_argument("--resume")
    p.set_defaults(fn=cmd_train_readout)

    p = sub.add_parser("train-joint", help="joint-reference sibling PI: pass@K continuations + fixed-point certification")
    _common(p); p.add_argument("--readout-snapshot"); p.add_argument("--max-steps", type=int)
    p.add_argument("--save-every", type=int); p.add_argument("--resume"); p.set_defaults(fn=cmd_train_joint)

    p = sub.add_parser("diagnose", help="readout | teacher | consistency reports on recorded states")
    _common(p); p.add_argument("--kind", required=True, choices=["readout", "teacher", "consistency"])
    p.add_argument("--states", nargs="+", required=True); p.add_argument("--snapshot")
    p.add_argument("--max-states", type=int); p.add_argument("--no-oracle", action="store_true")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("train-guidance", help="stage 2: on-policy training of J_psi with a frozen readout")
    _common(p); p.add_argument("--readout-snapshot"); p.add_argument("--max-steps", type=int)
    p.add_argument("--save-every", type=int); p.add_argument("--resume"); p.set_defaults(fn=cmd_train_guidance)

    p = sub.add_parser("evaluate", help="task accuracy + latency; threshold sweep via evaluation.commit_thresholds")
    _common(p); p.add_argument("--snapshot"); p.add_argument("--limit", type=int)
    p.add_argument("--shard", help="I/N: evaluate examples I, I+N, ... (batch size 1); merge with merge-eval")
    p.add_argument("--resume", action="store_true", help="keep rows already written under the same fingerprint")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("layer-profile", help="all-layer logit-lens profile on a fixed state bank + layer rule (§7.1)")
    _common(p); p.add_argument("--states", nargs="+", required=True); p.set_defaults(fn=cmd_layer_profile)

    p = sub.add_parser("readout-sweep", help="readout checkpoint metrics on fixed states; --select applies §7.1.2")
    _common(p); p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="*", default=[]); p.add_argument("--select", action="store_true")
    p.set_defaults(fn=cmd_readout_sweep)

    p = sub.add_parser("merge-eval", help="merge evaluate --shard outputs into rows_<tag>.jsonl + summary.json")
    p.add_argument("--output-dir", required=True); p.set_defaults(fn=cmd_merge_eval)

    args = ap.parse_args(argv)
    import time as _time
    t0 = _time.time()
    status = "ok"
    try:
        args.fn(args)
    except BaseException as exc:
        status = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        out_dir = getattr(args, "output_dir", None)
        if out_dir and int(os.environ.get("RANK", "0")) == 0:
            os.makedirs(out_dir, exist_ok=True)
            stats = {"command": args.cmd, "status": status, "wall_s": round(_time.time() - t0, 2),
                     "argv": sys.argv, "code_revision": code_revision(), "host": os.uname().nodename}
            if torch.cuda.is_available():
                stats["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                stats["gpu"] = torch.cuda.get_device_name()
            with open(os.path.join(out_dir, "run_stats.jsonl"), "a") as f:
                f.write(json.dumps(stats) + "\n")


if __name__ == "__main__":
    main()
