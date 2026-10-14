"""python -m selective_draft.cli <command> --config CONFIG [--set section.field=value ...]

commands: prepare-data | inspect | collect | train-readout | diagnose | train-guidance | evaluate
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
from .data import load_split, prepare_manifest, tokenize_prompts, parse_prediction, is_correct
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
    if args.device != "auto":
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "boundary_module": type(gm.backend.boundary_module(cfg.guidance.early_layer)).__name__,
        "early_layer_block_index": cfg.guidance.early_layer - 1,
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
    _dump_resolved(cfg, args.output_dir, {"command": "collect", "snapshot": snap_id, "n_examples": len(examples)})
    device = gm.device
    bs = cfg.evaluation.batch_size
    rng = torch.Generator(device=device).manual_seed(cfg.training.seed)
    all_summ, shard = [], 0
    for s in range(0, len(examples), bs):
        batch = examples[s:s + bs]
        ids, mask = tokenize_prompts(tok, [e.prompt_text for e in batch], device)
        metas = [{"example_id": e.example_id, "prompt_ids": ids[i].tolist(), "prompt_attention_mask": mask[i].tolist(),
                  "gold": e.answer_value, "guidance_mode": cfg.guidance.mode, "snapshot_id": snap_id,
                  "generation": cfg.generation.__dict__, "split": cfg.data.split} for i, e in enumerate(batch)]
        rec = StateRecorder(level, gm.backend.mask_token_id, gm.backend.eos_token_id, ids.shape[1], metas, topk=cfg.trace.topk)
        res = generate(gm, ids, mask, cfg.generation, guidance_mode=cfg.guidance.mode, recorder=rec, sample_rng=rng)
        extras = []
        for i, e in enumerate(batch):
            text = tok.decode(res.completion_ids(i), skip_special_tokens=True)
            pred = parse_prediction(text)
            extras.append({"generation": text, "parsed_answer": pred, "is_correct": is_correct(pred, e.answer_value),
                           "stats": res.stats})
        rolls = rec.finalize(res, extras)
        for r in rolls:
            replay_check(r)
        path = os.path.join(args.output_dir, f"rollouts-{shard:05d}.pt")
        save_rollouts(path, rolls)
        shard += 1
        all_summ.extend(rolls)
        print(f"[collect] {s + len(batch)}/{len(examples)} -> {path} nfe={res.stats['nfe']} "
              f"correct={sum(r['is_correct'] for r in rolls)}/{len(rolls)}")
    summ = summarize_rollouts(all_summ)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
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
    gm, tok, snap_id = _build(cfg, args, args.readout_snapshot)
    if not args.readout_snapshot and not args.resume:
        raise SystemExit("train-guidance needs --readout-snapshot (stage 1 output) or --resume")
    examples = load_split(cfg, tok)
    _dump_resolved(cfg, args.output_dir, {"command": "train-guidance", "readout_snapshot": snap_id, "n_examples": len(examples)})
    trainer = GuidanceTrainer(gm, cfg, args.output_dir, examples, tok)
    last = run_training(trainer, args.max_steps or cfg.training.max_steps, args.save_every or cfg.training.save_every, args.resume)
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
    _dump_resolved(cfg, args.output_dir, {"command": "evaluate", "snapshot": snap_id, "n_examples": len(examples)})
    out = evaluate(gm, tok, examples, cfg, cfg.guidance.mode, args.output_dir, snapshot_id=snap_id,
                   trace_level=cfg.trace.level)
    print(json.dumps(out["summaries"], indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="selective_draft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="write the sample-ID split manifest (never overwrites without --force)")
    _common(p); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_prepare_data)

    p = sub.add_parser("inspect", help="print backend layout, token ids, trainable names, resolved config")
    _common(p, output=False); p.add_argument("--snapshot"); p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("collect", help="decode the configured split and save CPU state shards")
    _common(p); p.add_argument("--snapshot"); p.add_argument("--limit", type=int); p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("train-readout", help="stage 1: train g_phi on recorded base states")
    _common(p); p.add_argument("--states", nargs="+", required=True); p.add_argument("--snapshot")
    p.add_argument("--max-steps", type=int); p.add_argument("--save-every", type=int); p.add_argument("--resume")
    p.set_defaults(fn=cmd_train_readout)

    p = sub.add_parser("diagnose", help="readout | teacher | consistency reports on recorded states")
    _common(p); p.add_argument("--kind", required=True, choices=["readout", "teacher", "consistency"])
    p.add_argument("--states", nargs="+", required=True); p.add_argument("--snapshot")
    p.add_argument("--max-states", type=int); p.add_argument("--no-oracle", action="store_true")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("train-guidance", help="stage 2: on-policy training of J_psi with a frozen readout")
    _common(p); p.add_argument("--readout-snapshot"); p.add_argument("--max-steps", type=int)
    p.add_argument("--save-every", type=int); p.add_argument("--resume"); p.set_defaults(fn=cmd_train_guidance)

    p = sub.add_parser("evaluate", help="task accuracy + latency; threshold sweep via evaluation.commit_thresholds")
    _common(p); p.add_argument("--snapshot"); p.add_argument("--limit", type=int); p.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
