"""Two kinds of saves: a module snapshot (readout + injector only, for inference / stage hand-off)
and a full resumable checkpoint (snapshot + optimizer + trainer state + RNG). Atomic rename."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from .config import Config
from .model import GuidedModel

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except Exception:  # pragma: no cover
    _st_load = _st_save = None

MODULE_FILE = "modules.safetensors"
COMPLETE_MARKER = "COMPLETE"


def code_revision() -> str:
    """git HEAD; on TACC the code is an rsync'ed copy without .git, so fall back to the .code_revision file
    written by tacc/sync_and_submit_math.sh (suffix '+dirty' when the synced tree had uncommitted changes)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    try:
        rev = open(os.path.join(root, ".code_revision")).read().strip()[:12]
        status = os.path.join(root, ".code_status")
        dirty = os.path.exists(status) and open(status).read().strip() != ""
        return rev + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def _save_modules(path: str, gm: GuidedModel) -> None:
    sd = {k: v.detach().cpu().contiguous() for k, v in gm.module_state_dict().items()}
    if _st_save is not None:
        _st_save(sd, path)
    else:  # pragma: no cover
        torch.save(sd, path + ".pt")


def _load_modules(path: str) -> Dict[str, torch.Tensor]:
    if os.path.exists(path) and _st_load is not None:
        return _st_load(path)
    return torch.load(path + ".pt", map_location="cpu")  # pragma: no cover


def snapshot_manifest(gm: GuidedModel, cfg: Config, stage: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    n_train = {n: p.numel() for n, p in gm.named_parameters() if not n.startswith("base.")}
    m = {
        "stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_revision": code_revision(),
        "model_path": cfg.model.path, "model_revision": cfg.model.revision, "backend": gm.backend.name,
        "backend_info": gm.backend.describe(), "early_layer": cfg.guidance.early_layer,
        "readout_rank": cfg.guidance.readout_rank, "injector_rank": cfg.guidance.injector_rank,
        "d_model": gm.backend.d_model, "vocab_size": gm.backend.vocab_size, "dtype": cfg.model.dtype,
        "module_param_counts": n_train, "module_params_total": sum(n_train.values()),
    }
    if extra:
        m.update(extra)
    return m


def save_snapshot(out_dir: str, gm: GuidedModel, cfg: Config, stage: str, metrics: Optional[Dict[str, Any]] = None,
                  extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_dir)) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".snap_", dir=os.path.dirname(os.path.abspath(out_dir)) or ".")
    try:
        _save_modules(os.path.join(tmp, MODULE_FILE), gm)
        cfg.dump(os.path.join(tmp, "config.json"))
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(snapshot_manifest(gm, cfg, stage, extra), f, indent=1)
        with open(os.path.join(tmp, "metrics.json"), "w") as f:
            json.dump(metrics or {}, f, indent=1)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    finally:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return out_dir


def load_snapshot(snap_dir: str, gm: GuidedModel, cfg: Config, strict: bool = True) -> Dict[str, Any]:
    with open(os.path.join(snap_dir, "manifest.json")) as f:
        man = json.load(f)
    checks = {
        "backend": (man.get("backend"), gm.backend.name),
        "early_layer": (man.get("early_layer"), cfg.guidance.early_layer),
        "d_model": (man.get("d_model"), gm.backend.d_model),
        "vocab_size": (man.get("vocab_size"), gm.backend.vocab_size),
        "readout_rank": (man.get("readout_rank"), cfg.guidance.readout_rank),
        "injector_rank": (man.get("injector_rank"), cfg.guidance.injector_rank),
        "model_path": (man.get("model_path"), cfg.model.path),
        "model_revision": (man.get("model_revision"), cfg.model.revision),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        msg = f"snapshot {snap_dir} does not match the current config: {bad}"
        if strict:
            raise ValueError(msg)
        print("[warn]", msg)
    gm.load_module_state_dict(_load_modules(os.path.join(snap_dir, MODULE_FILE)))
    return man


def rng_state(generators: Optional[Dict[str, torch.Generator]] = None) -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    st["generators"] = {k: g.get_state() for k, g in (generators or {}).items()}
    return st


def set_rng_state(st: Dict[str, Any], generators: Optional[Dict[str, torch.Generator]] = None) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])
    for k, g in (generators or {}).items():
        if k in st.get("generators", {}):
            g.set_state(st["generators"][k])


def save_checkpoint(ckpt_dir: str, gm: GuidedModel, cfg: Config, stage: str, optimizer: torch.optim.Optimizer,
                    trainer_state: Dict[str, Any], generators: Optional[Dict[str, torch.Generator]] = None,
                    rank: int = 0, metrics: Optional[Dict[str, Any]] = None,
                    sampler_state: Optional[Dict[str, Any]] = None) -> str:
    """Collect rank-local state, then let rank 0 publish one complete checkpoint.

    Every distributed rank must participate. No other rank deletes or replaces the directory.
    Save failures are broadcast so peers do not continue with a missing checkpoint.
    """
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    actual_rank = dist.get_rank() if distributed else 0
    if rank != actual_rank:
        raise ValueError(f"checkpoint rank={rank} but process-group rank={actual_rank}")
    local = {"trainer": trainer_state, "rng": rng_state(generators), "sampler": sampler_state}
    states = [None] * world_size if rank == 0 else None
    if distributed:
        dist.gather_object(local, states, dst=0)
    else:
        states[0] = local
    error = [None]
    if rank == 0:
        try:
            _write_checkpoint(ckpt_dir, gm, cfg, stage, optimizer, states, metrics)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"checkpoint save failed: {error[0]}")
    return ckpt_dir


def _write_checkpoint(ckpt_dir, gm, cfg, stage, optimizer, states, metrics):
    parent = os.path.dirname(os.path.abspath(ckpt_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".ckpt_", dir=parent)
    try:
        save_snapshot(os.path.join(tmp, "snapshot"), gm, cfg, stage, metrics)
        torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
        with open(os.path.join(tmp, "trainer_state.json"), "w") as f:
            json.dump(states[0]["trainer"], f, indent=1)  # readable rank-0 summary
        with open(os.path.join(tmp, "checkpoint.json"), "w") as f:
            json.dump({"format_version": 2, "world_size": len(states)}, f)
        for rank, state in enumerate(states):
            with open(os.path.join(tmp, f"trainer_state_rank-{rank}.json"), "w") as f:
                json.dump(state["trainer"], f, indent=1)
            torch.save(state["rng"], os.path.join(tmp, f"rng_state_rank-{rank}.pt"))
            torch.save(state["sampler"], os.path.join(tmp, f"sampler_state_rank-{rank}.pt"))
        with open(os.path.join(tmp, COMPLETE_MARKER), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.replace(tmp, ckpt_dir)
    finally:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return ckpt_dir


def load_checkpoint(ckpt_dir: str, gm: GuidedModel, cfg: Config, optimizer: Optional[torch.optim.Optimizer] = None,
                    generators: Optional[Dict[str, torch.Generator]] = None, rank: int = 0) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(ckpt_dir, COMPLETE_MARKER)):
        raise FileNotFoundError(f"{ckpt_dir} has no {COMPLETE_MARKER} marker (incomplete save)")
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    if rank != (dist.get_rank() if distributed else 0):
        raise ValueError("checkpoint rank does not match the current process group")
    metadata_path = os.path.join(ckpt_dir, "checkpoint.json")
    if not os.path.exists(metadata_path):
        raise ValueError("legacy checkpoint lacks per-rank sampler state; use its snapshot for a fresh run")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("world_size") != world_size:
        raise ValueError("exact resume requires the checkpoint's original world size")
    required = [f"trainer_state_rank-{rank}.json", f"rng_state_rank-{rank}.pt", f"sampler_state_rank-{rank}.pt"]
    for name in required:
        if not os.path.exists(os.path.join(ckpt_dir, name)):
            raise FileNotFoundError(f"checkpoint missing {name}")
    load_snapshot(os.path.join(ckpt_dir, "snapshot"), gm, cfg)
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location="cpu", weights_only=False))
    with open(os.path.join(ckpt_dir, f"trainer_state_rank-{rank}.json")) as f:
        state = json.load(f)
    rng_file = os.path.join(ckpt_dir, f"rng_state_rank-{rank}.pt")
    set_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False), generators)
    state["_sampler_state"] = torch.load(os.path.join(ckpt_dir, f"sampler_state_rank-{rank}.pt"),
                                        map_location="cpu", weights_only=False)
    return state
