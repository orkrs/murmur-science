"""Atomic model checkpoints."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file, save_file
    MODEL_FILE = "model.safetensors"
except ImportError:  # local smoke environments may not install optional serialization yet
    MODEL_FILE = "model.pt"
    def save_file(state, path):
        torch.save(state, path)

    def load_file(path, device="cpu"):
        return torch.load(path, map_location=device, weights_only=True)

from murmur.reproducibility import capture_rng_state, restore_rng_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint_atomic(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    metadata: dict,
    extra_state: dict | None = None,
) -> Path:
    """Write a complete checkpoint and publish it with an atomic rename."""
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(tmp / MODEL_FILE))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": capture_rng_state(), "extra": extra_state or {}},
        tmp / "state.pt",
    )
    payload = dict(metadata)
    payload["files"] = {name: _sha256(tmp / name) for name in (MODEL_FILE, "state.pt")}
    (tmp / "meta.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (tmp / "COMPLETED").write_text("ok\n", encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)
    return directory


def load_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra_restore: dict | None = None,
) -> dict:
    """Load a completed checkpoint and optionally restore training state."""
    directory = Path(directory)
    if not (directory / "COMPLETED").exists():
        raise ValueError(f"incomplete checkpoint: {directory}")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    model.load_state_dict(load_file(str(directory / MODEL_FILE), device="cpu"))
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"])
    if extra_restore is not None:
        extra_restore.update(state.get("extra", {}))
    return meta
