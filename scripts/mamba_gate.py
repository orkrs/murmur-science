"""Select a Mamba-3 MIMO kernel that completes forward *and* backward."""

import argparse
import json
from pathlib import Path

import torch

from murmur.model.mixers.mamba3 import MambaUnavailableError


PROFILES = {
    "rtx4090_smoke": {"d_model": 512, "sequence_length": 512, "rank": 2, "chunks": (32, 16, 8)},
    "rtx4090_350m": {"d_model": 1792, "sequence_length": 1024, "rank": 2, "chunks": (32, 16, 8)},
}


def _run_candidate(spec: dict, chunk_size: int) -> dict:
    """Run one actual BF16 MIMO forward/backward and return its measurements."""
    from mamba_ssm.modules.mamba3 import Mamba3

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Mamba3(
        d_model=spec["d_model"], d_state=128, headdim=64,
        is_mimo=True, mimo_rank=spec["rank"], chunk_size=chunk_size,
        dtype=torch.bfloat16,
    ).cuda().train()
    x = torch.randn(
        1, spec["sequence_length"], spec["d_model"], device="cuda",
        dtype=torch.bfloat16, requires_grad=True,
    )
    output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite MIMO loss")
    result = {
        "chunk_size": chunk_size,
        "loss": float(loss),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    del model, x, output, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"cuda": torch.cuda.is_available(), "profile": args.profile, "status": "unsupported", "attempts": []}
    try:
        if not torch.cuda.is_available():
            raise MambaUnavailableError("CUDA is unavailable")
        gpu = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        if "RTX 4090" not in gpu or capability != (8, 9):
            raise RuntimeError(f"expected RTX 4090 (sm_89), received {gpu} {capability}")
        spec = PROFILES[args.profile]
        result.update({"gpu": gpu, "compute_capability": list(capability), "spec": spec})
        for chunk_size in spec["chunks"]:
            try:
                attempt = _run_candidate(spec, chunk_size)
                result["attempts"].append({"status": "passed", **attempt})
                result.update({"status": "passed", "selected_chunk_size": chunk_size, **attempt})
                break
            except Exception as exc:
                result["attempts"].append({"status": "failed", "chunk_size": chunk_size, "error": str(exc)})
        if result["status"] != "passed":
            raise RuntimeError("no supported MIMO chunk size completed backward")
    except Exception as exc:
        result["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
