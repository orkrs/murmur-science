"""Run a small official Mamba 3 compatibility gate."""

import argparse
import json
from pathlib import Path

import torch

from murmur.model.mixers.mamba3 import Mamba3Block, MambaUnavailableError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"cuda": torch.cuda.is_available(), "status": "unsupported"}
    try:
        if not torch.cuda.is_available():
            raise MambaUnavailableError("CUDA is unavailable")
        device = torch.device("cuda")
        block = Mamba3Block(128, 256, 64, 32, 1e-5, 1.0, 0).to(device).half()
        x = torch.randn(2, 16, 128, device=device, dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            y, _ = block(x)
        if not torch.isfinite(y).all():
            raise RuntimeError("non-finite Mamba output")
        result.update({"status": "passed", "gpu": torch.cuda.get_device_name(0), "shape": list(y.shape)})
    except Exception as exc:
        result["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
