"""Record the exact runtime used for a training run."""

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


def collect_hardware_fingerprint() -> dict:
    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update({"gpu": props.name, "compute_capability": [props.major, props.minor], "vram_bytes": props.total_memory, "cuda": torch.version.cuda, "fp16": True, "bf16": bool(torch.cuda.is_bf16_supported())})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_hardware_fingerprint()
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
