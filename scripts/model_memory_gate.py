"""Prove that a complete configured model can execute one worst-depth optimizer step."""

import argparse
import json
from pathlib import Path

import torch

from murmur.config import load_run_config
from murmur.model.language_model import MurmurForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "failed", "config": str(args.config)}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        config = load_run_config(args.config)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = MurmurForCausalLM(config.model).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
        sequence_length = config.data.sequence_length + 1
        tokens = torch.randint(0, config.model.vocab_size, (1, sequence_length), device="cuda")
        depths = torch.full((1,), config.model.max_depth, device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.train.bf16):
            output = model(tokens, labels=tokens, depths=depths)
            loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite full-model loss")
        result = {
            "status": "passed",
            "loss": float(loss),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu": torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        result["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
