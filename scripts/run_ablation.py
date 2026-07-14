"""Run model parameter/FLOP summaries for R0/R1/M0 configs."""

import argparse
import json
from pathlib import Path

from murmur.config import load_run_config
from murmur.model.language_model import MurmurForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.configs:
        config = load_run_config(path)
        model = MurmurForCausalLM(config.model)
        counts = model.count_parameters()
        rows.append({"config": str(path), "mixer": config.model.mixer, **counts})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
