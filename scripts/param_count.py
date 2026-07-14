"""Count model parameters."""

import argparse
from pathlib import Path

from murmur.config import load_run_config
from murmur.model.language_model import MurmurForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", type=Path, required=True, help="Path to config TOML")
    args = parser.parse_args()

    config = load_run_config(args.config)
    model = MurmurForCausalLM(config.model)
    counts = model.count_parameters()

    print("Parameter counts:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    total = counts["total_unique"]
    print(f"\nTotal unique parameters: {total:,} ({total / 1e6:.2f}M)")

    if hasattr(config.model, "expected_parameter_range"):
        low, high = config.model.expected_parameter_range
        if low <= total <= high:
            print(f"✓ Within expected range [{low:,}, {high:,}]")
        else:
            print(f"✗ Outside expected range [{low:,}, {high:,}]")


if __name__ == "__main__":
    main()
