"""Build Murmur corpus artifacts from Hugging Face streaming datasets.

Usage:
    python scripts/build_hf_corpus.py --config configs/data/base_mix_v0_1.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from murmur.data.hf_builder import BuildPlan, DatasetSource, build_from_huggingface


def load_plan(path: Path, output_override: Path | None = None) -> BuildPlan:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    build = config.get("build", {})
    output_dir = output_override or Path(build.get("output_dir", "artifacts/hf_corpus"))
    sources = [DatasetSource(**item) for item in config.get("sources", [])]
    return BuildPlan(
        sources=sources,
        output_dir=output_dir,
        max_documents=build.get("max_documents"),
        max_tokens=build.get("max_tokens"),
        val_fraction=float(build.get("val_fraction", 0.02)),
        min_chars=int(build.get("min_chars", 20)),
        max_chars=int(build.get("max_chars", 2_000_000)),
        seed=int(build.get("seed", 17)),
        write_corpus=bool(build.get("write_corpus", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_from_huggingface(load_plan(args.config, args.output))
    print(
        "Built corpus: "
        f"{report['accepted_documents']} docs, "
        f"{report['estimated_tokens']} estimated tokens, "
        f"train={report['train_documents']}, val={report['val_documents']}"
    )


if __name__ == "__main__":
    main()
