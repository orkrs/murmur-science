"""Prepare packed train/validation shards from JSONL documents."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from murmur.config import load_run_config
from murmur.data.manifest import DatasetManifest, compute_shard_hash
from murmur.data.packing import pack_documents
from murmur.tokenizer import MurmurTokenizer


def read_documents(path: Path) -> list[str]:
    docs = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            docs.append(value if isinstance(value, str) else value["text"])
    return docs


def prepare(config_path: Path, tokenizer_path: Path, train_input: Path, val_input: Path, output: Path) -> None:
    config = load_run_config(config_path)
    tokenizer = MurmurTokenizer(tokenizer_path)
    output.mkdir(parents=True, exist_ok=True)
    for split, source in (("train", train_input), ("val", val_input)):
        paths, samples, tokens = pack_documents(
            read_documents(source), tokenizer, config.data.sequence_length,
            tokenizer.eos_id, output, shard_prefix=split,
        )
        manifest = DatasetManifest(
            dataset_name=source.stem, revision="local", license="user-supplied",
            source=str(source), filters={"dedup": "sha256", "split": split},
            tokenizer_fingerprint=tokenizer.fingerprint(),
            shard_hashes=[compute_shard_hash(path) for path in paths],
            num_documents=len(read_documents(source)), num_tokens=tokens,
            num_accepted=samples, num_rejected=0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest.to_json(output / f"{split}.manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.tokenizer, args.train_input, args.val_input, args.output)
