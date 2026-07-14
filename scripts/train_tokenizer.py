"""Train SentencePiece BPE tokenizer."""

import argparse
import hashlib
import json
from pathlib import Path

from murmur.tokenizer import MurmurTokenizer


def compute_corpus_hash(corpus_path: Path) -> str:
    """Compute SHA-256 hash of corpus file."""
    with open(corpus_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train Murmur tokenizer")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to training corpus")
    parser.add_argument("--output", type=Path, required=True, help="Output path for .model file")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    parser.add_argument(
        "--character-coverage", type=float, default=1.0, help="Character coverage"
    )
    args = parser.parse_args()

    print(f"Training tokenizer on {args.corpus}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  character_coverage: {args.character_coverage}")

    tokenizer = MurmurTokenizer.train(
        corpus_path=args.corpus,
        model_path=args.output,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    print("\nTokenizer trained successfully:")
    print(f"  model_path: {args.output}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  fingerprint: {tokenizer.fingerprint()}")

    corpus_hash = compute_corpus_hash(args.corpus)
    print(f"  corpus_hash: {corpus_hash}")

    metadata = {
        "corpus": str(args.corpus),
        "corpus_hash": corpus_hash,
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "fingerprint": tokenizer.fingerprint(),
    }

    metadata_path = args.output.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
