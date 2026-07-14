"""Document packing into fixed-length token sequences."""

import hashlib
import json
from pathlib import Path

import numpy as np

from murmur.tokenizer import MurmurTokenizer


def normalize_document(text: str) -> str:
    """Normalize document for deduplication.

    Strips leading/trailing whitespace and normalizes line endings.
    """
    return text.strip().replace("\r\n", "\n")


def compute_document_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized document."""
    normalized = normalize_document(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def pack_documents(
    documents: list[str],
    tokenizer: MurmurTokenizer,
    sequence_length: int,
    eos_id: int,
    output_dir: Path,
    shard_prefix: str = "shard",
    max_shard_size_mb: int = 100,
) -> tuple[list[Path], int, int]:
    """Pack documents into fixed-length token sequences and write to shards.

    Documents are tokenized, concatenated with EOS tokens between them,
    and split into chunks of exactly `sequence_length + 1` tokens.
    Each chunk becomes one training sample.

    Args:
        documents: List of document texts
        tokenizer: Tokenizer instance
        sequence_length: Target sequence length (samples will have seq_len + 1 tokens)
        eos_id: End-of-sequence token ID
        output_dir: Directory to write shard files
        shard_prefix: Prefix for shard filenames
        max_shard_size_mb: Maximum shard size in MB before rotating

    Returns:
        Tuple of (shard_paths, num_samples, num_tokens)

    Note:
        - Documents are deduplicated by normalized hash before packing
        - EOS token is inserted between documents (not at start/end of shard)
        - Tails of one document are NOT concatenated with heads of another
        - Each sample has exactly sequence_length + 1 tokens
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    unique_docs: list[str] = []
    for doc in documents:
        h = compute_document_hash(doc)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    sample_length = sequence_length + 1
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    shard_paths: list[Path] = []
    current_shard_tokens: list[int] = []
    shard_idx = 0
    samples_written = 0
    total_tokens = 0
    max_shard_bytes = max(1, max_shard_size_mb) * 1024 * 1024

    def flush_shard() -> None:
        nonlocal current_shard_tokens, shard_idx
        if not current_shard_tokens:
            return
        shard_path = output_dir / f"{shard_prefix}_{shard_idx:04d}.bin"
        _write_shard(shard_path, current_shard_tokens, sample_length)
        shard_paths.append(shard_path)
        shard_idx += 1
        current_shard_tokens = []

    for doc in unique_docs:
        tokens = tokenizer.encode(doc)
        tokens.append(eos_id)
        total_tokens += len(tokens)

        # Never cross a document boundary. The unused tail is intentionally
        # dropped instead of being joined to the next document.
        if len(tokens) < sample_length and tokens:
            tokens = tokens + [eos_id] * (sample_length - len(tokens))
        for start in range(0, len(tokens) - sample_length + 1, sample_length):
            sample = tokens[start : start + sample_length]
            current_shard_tokens.extend(sample)
            samples_written += 1
            if len(current_shard_tokens) * 2 >= max_shard_bytes:
                flush_shard()

    flush_shard()
    return shard_paths, samples_written, total_tokens


def _write_shard(path: Path, tokens: list[int], sample_length: int) -> None:
    """Write tokens to binary shard file (little-endian uint16)."""
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(str(path))

    idx_path = path.with_suffix(".idx")
    num_samples = len(tokens) // sample_length
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"num_samples": num_samples, "sample_length": sample_length}))


