"""Immutable dataset manifest for reproducible data pipelines."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable manifest describing a prepared dataset.

    Attributes:
        dataset_name: Human-readable dataset identifier
        revision: Dataset version/revision string
        license: License identifier (e.g., "Apache-2.0")
        source: Source description (e.g., "FineWeb-Edu", "The Stack v2")
        filters: Applied filters (e.g., dedup, language, quality)
        tokenizer_fingerprint: SHA-256 of tokenizer model
        shard_hashes: SHA-256 hashes of each shard file
        num_documents: Total number of documents
        num_tokens: Total number of tokens
        num_accepted: Documents accepted after filtering
        num_rejected: Documents rejected by filters
        created_at: ISO timestamp of creation
    """

    dataset_name: str
    revision: str
    license: str
    source: str
    filters: dict[str, Any]
    tokenizer_fingerprint: str
    shard_hashes: list[str]
    num_documents: int
    num_tokens: int
    num_accepted: int
    num_rejected: int
    created_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def fingerprint(self) -> str:
        """Compute SHA-256 of manifest content (excluding created_at)."""
        content = {
            "dataset_name": self.dataset_name,
            "revision": self.revision,
            "license": self.license,
            "source": self.source,
            "filters": self.filters,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "shard_hashes": self.shard_hashes,
            "num_documents": self.num_documents,
            "num_tokens": self.num_tokens,
            "num_accepted": self.num_accepted,
            "num_rejected": self.num_rejected,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def compute_shard_hash(shard_path: Path) -> str:
    """Compute SHA-256 hash of a shard file."""
    with open(shard_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
