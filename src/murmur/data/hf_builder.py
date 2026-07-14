"""Streaming Hugging Face corpus builder for Murmur.

The builder intentionally keeps the Hugging Face dependency lazy: unit tests and
offline preprocessing can use plain iterators, while Kaggle installs ``datasets``
and calls :func:`build_from_huggingface`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class DatasetSource:
    """One streaming dataset source and its sampling metadata."""

    name: str
    repo: str
    config: str | None = None
    split: str = "train"
    revision: str = "main"
    text_field: str = "text"
    weight: float = 1.0
    license: str = "unknown"
    language: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.repo.strip():
            raise ValueError("source name and repo must be non-empty")
        if self.weight <= 0:
            raise ValueError("source weight must be positive")


@dataclass(frozen=True)
class BuildPlan:
    """Deterministic build parameters."""

    sources: list[DatasetSource]
    output_dir: Path
    max_documents: int | None = None
    max_tokens: int | None = None
    val_fraction: float = 0.02
    min_chars: int = 20
    max_chars: int = 2_000_000
    seed: int = 17
    write_corpus: bool = True

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one source is required")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValueError("max_documents must be positive")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise ValueError("invalid character limits")


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(value: str) -> str:
    """Normalize text without damaging indentation, Markdown or LaTeX."""

    if not isinstance(value, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = _CONTROL_RE.sub("", value)
    lines = [line.rstrip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def split_for_validation(identifier: str, val_fraction: float | int) -> str:
    """Return a stable split based only on an identifier and fraction.

    ``val_fraction`` may be a float in ``[0, 1)`` or an integer in basis points
    (1000 means 10%), which makes the helper convenient for config files.
    """

    fraction = val_fraction / 10000 if isinstance(val_fraction, int) else val_fraction
    if not 0 <= fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return "val" if bucket < int(fraction * 10000) else "train"


def _record_identifier(record: Mapping[str, Any], text: str, source: DatasetSource) -> str:
    for key in ("id", "doc_id", "uuid", "url", "path", "blob_id"):
        value = record.get(key)
        if value:
            return f"{source.name}:{value}"
    return f"{source.name}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _estimate_tokens(text: str) -> int:
    # A stable pre-tokenizer estimate; exact counts are recorded later by packer.
    return max(1, len(text.encode("utf-8")) // 4)


def _source_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep useful provenance fields without copying large dataset payloads."""

    keys = (
        "id",
        "doc_id",
        "uuid",
        "url",
        "path",
        "blob_id",
        "date",
        "quality_score",
        "score",
        "language",
        "license",
    )
    metadata: dict[str, Any] = {}
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
        metadata[key] = value
    return metadata


def _weighted_schedule(sources: list[DatasetSource]) -> list[str]:
    scale = 100
    slots: list[str] = []
    for source in sources:
        slots.extend([source.name] * max(1, round(source.weight * scale)))
    return slots


def build_from_iterators(
    plan: BuildPlan,
    iterators: Mapping[str, Iterator[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build JSONL/corpus/provenance files from already-created iterators."""

    output = Path(plan.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_by_name = {source.name: source for source in plan.sources}
    active = {name: iterator for name, iterator in iterators.items() if name in source_by_name}
    schedule = _weighted_schedule(plan.sources)
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {
        source.name: {"seen": 0, "accepted": 0, "rejected": 0, "tokens": 0}
        for source in plan.sources
    }
    total_tokens = 0
    schedule_index = 0

    while active:
        if plan.max_documents is not None and len(train_records) + len(val_records) >= plan.max_documents:
            break
        if plan.max_tokens is not None and total_tokens >= plan.max_tokens:
            break
        name = schedule[schedule_index % len(schedule)]
        schedule_index += 1
        if name not in active:
            if not any(candidate in active for candidate in schedule):
                break
            continue
        source = source_by_name[name]
        try:
            raw = next(active[name])
        except StopIteration:
            del active[name]
            continue
        stats[name]["seen"] += 1
        if not isinstance(raw, Mapping):
            stats[name]["rejected"] += 1
            continue
        value = raw.get(source.text_field)
        if not isinstance(value, str):
            stats[name]["rejected"] += 1
            continue
        try:
            text = normalize_text(value)
        except (TypeError, ValueError):
            stats[name]["rejected"] += 1
            continue
        if len(text) < plan.min_chars or len(text) > plan.max_chars:
            stats[name]["rejected"] += 1
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen:
            stats[name]["rejected"] += 1
            continue
        seen.add(content_hash)
        identifier = _record_identifier(raw, text, source)
        estimated_tokens = _estimate_tokens(text)
        record = {
            "text": text,
            "source": source.name,
            "source_id": identifier,
            "repo": source.repo,
            "revision": source.revision,
            "license": source.license,
            "language": source.language,
            "text_sha256": content_hash,
            "source_metadata": _source_metadata(raw),
        }
        target = split_for_validation(f"{plan.seed}:{identifier}", plan.val_fraction)
        (val_records if target == "val" else train_records).append(record)
        stats[name]["accepted"] += 1
        stats[name]["tokens"] += estimated_tokens
        total_tokens += estimated_tokens

    _write_jsonl(output / "train.jsonl", train_records)
    _write_jsonl(output / "val.jsonl", val_records)
    if plan.write_corpus:
        with (output / "corpus.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for record in train_records + val_records:
                handle.write(record["text"] + "\n\n")

    provenance = {
        "format_version": "murmur-hf-corpus-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": {
            "max_documents": plan.max_documents,
            "max_tokens_estimate": plan.max_tokens,
            "val_fraction": plan.val_fraction,
            "min_chars": plan.min_chars,
            "max_chars": plan.max_chars,
            "seed": plan.seed,
        },
        "sources": [asdict(source) for source in plan.sources],
        "stats": stats,
        "accepted_documents": len(train_records) + len(val_records),
        "train_documents": len(train_records),
        "val_documents": len(val_records),
        "estimated_tokens": total_tokens,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_hf_iterators(plan: BuildPlan) -> dict[str, Iterator[Mapping[str, Any]]]:
    """Create lazy Hugging Face streaming iterators for a build plan."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Install the 'datasets' package to use Hugging Face loading") from exc
    result: dict[str, Iterator[Mapping[str, Any]]] = {}
    for source in plan.sources:
        kwargs: dict[str, Any] = {
            "split": source.split,
            "streaming": True,
            "revision": source.revision,
        }
        dataset = (
            load_dataset(source.repo, source.config, **kwargs)
            if source.config
            else load_dataset(source.repo, **kwargs)
        )
        result[source.name] = iter(dataset)
    return result


def build_from_huggingface(plan: BuildPlan) -> dict[str, Any]:
    """Download/stream configured sources and write Murmur artifacts."""

    return build_from_iterators(plan, load_hf_iterators(plan))
