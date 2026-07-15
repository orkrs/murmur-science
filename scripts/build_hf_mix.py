"""Build the English smoke or the mixed 350M corpus from public HF datasets.

The builder streams records, samples sources by token-mix weights, normalizes chat
and plain-text records into one causal-LM format, and writes JSONL/corpus files
incrementally so a RunPod session does not need to hold the corpus in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from murmur.data.hf_builder import normalize_text, split_for_validation


@dataclass(frozen=True)
class MixSource:
    name: str
    repo: str
    weight: int
    language: str
    config: str | None = None
    split: str = "train"
    revision: str = "main"
    license: str = "verify-source-card"


ENGLISH_SMOKE_SOURCES = [
    MixSource(
        name="fineweb_edu",
        repo="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        weight=100,
        language="en",
        license="odc-by-1.0+commoncrawl-terms",
    )
]

MIXED_350M_SOURCES = [
    MixSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", 45, "en", "sample-10BT", license="odc-by-1.0+commoncrawl-terms"),
    MixSource("ru_big", "ZeroAgency/ru-big-russian-dataset", 15, "ru", license="mit+upstream-terms"),
    MixSource("t_wix", "t-tech/T-Wix", 8, "ru", license="odc-by-1.0+upstream-terms"),
    MixSource("ling_coder", "inclusionAI/Ling-Coder-SyntheticQA", 12, "code", license="apache-2.0"),
    MixSource("open_code", "nvidia/OpenCodeReasoning", 8, "code", "split_0", license="cc-by-4.0"),
    MixSource("openr1_math", "open-r1/OpenR1-Math-220k", 7, "math", "default", license="apache-2.0"),
    MixSource("synthetic_code", "open-r1/SYNTHETIC-1-SFT-Data-Code_decontaminated", 2, "code", license="verify-source-card"),
    MixSource("deepcoder", "agentica-org/DeepCoder-Preview-Dataset", 2, "code", license="mit"),
    MixSource("openhermes_ru", "d0rj/OpenHermes-2.5-ru", 1, "ru", license="apache-2.0+translation-terms"),
]


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content(item) for item in value)
    if isinstance(value, Mapping):
        return _content(value.get("content", value.get("text", "")))
    return ""


def extract_record_text(record: Mapping[str, Any]) -> str:
    """Convert common HF text/chat/code schemas to causal-LM training text."""

    for key in ("text", "content", "document", "body"):
        value = _content(record.get(key))
        if value.strip():
            return value
    for key in ("messages", "conversation", "conversations"):
        messages = record.get(key)
        if isinstance(messages, list):
            parts = []
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                role = str(message.get("role", message.get("from", "user")))
                content = _content(message.get("content", message.get("value", ""))).strip()
                if content:
                    parts.append(f"<|{role}|>\n{content}")
            if parts:
                return "\n".join(parts)
    for fields in (("problem", "solution"), ("problem", "answer"), ("input", "output"), ("prompt", "completion")):
        values = [_content(record.get(field)).strip() for field in fields]
        if all(values):
            return f"<|user|>\n{values[0]}\n<|assistant|>\n{values[1]}"
    return ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _schedule(sources: list[MixSource]) -> list[str]:
    return [source.name for source in sources for _ in range(source.weight)]


def write_mix(
    output_dir: Path,
    *,
    sources: list[MixSource],
    iterators: Mapping[str, Iterator[Mapping[str, Any]]],
    max_tokens: int,
    max_documents: int | None = None,
    val_fraction: float = 0.02,
    seed: int = 17,
) -> dict[str, Any]:
    """Write a deterministic, weighted, streaming corpus to ``output_dir``."""

    if max_tokens < 1 or not sources:
        raise ValueError("max_tokens and sources are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_name = {source.name: source for source in sources}
    active = {name: iterator for name, iterator in iterators.items() if name in source_by_name}
    schedule = _schedule(sources)
    stats = {source.name: {"seen": 0, "accepted": 0, "tokens": 0} for source in sources}
    seen: set[str] = set()
    total_tokens = 0
    documents = 0
    schedule_index = 0
    handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "corpus": (output_dir / "corpus.txt").open("w", encoding="utf-8"),
    }
    try:
        while active and total_tokens < max_tokens:
            if max_documents is not None and documents >= max_documents:
                break
            name = schedule[schedule_index % len(schedule)]
            schedule_index += 1
            if name not in active:
                continue
            source = source_by_name[name]
            try:
                record = next(active[name])
            except StopIteration:
                del active[name]
                continue
            stats[name]["seen"] += 1
            text = normalize_text(extract_record_text(record))
            if len(text) < 40 or len(text) > 2_000_000:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            estimate = _estimate_tokens(text)
            if total_tokens + estimate > max_tokens:
                continue
            seen.add(digest)
            source_id = str(record.get("id", record.get("uuid", digest)))
            split = split_for_validation(f"{seed}:{source.name}:{source_id}", val_fraction)
            row = {
                "text": text,
                "source": source.name,
                "source_id": source_id,
                "repo": source.repo,
                "revision": source.revision,
                "license": source.license,
                "language": source.language,
                "text_sha256": digest,
            }
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            handles["corpus"].write(text + "\n\n")
            stats[name]["accepted"] += 1
            stats[name]["tokens"] += estimate
            total_tokens += estimate
            documents += 1
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "format_version": "murmur-hf-mix-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_tokens_estimate": max_tokens,
        "accepted_documents": documents,
        "estimated_tokens": total_tokens,
        "sources": [asdict(source) for source in sources],
        "stats": stats,
    }
    (output_dir / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_iterators(sources: list[MixSource]) -> dict[str, Iterator[Mapping[str, Any]]]:
    from datasets import load_dataset

    iterators = {}
    for source in sources:
        kwargs = {"split": source.split, "streaming": True, "revision": source.revision}
        dataset = load_dataset(source.repo, source.config, **kwargs) if source.config else load_dataset(source.repo, **kwargs)
        iterators[source.name] = iter(dataset)
    return iterators


def build_profile(profile: str, output_dir: Path, max_tokens: int) -> dict[str, Any]:
    if profile == "english_smoke":
        sources = ENGLISH_SMOKE_SOURCES
    elif profile == "mixed_350m":
        sources = MIXED_350M_SOURCES
    else:
        raise ValueError(f"unknown profile: {profile}")
    return write_mix(output_dir, sources=sources, iterators=_load_iterators(sources), max_tokens=max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("english_smoke", "mixed_350m"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    report = build_profile(args.profile, args.output, args.max_tokens)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
