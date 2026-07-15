import json
from pathlib import Path

from scripts.build_hf_mix import (
    ENGLISH_SMOKE_SOURCES,
    MIXED_350M_SOURCES,
    extract_record_text,
    write_mix,
)


def test_profiles_have_expected_language_and_token_weights():
    assert [source.name for source in ENGLISH_SMOKE_SOURCES] == ["fineweb_edu"]
    assert ENGLISH_SMOKE_SOURCES[0].config == "sample-10BT"
    assert sum(source.weight for source in MIXED_350M_SOURCES) == 100
    assert {source.language for source in MIXED_350M_SOURCES} >= {"en", "ru", "code"}


def test_extract_record_text_supports_plain_text_and_chat_messages():
    assert extract_record_text({"text": "hello"}) == "hello"
    assert extract_record_text(
        {"messages": [{"role": "user", "content": "Задача"}, {"role": "assistant", "content": "Ответ"}]}
    ) == "<|user|>\nЗадача\n<|assistant|>\nОтвет"
    assert extract_record_text({"content": "code"}) == "code"


def test_write_mix_produces_provenance_and_stable_splits(tmp_path: Path):
    iterators = {
        "fineweb_edu": iter([{"text": "An English educational document with enough content for the smoke builder."}]),
        "ru_big": iter([{"conversation": [{"role": "user", "content": "Русский вопрос"}, {"role": "assistant", "content": "Русский ответ"}]}]),
    }
    sources = [
        next(source for source in MIXED_350M_SOURCES if source.name == "fineweb_edu"),
        next(source for source in MIXED_350M_SOURCES if source.name == "ru_big"),
    ]
    report = write_mix(tmp_path, sources=sources, iterators=iterators, max_tokens=1000, val_fraction=0.5)
    assert report["accepted_documents"] == 2
    assert (tmp_path / "corpus.txt").exists()
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    rows += [json.loads(line) for line in (tmp_path / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["source"] in {"fineweb_edu", "ru_big"} for row in rows)
