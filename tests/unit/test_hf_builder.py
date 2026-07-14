import json
from pathlib import Path

from murmur.data.hf_builder import (
    BuildPlan,
    DatasetSource,
    build_from_iterators,
    normalize_text,
    split_for_validation,
)
from scripts.build_hf_corpus import load_plan


def test_normalize_text_preserves_code_and_removes_control_noise():
    text = "  def f():\r\n\treturn 1\x00  \n"
    assert normalize_text(text) == "def f():\n\treturn 1"


def test_split_for_validation_is_deterministic_and_hash_based():
    first = split_for_validation("same-id", 1000)
    assert first == split_for_validation("same-id", 1000)
    assert first in {"train", "val"}


def test_build_from_iterators_writes_balanced_jsonl_and_provenance(tmp_path: Path):
    sources = [
        DatasetSource(name="en", repo="test/en", weight=0.5, license="odc-by"),
        DatasetSource(name="ru", repo="test/ru", weight=0.5, license="odc-by"),
    ]
    iterators = {
        "en": iter([{"text": "English document one", "id": "en-1", "url": "https://en/1", "quality_score": 0.9}, {"text": "English document two", "id": "en-2"}]),
        "ru": iter([{"text": "Русский документ один", "id": "ru-1"}, {"text": "Русский документ два", "id": "ru-2"}]),
    }
    report = build_from_iterators(
        BuildPlan(sources=sources, output_dir=tmp_path, max_documents=4, val_fraction=0.25),
        iterators,
    )

    assert report["accepted_documents"] == 4
    assert (tmp_path / "corpus.txt").exists()
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    manifest = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["sources"][0]["repo"] == "test/en"
    train = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val = (tmp_path / "val.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) + len(val) == 4
    assert all("source" in json.loads(line) for line in train + val)
    assert any(json.loads(line).get("source_metadata", {}).get("url") == "https://en/1" for line in train + val)


def test_load_plan_reads_toml_and_allows_output_override(tmp_path: Path):
    config = tmp_path / "mix.toml"
    config.write_text(
        '[build]\nmax_documents = 3\n\n[[sources]]\nname = "x"\nrepo = "org/x"\nweight = 1.0\n',
        encoding="utf-8",
    )
    plan = load_plan(config, tmp_path / "override")
    assert plan.max_documents == 3
    assert plan.output_dir == tmp_path / "override"
    assert plan.sources[0].repo == "org/x"
