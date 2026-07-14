"""Tests for stateful token sampler."""

from pathlib import Path

import pytest

from murmur.data.dataset import PackedTokenDataset
from murmur.data.packing import pack_documents
from murmur.data.sampler import StatefulTokenSampler
from murmur.tokenizer import MurmurTokenizer


class TestStatefulSampler:
    """Test sampler state persistence."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """Create dataset for testing."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        tokenizer = MurmurTokenizer.train(
            corpus_path, tmp_path / "tokenizer.model", vocab_size=1000
        )
        docs = ["Document " + str(i) for i in range(20)]
        shard_paths, _, _ = pack_documents(
            documents=docs,
            tokenizer=tokenizer,
            sequence_length=16,
            eos_id=tokenizer.eos_id,
            output_dir=tmp_path / "shards",
        )
        return PackedTokenDataset(shard_paths, sample_length=17)

    def test_sampler_iteration(self, dataset):
        """Sampler iterates over all indices."""
        sampler = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        indices = list(sampler)
        assert len(indices) == len(dataset)

    def test_sampler_state_dict(self, dataset):
        """Sampler state can be saved and restored."""
        sampler = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        iter1 = iter(sampler)
        next(iter1)
        next(iter1)
        state = sampler.state_dict()
        assert "epoch" in state
        assert "cursor" in state
        assert state["cursor"] == 2
        assert state["epoch"] == 0

    def test_sampler_resume(self, dataset):
        """Sampler resumes from saved state."""
        sampler1 = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        iter1 = iter(sampler1)
        next(iter1)
        next(iter1)
        state = sampler1.state_dict()

        sampler2 = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        sampler2.load_state_dict(state)
        iter2 = iter(sampler2)
        next(iter2)
        assert sampler2.cursor == 3

    def test_sampler_deterministic(self, dataset):
        """Same seed produces same order."""
        sampler1 = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        indices1 = list(sampler1)

        sampler2 = StatefulTokenSampler(dataset, batch_size=2, seed=42)
        indices2 = list(sampler2)

        assert indices1 == indices2
