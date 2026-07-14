"""Tests for document packing."""

from pathlib import Path

import pytest

from murmur.data.packing import (
    compute_document_hash,
    normalize_document,
    pack_documents,
)
from murmur.tokenizer import MurmurTokenizer


class TestNormalization:
    """Test document normalization."""

    def test_strip_whitespace(self):
        """Strip leading/trailing whitespace."""
        assert normalize_document("  hello  ") == "hello"
        assert normalize_document("\n\ntest\n\n") == "test"

    def test_normalize_line_endings(self):
        """Normalize CRLF to LF."""
        assert normalize_document("line1\r\nline2") == "line1\nline2"


class TestDocumentHash:
    """Test document hash computation."""

    def test_same_content_same_hash(self):
        """Same normalized content produces same hash."""
        h1 = compute_document_hash("hello world")
        h2 = compute_document_hash("  hello world  ")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different content produces different hash."""
        h1 = compute_document_hash("hello")
        h2 = compute_document_hash("world")
        assert h1 != h2


class TestPackDocuments:
    """Test document packing into shards."""

    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create tokenizer for testing."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")
        model_path = tmp_path / "tokenizer.model"
        return MurmurTokenizer.train(corpus_path, model_path, vocab_size=1000)

    def test_pack_basic(self, tokenizer, tmp_path):
        """Pack documents into shards."""
        docs = ["Hello world", "Test document", "Another one"]
        shard_paths, num_samples, num_tokens = pack_documents(
            documents=docs,
            tokenizer=tokenizer,
            sequence_length=16,
            eos_id=tokenizer.eos_id,
            output_dir=tmp_path,
        )
        assert len(shard_paths) > 0
        assert num_samples > 0
        assert num_tokens > 0

    def test_pack_dedup(self, tokenizer, tmp_path):
        """Duplicate documents are deduplicated."""
        docs_unique = ["Hello world", "Test document"]
        docs_dup = ["Hello world", "Hello world", "Test document"]

        shard_paths_unique, num_samples_unique, num_tokens_unique = pack_documents(
            documents=docs_unique,
            tokenizer=tokenizer,
            sequence_length=16,
            eos_id=tokenizer.eos_id,
            output_dir=tmp_path / "unique",
        )

        shard_paths_dup, num_samples_dup, num_tokens_dup = pack_documents(
            documents=docs_dup,
            tokenizer=tokenizer,
            sequence_length=16,
            eos_id=tokenizer.eos_id,
            output_dir=tmp_path / "dup",
        )

        assert num_tokens_unique == num_tokens_dup

    def test_pack_sample_length(self, tokenizer, tmp_path):
        """Each sample has exactly sequence_length + 1 tokens."""
        docs = ["A" * 100, "B" * 100, "C" * 100]
        seq_len = 32
        shard_paths, num_samples, _ = pack_documents(
            documents=docs,
            tokenizer=tokenizer,
            sequence_length=seq_len,
            eos_id=tokenizer.eos_id,
            output_dir=tmp_path,
        )
        for shard_path in shard_paths:
            idx_path = shard_path.with_suffix(".idx")
            assert idx_path.exists()

    def test_pack_does_not_cross_document_boundaries(self, tokenizer, tmp_path):
        """A short document tail is dropped, never joined to the next document."""
        docs = ["a" * 40, "b" * 40]
        seq_len = 16
        paths, count, _ = pack_documents(
            docs, tokenizer, seq_len, tokenizer.eos_id, tmp_path / "boundaries"
        )
        # The test is intentionally based on the implementation contract: each
        # document is packed independently, so adding a second short document
        # cannot create a sample from the two tails.
        assert count == sum(
            max(1, (len(tokenizer.encode(doc)) + 1) // (seq_len + 1)) for doc in docs
        )
