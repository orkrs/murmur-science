"""Tests for byte-fallback tokenizer."""

from pathlib import Path

import pytest

from murmur.tokenizer import MurmurTokenizer


class TestTokenizerRoundTrip:
    """Test exact round-trip encoding/decoding."""

    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        model_path = tmp_path / "tokenizer.model"
        tokenizer = MurmurTokenizer.train(
            corpus_path=corpus_path,
            model_path=model_path,
            vocab_size=1000,
        )
        return tokenizer

    def test_english_roundtrip(self, tokenizer):
        """English text round-trips exactly."""
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_russian_roundtrip(self, tokenizer):
        """Russian text round-trips exactly."""
        text = "Привет, мир! Это тест русского языка."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_code_roundtrip(self, tokenizer):
        """Python code round-trips exactly."""
        text = 'def hello():\n    print("Hello, World!")\n    return 42'
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_whitespace_roundtrip(self, tokenizer):
        """Whitespace (spaces, tabs, newlines) round-trips exactly."""
        text = "  \t\n  multiple   spaces\t\ttabs\n\nnewlines"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_emoji_roundtrip(self, tokenizer):
        """Emoji round-trips exactly."""
        text = "🎉 🚀 💻 🌍"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_combining_marks_roundtrip(self, tokenizer):
        """Combining marks (accents) round-trips exactly."""
        text = "café naïve résumé"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_arbitrary_utf8_roundtrip(self, tokenizer):
        """Arbitrary valid UTF-8 strings round-trip exactly."""
        texts = [
            "π ≈ 3.14159",
            "√2 ≈ 1.414",
            "∫ f(x) dx = F(x) + C",
            "Москва — столица России",
            "温度は25°Cです",
        ]
        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            assert decoded == text, f"Failed for: {text}"

    def test_empty_string_roundtrip(self, tokenizer):
        """Empty string round-trips exactly."""
        text = ""
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text


class TestTokenizerSpecialTokens:
    """Test special token IDs."""

    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        model_path = tmp_path / "tokenizer.model"
        tokenizer = MurmurTokenizer.train(
            corpus_path=corpus_path,
            model_path=model_path,
            vocab_size=1000,
        )
        return tokenizer

    def test_special_token_ids(self, tokenizer):
        """Special tokens have fixed IDs."""
        assert tokenizer.unk_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        assert tokenizer.pad_id == 3

    def test_vocab_size_within_uint16(self, tokenizer):
        """Vocabulary size fits in uint16."""
        assert tokenizer.vocab_size <= 65535


class TestTokenizerFingerprint:
    """Test tokenizer fingerprint for reproducibility."""

    def test_fingerprint_stability(self, tmp_path):
        """Same tokenizer instance produces same fingerprint."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        model_path = tmp_path / "tokenizer.model"

        tokenizer = MurmurTokenizer.train(
            corpus_path=corpus_path,
            model_path=model_path,
            vocab_size=1000,
        )

        # Same tokenizer instance should produce same fingerprint
        fingerprint1 = tokenizer.fingerprint()
        fingerprint2 = tokenizer.fingerprint()
        assert fingerprint1 == fingerprint2

        # Note: SentencePiece training is not fully deterministic across runs,
        # so we only test that the same trained model produces consistent fingerprints.
        # For production, use a pre-trained model file with a fixed fingerprint.

    def test_fingerprint_format(self, tmp_path):
        """Fingerprint is SHA-256 hex string."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        model_path = tmp_path / "tokenizer.model"
        tokenizer = MurmurTokenizer.train(
            corpus_path=corpus_path,
            model_path=model_path,
            vocab_size=1000,
        )

        fingerprint = tokenizer.fingerprint()
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestTokenizerVocabSize:
    """Test vocabulary size constraints."""

    def test_reject_vocab_exceeding_uint16(self, tmp_path):
        """Reject vocabulary size exceeding uint16."""
        corpus_path = Path("tests/fixtures/tokenizer_corpus.txt")
        if not corpus_path.exists():
            pytest.skip("tokenizer_corpus.txt not found")

        model_path = tmp_path / "tokenizer.model"
        with pytest.raises(ValueError, match="vocab_size.*uint16"):
            MurmurTokenizer.train(
                corpus_path=corpus_path,
                model_path=model_path,
                vocab_size=70000,
            )
