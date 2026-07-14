"""Byte-fallback BPE tokenizer using SentencePiece."""

import hashlib
from pathlib import Path

import sentencepiece as spm


class MurmurTokenizer:
    """SentencePiece BPE tokenizer with byte fallback.

    Special tokens:
        0: <unk> (unknown)
        1: <s> (beginning of sequence)
        2: </s> (end of sequence)
        3: <pad> (padding)
    """

    def __init__(self, model_path: Path):
        """Load tokenizer from model file.

        Args:
            model_path: Path to .model file
        """
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))

    @classmethod
    def train(
        cls,
        corpus_path: Path,
        model_path: Path,
        vocab_size: int,
        character_coverage: float = 1.0,
    ) -> "MurmurTokenizer":
        """Train tokenizer on corpus.

        Args:
            corpus_path: Path to training corpus (text file)
            model_path: Output path for .model file
            vocab_size: Target vocabulary size
            character_coverage: Character coverage for training

        Returns:
            Trained MurmurTokenizer

        Raises:
            ValueError: If vocab_size exceeds uint16 range
        """
        if vocab_size > 65535:
            raise ValueError(
                f"vocab_size ({vocab_size}) exceeds uint16 range (max 65535)"
            )

        corpus_path = Path(corpus_path)
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_path.with_suffix("")),
            vocab_size=vocab_size,
            model_type="bpe",
            byte_fallback=True,
            split_by_whitespace=False,
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            character_coverage=character_coverage,
            normalization_rule_name="identity",
            shuffle_input_sentence=False,
            seed_sentencepiece_size=1000000,
            num_threads=1,
        )

        return cls(model_path)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.sp.Encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        return self.sp.Decode(ids)

    def fingerprint(self) -> str:
        """Compute SHA-256 fingerprint of model file.

        Returns:
            Hex-encoded SHA-256 hash
        """
        with open(self.model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.sp.GetPieceSize()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return 0

    @property
    def bos_id(self) -> int:
        """Beginning of sequence token ID."""
        return 1

    @property
    def eos_id(self) -> int:
        """End of sequence token ID."""
        return 2

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return 3
