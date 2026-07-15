"""Model configuration dataclasses."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for Murmur-RSM model architecture."""

    d_model: int
    vocab_size: int
    n_prelude: int
    n_core: int
    n_coda: int
    q_heads: int
    kv_heads: int
    head_dim: int
    ffn_dim: int
    max_seq_len: int
    min_depth: int
    max_depth: int
    mixer: Literal["gqa", "mamba3", "mamba3_mimo"] = "gqa"
    mamba_chunk_size: int | None = None
    rope_theta: float = 10000.0
    rope_dim: int | None = None
    residual_scale: float = 1.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration constraints."""
        if self.d_model < 1 or self.q_heads < 1 or self.kv_heads < 1:
            raise ValueError("d_model, q_heads and kv_heads must be positive")
        if self.head_dim < 1 or self.ffn_dim < 1 or self.max_seq_len < 1:
            raise ValueError("head_dim, ffn_dim and max_seq_len must be positive")
        if self.d_model % self.q_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by q_heads ({self.q_heads})"
            )
        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"q_heads ({self.q_heads}) must be divisible by kv_heads ({self.kv_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.rope_dim is not None and (
            self.rope_dim < 2 or self.rope_dim > self.head_dim or self.rope_dim % 2
        ):
            raise ValueError("rope_dim must be a positive even value <= head_dim")
        if self.min_depth < 1:
            raise ValueError(f"min_depth ({self.min_depth}) must be >= 1")
        if self.max_depth < self.min_depth:
            raise ValueError(
                f"max_depth ({self.max_depth}) must be >= min_depth ({self.min_depth})"
            )
        if self.n_prelude < 0 or self.n_core < 0 or self.n_coda < 0:
            raise ValueError("n_prelude, n_core, n_coda must be non-negative")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be >= 1")
        if self.mamba_chunk_size is not None and self.mamba_chunk_size < 8:
            raise ValueError("mamba_chunk_size must be >= 8 when specified")

    @property
    def effective_rope_dim(self) -> int:
        """Return actual RoPE dimension (defaults to head_dim if not specified)."""
        return self.rope_dim if self.rope_dim is not None else self.head_dim

    @property
    def n_total_blocks(self) -> int:
        """Total number of transformer blocks."""
        return self.n_prelude + self.n_core + self.n_coda

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "n_prelude": self.n_prelude,
            "n_core": self.n_core,
            "n_coda": self.n_coda,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "max_seq_len": self.max_seq_len,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "mixer": self.mixer,
            "mamba_chunk_size": self.mamba_chunk_size,
            "rope_theta": self.rope_theta,
            "rope_dim": self.rope_dim,
            "residual_scale": self.residual_scale,
            "norm_eps": self.norm_eps,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**data)
