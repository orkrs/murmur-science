"""GQA sequence mixer."""

import torch
import torch.nn as nn

from murmur.model.attention import GQAAttention


class GQAMixer(nn.Module):
    """GQA attention as a sequence mixer.

    Wraps GQAAttention to conform to SequenceMixer protocol.
    """

    def __init__(
        self,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        """Initialize GQA mixer.

        Args:
            d_model: Model dimension
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length
            rope_theta: RoPE base frequency
        """
        super().__init__()
        self.attention = GQAAttention(
            d_model, q_heads, kv_heads, head_dim, max_seq_len, rope_theta
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: dict | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            positions: Position indices
            cache: Optional cache dict with 'kv' key
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, new_cache)
        """
        kv_cache = cache.get("kv") if cache else None
        out, new_kv = self.attention(x, positions, kv_cache, use_cache)
        new_cache = {"kv": new_kv} if use_cache else None
        return out, new_cache
