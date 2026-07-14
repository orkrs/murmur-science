"""Transformer block."""

import torch
import torch.nn as nn

from murmur.model.attention import GQAAttention
from murmur.model.mlp import SwiGLU
from murmur.model.norms import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention and SwiGLU MLP.

    Structure:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))
    """

    def __init__(
        self,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
    ):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Dimension per head
            ffn_dim: FFN intermediate dimension
            max_seq_len: Maximum sequence length
            rope_theta: RoPE base frequency
            norm_eps: Normalization epsilon
            residual_scale: Residual connection scaling factor
        """
        super().__init__()
        self.residual_scale = residual_scale

        self.attn_norm = RMSNorm(d_model, norm_eps)
        self.attention = GQAAttention(
            d_model, q_heads, kv_heads, head_dim, max_seq_len, rope_theta
        )

        self.mlp_norm = RMSNorm(d_model, norm_eps)
        self.mlp = SwiGLU(d_model, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            positions: Position indices for RoPE
            kv_cache: Optional cached (k, v)
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, new_cache)
        """
        attn_out, new_cache = self.attention(
            self.attn_norm(x), positions, kv_cache, use_cache
        )
        x = x + attn_out * self.residual_scale

        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + mlp_out * self.residual_scale

        return x, new_cache
