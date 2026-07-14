"""Rotary Position Embedding (RoPE)."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for transformer attention.

    Applies rotation to query and key vectors based on their position,
    enabling relative position encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """Initialize RoPE.

        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length
            theta: Base for frequency computation
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotation matrices.

        Args:
            x: Input tensor (used for device/dtype)
            positions: Position indices [batch, seq_len] or None for arange

        Returns:
            Tuple of (cos, sin) tensors of shape [seq_len, dim]
        """
        if positions is None:
            seq_len = x.shape[-2]
            positions = torch.arange(seq_len, device=x.device)
        positions = positions.to(device=x.device, dtype=torch.float32)
        if positions.ndim == 0:
            positions = positions.reshape(1)
        if positions.ndim not in (1, 2):
            raise ValueError("positions must have shape [seq] or [batch, seq]")

        freqs = positions[..., None] * self.inv_freq.to(x.device)[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, kv_heads, seq_len, head_dim]
        cos: Cosine tensor [seq_len, head_dim]
        sin: Sine tensor [seq_len, head_dim]

    Returns:
        Tuple of rotated (q, k)
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError("cos/sin must have shape [seq, dim] or [batch, seq, dim]")

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot
