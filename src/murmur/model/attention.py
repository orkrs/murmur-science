"""Grouped Query Attention (GQA)."""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from murmur.model.rotary import RotaryEmbedding, apply_rotary_pos_emb


class GQAAttention(nn.Module):
    """Grouped Query Attention with RoPE.

    Uses fewer KV heads than Q heads for memory efficiency.
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
        """Initialize GQA attention.

        Args:
            d_model: Model dimension
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Dimension per head
            max_seq_len: Maximum sequence length for RoPE
            rope_theta: RoPE base frequency
        """
        super().__init__()
        if q_heads % kv_heads != 0:
            raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})")

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.num_groups = q_heads // kv_heads

        self.q_proj = nn.Linear(d_model, q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(q_heads * head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

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
            kv_cache: Optional cached (k, v) from previous tokens
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, new_cache)
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if kv_cache is None else kv_cache[0].shape[2]
        if positions is None:
            positions = torch.arange(
                past_len, past_len + seq_len, device=x.device, dtype=torch.long
            )
        cos, sin = self.rope(q, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        if kv_cache is None:
            out = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif seq_len == 1:
            out = functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            key_len = k.shape[2]
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(key_len, device=x.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            mask = allowed.unsqueeze(0).unsqueeze(0)
            out = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.o_proj(out)

        return out, new_cache
