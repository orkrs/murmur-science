"""Integration tests for R0 (fixed-depth) and R1 (recurrent) cores."""

import torch

from murmur.model.block import TransformerBlock
from murmur.model.recurrent import RecurrentCore


class TestR0R1Forward:
    """Test R0 and R1 forward passes."""

    def test_r0_fixed_depth(self):
        """R0: fixed-depth transformer (no weight sharing)."""
        blocks = torch.nn.ModuleList([
            TransformerBlock(d_model=64, q_heads=2, kv_heads=1, head_dim=32, ffn_dim=128)
            for _ in range(3)
        ])
        x = torch.randn(2, 8, 64)
        for block in blocks:
            x, _ = block(x)
        assert x.shape == (2, 8, 64)

    def test_r1_recurrent_depth(self):
        """R1: recurrent depth with weight sharing."""
        core = RecurrentCore(
            n_blocks=2, d_model=64, q_heads=2, kv_heads=1,
            head_dim=32, ffn_dim=128
        )
        hidden = torch.randn(2, 8, 64)
        injection = torch.randn(2, 8, 64)
        depths = torch.tensor([3, 3])
        out = core(hidden, injection, depths)
        assert out.shape == (2, 8, 64)

    def test_r1_parameter_count(self):
        """R1 has fewer parameters than R0 for same effective depth."""
        n_blocks = 2
        d_model = 64
        q_heads = 2
        kv_heads = 1
        head_dim = 32
        ffn_dim = 128

        r0_blocks = [
            TransformerBlock(d_model, q_heads, kv_heads, head_dim, ffn_dim)
            for _ in range(n_blocks)
        ]
        r0_params = sum(sum(p.numel() for p in b.parameters()) for b in r0_blocks)

        core = RecurrentCore(n_blocks, d_model, q_heads, kv_heads, head_dim, ffn_dim)
        r1_params = sum(p.numel() for p in core.parameters())

        assert r1_params == r0_params + sum(p.numel() for p in core.injection.parameters()) + sum(p.numel() for p in core.norm.parameters())
