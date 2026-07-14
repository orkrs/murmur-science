"""Tests for transformer block."""

import torch

from murmur.model.block import TransformerBlock


class TestTransformerBlock:
    """Test transformer block."""

    def test_output_shape(self):
        """Output has correct shape."""
        block = TransformerBlock(
            d_model=256, q_heads=4, kv_heads=2, head_dim=64, ffn_dim=512
        )
        x = torch.randn(2, 16, 256)
        out, _ = block(x)
        assert out.shape == (2, 16, 256)

    def test_residual_connection(self):
        """Residual connection preserves information."""
        block = TransformerBlock(
            d_model=128, q_heads=4, kv_heads=2, head_dim=32, ffn_dim=256
        )
        x = torch.randn(1, 8, 128)
        out, _ = block(x)
        assert not torch.allclose(out, x)

    def test_no_nan_fp32(self):
        """No NaN in FP32 forward/backward."""
        block = TransformerBlock(
            d_model=128, q_heads=4, kv_heads=2, head_dim=32, ffn_dim=256
        )
        x = torch.randn(2, 8, 128, requires_grad=True)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(x.grad).any()

    def test_kv_cache(self):
        """KV cache works correctly."""
        block = TransformerBlock(
            d_model=128, q_heads=4, kv_heads=2, head_dim=32, ffn_dim=256
        )

        x1 = torch.randn(1, 4, 128)
        out1, cache1 = block(x1, use_cache=True)

        x2 = torch.randn(1, 1, 128)
        out2, cache2 = block(x2, kv_cache=cache1, use_cache=True)

        assert cache2[0].shape[2] == 5
