"""Tests for GQA attention."""

import torch

from murmur.model.attention import GQAAttention


class TestGQAAttention:
    """Test GQA attention layer."""

    def test_output_shape(self):
        """Output has correct shape."""
        attn = GQAAttention(d_model=512, q_heads=8, kv_heads=2, head_dim=64)
        x = torch.randn(2, 16, 512)
        out, _ = attn(x)
        assert out.shape == (2, 16, 512)

    def test_gqa_kv_repeat(self):
        """GQA repeats KV heads to match Q heads."""
        attn = GQAAttention(d_model=256, q_heads=8, kv_heads=2, head_dim=32)
        assert attn.num_groups == 4

    def test_causal_mask(self):
        """Future tokens don't affect past tokens."""
        attn = GQAAttention(d_model=128, q_heads=4, kv_heads=2, head_dim=32)
        x = torch.randn(1, 8, 128)
        out1, _ = attn(x)

        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(128)
        out2, _ = attn(x_modified)

        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)

    def test_kv_cache(self):
        """KV cache works correctly."""
        attn = GQAAttention(d_model=128, q_heads=4, kv_heads=2, head_dim=32)

        x1 = torch.randn(1, 4, 128)
        out1, cache1 = attn(x1, use_cache=True)

        x2 = torch.randn(1, 1, 128)
        out2, cache2 = attn(x2, kv_cache=cache1, use_cache=True)

        assert cache2[0].shape[2] == 5
        assert cache2[1].shape[2] == 5

    def test_cached_decode_matches_full_forward(self):
        """Incremental decode uses the same positions and causal mask as full pass."""
        torch.manual_seed(0)
        attn = GQAAttention(d_model=128, q_heads=4, kv_heads=2, head_dim=32)
        x = torch.randn(1, 6, 128)
        full, _ = attn(x)
        _, cache = attn(x[:, :4], use_cache=True)
        step, _ = attn(x[:, 4:], kv_cache=cache, use_cache=True)
        assert torch.allclose(full[:, 4:], step, atol=1e-5, rtol=1e-5)

    def test_no_nan_fp32(self):
        """No NaN in FP32 forward/backward."""
        attn = GQAAttention(d_model=128, q_heads=4, kv_heads=2, head_dim=32)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(x.grad).any()
