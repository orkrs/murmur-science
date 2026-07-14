"""Tests for recurrent core."""

import torch
import torch.nn as nn

import murmur.model.recurrent as recurrent_module
from murmur.model.recurrent import RecurrentCore, StableInputInjection


class TestStableInputInjection:
    """Test stable input injection."""

    def test_output_shape(self):
        """Output has correct shape."""
        injection = StableInputInjection(d_model=128)
        e = torch.randn(2, 8, 128)
        out = injection(e, step=0)
        assert out.shape == (2, 8, 128)

    def test_decay_over_steps(self):
        """Injection decays over steps."""
        injection = StableInputInjection(d_model=64)
        e = torch.randn(1, 4, 64)
        out0 = injection(e, step=0)
        out1 = injection(e, step=1)
        out2 = injection(e, step=2)
        assert out0.norm() >= out1.norm() >= out2.norm()


class TestRecurrentCore:
    """Test recurrent core with weight sharing."""

    def test_output_shape(self):
        """Output has correct shape."""
        core = RecurrentCore(
            n_blocks=2, d_model=128, q_heads=4, kv_heads=2,
            head_dim=32, ffn_dim=256
        )
        hidden = torch.randn(2, 8, 128)
        injection = torch.randn(2, 8, 128)
        depths = torch.tensor([2, 3])
        out = core(hidden, injection, depths)
        assert out.shape == (2, 8, 128)

    def test_weight_sharing(self):
        """All blocks share the same parameters."""
        core = RecurrentCore(
            n_blocks=2, d_model=128, q_heads=4, kv_heads=2,
            head_dim=32, ffn_dim=256
        )
        block0_params = set(id(p) for p in core.blocks[0].parameters())
        block1_params = set(id(p) for p in core.blocks[1].parameters())
        assert block0_params.isdisjoint(block1_params)

    def test_variable_depth(self):
        """Different sequences can have different depths."""
        core = RecurrentCore(
            n_blocks=1, d_model=64, q_heads=2, kv_heads=1,
            head_dim=32, ffn_dim=128
        )
        hidden = torch.randn(3, 4, 64)
        injection = torch.randn(3, 4, 64)
        depths = torch.tensor([1, 2, 3])
        out = core(hidden, injection, depths)
        assert out.shape == (3, 4, 64)

    def test_no_nan_fp32(self):
        """No NaN in FP32 forward/backward."""
        core = RecurrentCore(
            n_blocks=1, d_model=64, q_heads=2, kv_heads=1,
            head_dim=32, ffn_dim=128
        )
        hidden = torch.randn(2, 4, 64, requires_grad=True)
        injection = torch.randn(2, 4, 64)
        depths = torch.tensor([2, 2])
        out = core(hidden, injection, depths)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(hidden.grad).any()

    def test_t1_equivalence(self):
        """T=1 should behave like a single pass."""
        core = RecurrentCore(
            n_blocks=2, d_model=64, q_heads=2, kv_heads=1,
            head_dim=32, ffn_dim=128
        )
        hidden = torch.randn(1, 4, 64)
        injection = torch.zeros(1, 4, 64)
        depths = torch.tensor([1])
        out = core(hidden, injection, depths)
        assert out.shape == (1, 4, 64)

    def test_mamba3_mimo_core_uses_rank_two_mixer(self, monkeypatch):
        """The experimental MIMO branch must not silently instantiate SISO."""
        calls = []

        class FakeMambaBlock(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                calls.append(kwargs)

            def forward(self, x, positions=None, kv_cache=None, use_cache=False):
                return x, None

        monkeypatch.setattr(recurrent_module, "Mamba3Block", FakeMambaBlock)
        core = RecurrentCore(
            n_blocks=2, d_model=64, q_heads=2, kv_heads=1,
            head_dim=32, ffn_dim=128, mixer="mamba3_mimo"
        )

        assert len(core.blocks) == 2
        assert [call["is_mimo"] for call in calls] == [True, True]
        assert [call["mimo_rank"] for call in calls] == [2, 2]
