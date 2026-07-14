"""Tests for MurmurForCausalLM."""

import pytest
import torch

from murmur.model.config import ModelConfig
from murmur.model.language_model import MurmurForCausalLM


class TestMurmurForCausalLM:
    """Test causal language model."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ModelConfig(
            d_model=128,
            vocab_size=1000,
            n_prelude=1,
            n_core=2,
            n_coda=1,
            q_heads=4,
            kv_heads=2,
            head_dim=32,
            ffn_dim=256,
            max_seq_len=64,
            min_depth=1,
            max_depth=4,
        )

    def test_output_shape(self, config):
        """Output has correct shape."""
        model = MurmurForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        output = model(input_ids)
        assert output.logits.shape == (2, 16, config.vocab_size)

    def test_tied_weights(self, config):
        """LM head and embedding share weights."""
        model = MurmurForCausalLM(config)
        assert model.lm_head.weight is model.token_embedding.weight

    def test_causal_loss(self, config):
        """Loss computed with correct shift."""
        model = MurmurForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_ignore_index(self, config):
        """Loss ignores padding tokens."""
        model = MurmurForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        labels[:, :8] = -100
        output = model(input_ids, labels=labels)
        assert output.loss is not None

    def test_variable_depth(self, config):
        """Different sequences can have different depths."""
        model = MurmurForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (3, 16))
        depths = torch.tensor([1, 2, 3])
        output = model(input_ids, depths=depths)
        assert output.logits.shape == (3, 16, config.vocab_size)

    def test_parameter_count(self, config):
        """Parameter counting works."""
        model = MurmurForCausalLM(config)
        counts = model.count_parameters()
        assert "total_unique" in counts
        assert counts["total_unique"] > 0
        assert counts["embedding"] > 0
        assert counts["core"] > 0

    def test_no_nan_fp32(self, config):
        """No NaN in FP32 forward/backward."""
        model = MurmurForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        output.loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_model_cached_decode_matches_full(self, config):
        """The model-level recurrent cache preserves logits."""
        torch.manual_seed(1)
        model = MurmurForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        full = model(input_ids, depths=torch.tensor([2])).logits
        prefix = model(input_ids[:, :3], depths=torch.tensor([2]), use_cache=True)
        cache = prefix.diagnostics["cache"]
        tail = model(
            input_ids[:, 3:], depths=torch.tensor([2]), cache=cache, use_cache=True
        ).logits
        assert torch.allclose(full[:, 3:], tail, atol=1e-5, rtol=1e-5)
