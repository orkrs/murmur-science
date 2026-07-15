"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from murmur.config import ConfigError, TrainConfig, load_run_config


class TestConfigLoading:
    """Test configuration loading from TOML files."""

    def test_load_smoke_config(self):
        """Load smoke_gqa.toml successfully."""
        config_path = Path("configs/smoke_gqa.toml")
        if not config_path.exists():
            pytest.skip("smoke_gqa.toml not found")

        config = load_run_config(config_path)
        assert config.model.d_model == 512
        assert config.model.vocab_size == 32000
        assert config.model.n_prelude == 1
        assert config.model.n_core == 2
        assert config.model.n_coda == 1
        assert config.model.q_heads == 8
        assert config.model.kv_heads == 2
        assert config.data.sequence_length == 512
        assert config.train.max_tokens == 10000000

    def test_load_prototype_config(self):
        """Load prototype_gqa.toml successfully."""
        config_path = Path("configs/prototype_gqa.toml")
        if not config_path.exists():
            pytest.skip("prototype_gqa.toml not found")

        config = load_run_config(config_path)
        assert config.model.d_model == 1024
        assert config.model.vocab_size == 48000
        assert config.model.n_prelude == 2
        assert config.model.n_core == 4
        assert config.model.n_coda == 2

    def test_reject_unknown_top_level_keys(self, tmp_path):
        """Reject configuration with unknown top-level keys."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\n\n[unknown_section]\nfoo = 1\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="unknown top-level"):
            load_run_config(path)

    def test_reject_unknown_model_keys(self, tmp_path):
        """Reject configuration with unknown model keys."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nunknown_param = true\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="unknown keys in \\[model\\]"):
            load_run_config(path)

    def test_reject_unknown_train_keys(self, tmp_path):
        """Reject configuration with unknown train keys."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'x'\nval_path = 'y'\nsequence_length = 512\nbatch_size = 4\n\n"
            "[train]\nseed = 17\nunknown = true\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="unknown keys in \\[train\\]"):
            load_run_config(path)


class TestModelConfigValidation:
    """Test ModelConfig validation constraints."""

    def test_d_model_divisible_by_q_heads(self, tmp_path):
        """d_model must be divisible by q_heads."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 513\nq_heads = 8\nvocab_size = 32000\n"
            "n_prelude = 1\nn_core = 2\nn_coda = 1\nkv_heads = 2\n"
            "head_dim = 64\nffn_dim = 1408\nmax_seq_len = 512\n"
            "min_depth = 1\nmax_depth = 4\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="d_model.*divisible.*q_heads"):
            load_run_config(path)

    def test_q_heads_divisible_by_kv_heads(self, tmp_path):
        """q_heads must be divisible by kv_heads."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nq_heads = 8\nkv_heads = 3\nvocab_size = 32000\n"
            "n_prelude = 1\nn_core = 2\nn_coda = 1\n"
            "head_dim = 64\nffn_dim = 1408\nmax_seq_len = 512\n"
            "min_depth = 1\nmax_depth = 4\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="q_heads.*divisible.*kv_heads"):
            load_run_config(path)

    def test_head_dim_must_be_even(self, tmp_path):
        """head_dim must be even for RoPE."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nq_heads = 8\nkv_heads = 2\nvocab_size = 32000\n"
            "n_prelude = 1\nn_core = 2\nn_coda = 1\n"
            "head_dim = 63\nffn_dim = 1408\nmax_seq_len = 512\n"
            "min_depth = 1\nmax_depth = 4\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="head_dim.*even"):
            load_run_config(path)

    def test_min_depth_must_be_positive(self, tmp_path):
        """min_depth must be >= 1."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nq_heads = 8\nkv_heads = 2\nvocab_size = 32000\n"
            "n_prelude = 1\nn_core = 2\nn_coda = 1\n"
            "head_dim = 64\nffn_dim = 1408\nmax_seq_len = 512\n"
            "min_depth = 0\nmax_depth = 4\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="min_depth.*must be >= 1"):
            load_run_config(path)

    def test_max_depth_must_be_gte_min_depth(self, tmp_path):
        """max_depth must be >= min_depth."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nq_heads = 8\nkv_heads = 2\nvocab_size = 32000\n"
            "n_prelude = 1\nn_core = 2\nn_coda = 1\n"
            "head_dim = 64\nffn_dim = 1408\nmax_seq_len = 512\n"
            "min_depth = 4\nmax_depth = 2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="max_depth.*must be >= min_depth"):
            load_run_config(path)


class TestDataConfigValidation:
    """Test DataConfig validation."""

    def test_sequence_length_must_be_positive(self, tmp_path):
        """sequence_length must be >= 1."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'x'\nval_path = 'y'\n"
            "sequence_length = 0\nbatch_size = 4\n\n"
            "[train]\nseed = 42\nmax_tokens = 1000\nlearning_rate = 3e-4\n"
            "weight_decay = 0.1\nwarmup_steps = 100\ngrad_accum_steps = 1\ngrad_clip_norm = 1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="sequence_length.*must be >= 1"):
            load_run_config(path)

    def test_batch_size_must_be_positive(self, tmp_path):
        """batch_size must be >= 1."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'x'\nval_path = 'y'\n"
            "sequence_length = 512\nbatch_size = 0\n\n"
            "[train]\nseed = 42\nmax_tokens = 1000\nlearning_rate = 3e-4\n"
            "weight_decay = 0.1\nwarmup_steps = 100\ngrad_accum_steps = 1\ngrad_clip_norm = 1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="batch_size.*must be >= 1"):
            load_run_config(path)


class TestTrainConfigValidation:
    """Test TrainConfig validation."""

    def test_bf16_is_allowed_without_fp16(self):
        config = TrainConfig(42, 1000, 3e-4, 0.1, 10, 1, 1.0, fp16=False, bf16=True)

        assert config.bf16 is True
        assert config.to_dict()["bf16"] is True

    def test_rejects_enabling_fp16_and_bf16_together(self):
        with pytest.raises(ConfigError, match="cannot both be enabled"):
            TrainConfig(42, 1000, 3e-4, 0.1, 10, 1, 1.0, fp16=True, bf16=True)

    def test_learning_rate_must_be_positive(self, tmp_path):
        """learning_rate must be > 0."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'x'\nval_path = 'y'\nsequence_length = 512\nbatch_size = 4\n\n"
            "[train]\nseed = 42\nmax_tokens = 1000\n"
            "learning_rate = 0.0\nweight_decay = 0.1\n"
            "warmup_steps = 100\ngrad_accum_steps = 1\ngrad_clip_norm = 1.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="learning_rate.*must be > 0"):
            load_run_config(path)

    def test_grad_clip_norm_must_be_positive(self, tmp_path):
        """grad_clip_norm must be > 0."""
        path = tmp_path / "bad.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'x'\nval_path = 'y'\nsequence_length = 512\nbatch_size = 4\n\n"
            "[train]\nseed = 42\nmax_tokens = 1000\n"
            "learning_rate = 3e-4\nweight_decay = 0.1\n"
            "warmup_steps = 100\ngrad_accum_steps = 1\ngrad_clip_norm = 0.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="grad_clip_norm.*must be > 0"):
            load_run_config(path)


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_to_dict_roundtrip(self, tmp_path):
        """Configuration can be serialized to dict and back."""
        path = tmp_path / "config.toml"
        path.write_text(
            "[model]\nd_model = 512\nvocab_size = 32000\nn_prelude = 1\nn_core = 2\n"
            "n_coda = 1\nq_heads = 8\nkv_heads = 2\nhead_dim = 64\nffn_dim = 1408\n"
            "max_seq_len = 512\nmin_depth = 1\nmax_depth = 4\n\n"
            "[data]\ntrain_path = 'train.bin'\nval_path = 'val.bin'\n"
            "sequence_length = 512\nbatch_size = 4\n\n"
            "[train]\nseed = 42\nmax_tokens = 10000000\nlearning_rate = 3e-4\n"
            "weight_decay = 0.1\nwarmup_steps = 100\ngrad_accum_steps = 4\n"
            "grad_clip_norm = 1.0\n",
            encoding="utf-8",
        )
        config = load_run_config(path)
        data = config.to_dict()
        assert data["model"]["d_model"] == 512
        assert data["data"]["sequence_length"] == 512
        assert data["train"]["seed"] == 42
