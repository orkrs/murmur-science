"""Run configuration loading and validation."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from murmur.model.config import ModelConfig


class ConfigError(Exception):
    """Configuration validation error."""

    pass


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline configuration."""

    train_path: str
    val_path: str
    sequence_length: int
    batch_size: int
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_seed: int = 42

    def __post_init__(self):
        if self.sequence_length < 1:
            raise ConfigError(f"sequence_length ({self.sequence_length}) must be >= 1")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size ({self.batch_size}) must be >= 1")
        if self.num_workers < 0:
            raise ConfigError(f"num_workers ({self.num_workers}) must be >= 0")

    def to_dict(self) -> dict:
        return {
            "train_path": self.train_path,
            "val_path": self.val_path,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle_seed": self.shuffle_seed,
        }


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    seed: int
    max_tokens: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_accum_steps: int
    grad_clip_norm: float
    fp16: bool = True
    bf16: bool = False
    log_interval: int = 10
    val_interval: int = 100
    checkpoint_interval: int = 500
    max_grad_skips: int = 10

    def __post_init__(self):
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens ({self.max_tokens}) must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate ({self.learning_rate}) must be > 0")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay ({self.weight_decay}) must be >= 0")
        if self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps ({self.warmup_steps}) must be >= 0")
        if self.grad_accum_steps < 1:
            raise ConfigError(f"grad_accum_steps ({self.grad_accum_steps}) must be >= 1")
        if self.grad_clip_norm <= 0:
            raise ConfigError(f"grad_clip_norm ({self.grad_clip_norm}) must be > 0")
        if self.fp16 and self.bf16:
            raise ConfigError("fp16 and bf16 cannot both be enabled")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "grad_clip_norm": self.grad_clip_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "log_interval": self.log_interval,
            "val_interval": self.val_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "max_grad_skips": self.max_grad_skips,
        }


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 8
    max_depth: int | None = None
    num_samples: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
        if self.batch_size < 1:
            raise ConfigError(f"batch_size ({self.batch_size}) must be >= 1")
        if self.num_samples < 1:
            raise ConfigError(f"num_samples ({self.num_samples}) must be >= 1")
        if self.temperature <= 0:
            raise ConfigError(f"temperature ({self.temperature}) must be > 0")

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "max_depth": self.max_depth,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class RunConfig:
    """Complete run configuration."""

    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    run_name: str = "default"
    output_dir: str = "artifacts/runs"

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
            "run_name": self.run_name,
            "output_dir": self.output_dir,
        }


def _validate_keys(data: dict, allowed: set[str], section: str) -> None:
    """Validate that no unknown keys are present."""
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in [{section}]: {', '.join(sorted(unknown))}")


def load_run_config(path: Path) -> RunConfig:
    """Load and validate run configuration from TOML file.

    Args:
        path: Path to TOML configuration file

    Returns:
        Validated RunConfig

    Raises:
        ConfigError: If configuration is invalid or contains unknown keys
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Validate top-level keys
    top_allowed = {"model", "data", "train", "eval", "run_name", "output_dir"}
    unknown = set(data.keys()) - top_allowed
    if unknown:
        raise ConfigError(f"unknown top-level keys: {', '.join(sorted(unknown))}")

    # Model config (required)
    if "model" not in data:
        raise ConfigError("missing required section [model]")
    model_data = data["model"]
    model_allowed = {
        "d_model",
        "vocab_size",
        "n_prelude",
        "n_core",
        "n_coda",
        "q_heads",
        "kv_heads",
        "head_dim",
        "ffn_dim",
        "max_seq_len",
        "min_depth",
        "max_depth",
        "mixer",
        "rope_theta",
        "rope_dim",
        "residual_scale",
        "norm_eps",
        "tie_embeddings",
    }
    _validate_keys(model_data, model_allowed, "model")
    model = ModelConfig.from_dict(model_data)

    # Data config (required)
    if "data" not in data:
        raise ConfigError("missing required section [data]")
    data_section = data["data"]
    data_allowed = {
        "train_path",
        "val_path",
        "sequence_length",
        "batch_size",
        "num_workers",
        "pin_memory",
        "shuffle_seed",
    }
    _validate_keys(data_section, data_allowed, "data")
    data_config = DataConfig(**data_section)

    # Train config (required)
    if "train" not in data:
        raise ConfigError("missing required section [train]")
    train_data = data["train"]
    train_allowed = {
        "seed",
        "max_tokens",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "grad_accum_steps",
        "grad_clip_norm",
        "fp16",
        "bf16",
        "log_interval",
        "val_interval",
        "checkpoint_interval",
        "max_grad_skips",
    }
    _validate_keys(train_data, train_allowed, "train")
    train_config = TrainConfig(**train_data)

    # Eval config
    eval_data = data.get("eval", {})
    eval_allowed = {
        "batch_size",
        "max_depth",
        "num_samples",
        "temperature",
        "top_k",
        "top_p",
    }
    _validate_keys(eval_data, eval_allowed, "eval")
    eval_config = EvalConfig(**eval_data)

    return RunConfig(
        model=model,
        data=data_config,
        train=train_config,
        eval=eval_config,
        run_name=data.get("run_name", "default"),
        output_dir=data.get("output_dir", "artifacts/runs"),
    )
