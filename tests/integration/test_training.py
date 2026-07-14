"""Integration tests for the first real training/checkpoint path."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from murmur.config import DataConfig, EvalConfig, ModelConfig, RunConfig, TrainConfig
from murmur.model.language_model import MurmurForCausalLM
from murmur.training.trainer import Trainer


def tiny_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(32, 64, 1, 1, 1, 2, 1, 16, 64, 16, 1, 2),
        data=DataConfig("train.bin", "val.bin", 16, 2, 0, False),
        train=TrainConfig(7, 64, 1e-3, 0.0, 1, 1, 1.0, False, checkpoint_interval=2),
        eval=EvalConfig(),
    )


def test_trainer_writes_and_restores_checkpoint(tmp_path: Path):
    config = tiny_config()
    data = torch.randint(0, 64, (8, 17))
    loader = [batch[0] for batch in DataLoader(TensorDataset(data), batch_size=2)]
    trainer = Trainer(MurmurForCausalLM(config.model), loader, config, tmp_path / "run", "cpu")
    result = trainer.train(max_steps=2)
    assert result["tokens_seen"] == 64
    checkpoint = tmp_path / "run" / "checkpoints" / "last"
    resumed = Trainer(MurmurForCausalLM(config.model), loader, config, tmp_path / "resume", "cpu")
    assert resumed.resume(checkpoint)["step"] == 2
