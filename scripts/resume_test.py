"""Run a short train/checkpoint/resume smoke test."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from murmur.config import load_run_config
from murmur.model.language_model import MurmurForCausalLM
from murmur.training.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_run_config(args.config)
    data = torch.randint(0, config.model.vocab_size, (config.data.batch_size * 4, config.data.sequence_length + 1))
    loader = DataLoader(TensorDataset(data), batch_size=config.data.batch_size)
    loader = [batch[0] for batch in loader]
    trainer = Trainer(MurmurForCausalLM(config.model), loader, config, args.output, "cuda" if torch.cuda.is_available() else "cpu")
    trainer.train(max_steps=2)
    checkpoint = args.output / "checkpoints" / "last"
    resumed = Trainer(MurmurForCausalLM(config.model), loader, config, args.output / "resumed", "cuda" if torch.cuda.is_available() else "cpu")
    resumed.resume(checkpoint)
    print({"step": resumed.step, "tokens_seen": resumed.tokens_seen})
