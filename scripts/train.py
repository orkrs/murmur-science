"""Train Murmur from a TOML run configuration."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from murmur.config import load_run_config
from murmur.data.dataset import PackedTokenDataset
from murmur.data.sampler import StatefulTokenSampler
from murmur.model.language_model import MurmurForCausalLM
from murmur.training.trainer import Trainer


def shard_paths(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.bin"))
    if path.exists():
        return [path]
    prefixed = sorted(path.parent.glob(f"{path.stem}_*.bin"))
    if not prefixed:
        raise FileNotFoundError(f"no data shards found for {path}")
    return prefixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_run_config(args.config)
    dataset = PackedTokenDataset(shard_paths(config.data.train_path), config.data.sequence_length + 1)
    sampler = StatefulTokenSampler(dataset, config.data.batch_size, config.data.shuffle_seed)
    # Stateful resume is exact only when the sampler is not advanced by worker
    # prefetch. T4 runs therefore deliberately use the main process loader.
    loader = DataLoader(dataset, batch_size=config.data.batch_size, sampler=sampler, num_workers=0, pin_memory=config.data.pin_memory)
    model = MurmurForCausalLM(config.model)
    trainer = Trainer(model, loader, config, args.run_dir, args.device, sampler=sampler)
    if args.resume:
        trainer.resume(args.resume)
    print(trainer.train())


if __name__ == "__main__":
    main()
