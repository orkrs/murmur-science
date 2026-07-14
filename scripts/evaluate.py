"""Evaluate validation loss and depth curve."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from murmur.config import load_run_config
from murmur.data.dataset import PackedTokenDataset
from murmur.model.language_model import MurmurForCausalLM
from murmur.training.checkpoint import MODEL_FILE, load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_run_config(args.config)
    dataset_path = Path(config.data.val_path)
    if dataset_path.is_dir():
        paths = sorted(dataset_path.glob("*.bin"))
    elif dataset_path.exists():
        paths = [dataset_path]
    else:
        paths = sorted(dataset_path.parent.glob(f"{dataset_path.stem}_*.bin"))
    if not paths:
        raise FileNotFoundError(f"no validation shards found for {dataset_path}")
    loader = DataLoader(PackedTokenDataset(paths, config.data.sequence_length + 1), batch_size=config.eval.batch_size)
    model = MurmurForCausalLM(config.model).to(args.device).eval()
    checkpoint_file = args.checkpoint / MODEL_FILE if args.checkpoint.is_dir() else args.checkpoint
    model.load_state_dict(load_file(str(checkpoint_file), device="cpu"))
    rows = []
    with torch.no_grad():
        for depth in range(config.model.min_depth, (config.eval.max_depth or config.model.max_depth) + 1):
            total, count = 0.0, 0
            for tokens in loader:
                tokens = tokens.to(args.device)
                output = model(tokens, labels=tokens, depths=torch.full((tokens.shape[0],), depth, device=args.device))
                total += float(output.loss) * tokens.shape[0]
                count += tokens.shape[0]
            rows.append({"depth": depth, "loss": total / max(1, count), "perplexity": torch.exp(torch.tensor(total / max(1, count))).item()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
