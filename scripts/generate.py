"""Greedy/top-k generation from a saved model."""

import argparse
from pathlib import Path

import torch

from murmur.config import load_run_config
from murmur.model.language_model import MurmurForCausalLM
from murmur.tokenizer import MurmurTokenizer
from murmur.training.checkpoint import MODEL_FILE, load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_run_config(args.config)
    tokenizer = MurmurTokenizer(args.tokenizer)
    model = MurmurForCausalLM(config.model).to(args.device).eval()
    checkpoint_file = args.checkpoint / MODEL_FILE if args.checkpoint.is_dir() else args.checkpoint
    model.load_state_dict(load_file(str(checkpoint_file), device="cpu"))
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    cache = None
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            output = model(ids if cache is None else ids[:, -1:], depths=torch.tensor([config.model.max_depth], device=args.device), cache=cache, use_cache=True)
            cache = output.diagnostics["cache"]
            next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if int(next_id.item()) == tokenizer.eos_id:
                break
    print(tokenizer.decode(ids[0].tolist()))


if __name__ == "__main__":
    main()
