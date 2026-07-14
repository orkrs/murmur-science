"""Validation metrics for causal language models."""

import torch

from murmur.model.language_model import MurmurForCausalLM


@torch.no_grad()
def evaluate_loss(model: MurmurForCausalLM, batches, device: torch.device, depth: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_sequences = 0
    for tokens in batches:
        tokens = tokens.to(device)
        output = model(tokens, labels=tokens, depths=torch.full((tokens.shape[0],), depth, device=device))
        total_loss += float(output.loss) * tokens.shape[0]
        total_sequences += tokens.shape[0]
    loss = total_loss / max(1, total_sequences)
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss)))}
