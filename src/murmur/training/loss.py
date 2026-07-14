"""Loss helpers."""

import torch
import torch.nn.functional as functional


def causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute next-token CE with -100 as the ignore index."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, seq, vocab] and labels [batch, seq]")
    return functional.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def bits_per_byte(loss: torch.Tensor, bytes_per_token: float) -> torch.Tensor:
    """Convert nats/token to bits/byte for tokenizer-independent reporting."""
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    return loss.detach() / torch.log(torch.tensor(2.0, device=loss.device)) / bytes_per_token
