"""Optimizer construction."""

import torch


def build_adamw(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build AdamW with decay only on matrix weights."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith("embedding.weight"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
