"""Learning-rate schedules."""

import math


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps and warmup_steps > 0:
        return float(step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))
