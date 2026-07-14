"""Depth sweep helpers."""

from murmur.evaluation.language_model import evaluate_loss


def depth_curve(model, batches, device, min_depth: int, max_depth: int) -> list[dict]:
    batches = list(batches)
    return [{"depth": depth, **evaluate_loss(model, batches, device, depth)} for depth in range(min_depth, max_depth + 1)]
