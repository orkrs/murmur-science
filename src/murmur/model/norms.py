"""Normalization layers."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes: x * rsqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
