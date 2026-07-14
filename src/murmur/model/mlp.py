"""SwiGLU MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Computes: down(silu(gate(x)) * up(x))
    """

    def __init__(self, d_model: int, ffn_dim: int):
        """Initialize SwiGLU.

        Args:
            d_model: Model dimension
            ffn_dim: Intermediate dimension
        """
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        return self.down_proj(functional.silu(self.gate_proj(x)) * self.up_proj(x))
