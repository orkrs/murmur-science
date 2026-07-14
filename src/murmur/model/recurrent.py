"""Recurrent core with weight sharing and stable input injection."""

import torch
import torch.nn as nn

from murmur.model.block import TransformerBlock
from murmur.model.mixers.mamba3 import Mamba3Block
from murmur.model.norms import RMSNorm


class StableInputInjection(nn.Module):
    """Stable input injection for recurrent depth.

    Injects the original input into each recurrent step using a
    continuous-time decay mechanism to prevent state explosion.
    """

    def __init__(self, d_model: int):
        """Initialize stable input injection.

        Args:
            d_model: Model dimension
        """
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        self.log_decay = nn.Parameter(torch.zeros(d_model))

    def forward(self, e: torch.Tensor, step: int) -> torch.Tensor:
        """Inject input at given step.

        Args:
            e: Input embedding [batch, seq_len, d_model]
            step: Current recurrent step (0-indexed)

        Returns:
            Injected input [batch, seq_len, d_model]
        """
        decay = torch.exp(self.log_decay).clamp(min=1e-6, max=1.0)
        decay_factor = decay ** step
        return self.input_proj(e) * decay_factor


class RecurrentCore(nn.Module):
    """Shared recurrent transformer core.

    Executes the same set of transformer blocks T times with weight sharing.
    Supports per-sequence variable depth through masking.
    """

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
        mixer: str = "gqa",
    ):
        """Initialize recurrent core.

        Args:
            n_blocks: Number of physical transformer blocks
            d_model: Model dimension
            q_heads: Number of query heads
            kv_heads: Number of key/value heads
            head_dim: Dimension per head
            ffn_dim: FFN intermediate dimension
            max_seq_len: Maximum sequence length
            rope_theta: RoPE base frequency
            norm_eps: Normalization epsilon
            residual_scale: Residual connection scaling
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model

        if mixer not in {"gqa", "mamba3"}:
            raise ValueError(f"unknown mixer: {mixer}")
        if mixer == "mamba3":
            self.blocks = nn.ModuleList([
                Mamba3Block(d_model, ffn_dim, d_state=128, headdim=head_dim, norm_eps=norm_eps, residual_scale=residual_scale, layer_idx=i)
                for i in range(n_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    d_model, q_heads, kv_heads, head_dim, ffn_dim,
                    max_seq_len, rope_theta, norm_eps, residual_scale
                )
                for _ in range(n_blocks)
            ])

        self.injection = StableInputInjection(d_model)
        self.norm = RMSNorm(d_model, norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        depths: torch.Tensor,
        positions: torch.Tensor | None = None,
        caches: list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor] | None]]]:
        """Forward pass with shared weights.

        Args:
            hidden: Input hidden states [batch, seq_len, d_model]
            injection: Input to inject at each step [batch, seq_len, d_model]
            depths: Per-sequence depth [batch] (number of recurrent steps)
            positions: Position indices for RoPE

        Returns:
            Output hidden states [batch, seq_len, d_model]
        """
        batch_size = hidden.shape[0]
        if depths.ndim != 1 or depths.shape[0] != batch_size:
            raise ValueError("depths must have shape [batch]")
        if torch.any(depths < 1):
            raise ValueError("all recurrent depths must be >= 1")
        max_depth = int(depths.max().detach().cpu())
        if caches is None:
            caches = [[None for _ in self.blocks] for _ in range(max_depth)]
        elif len(caches) < max_depth or any(len(row) != len(self.blocks) for row in caches):
            raise ValueError("core cache shape does not match recurrent depth")

        active_mask = torch.arange(max_depth, device=hidden.device).unsqueeze(0) < depths.unsqueeze(1)

        for step in range(max_depth):
            step_mask = active_mask[:, step].unsqueeze(1).unsqueeze(2)

            injected = self.injection(injection, step)
            hidden_with_inject = hidden + injected

            for block_idx, block in enumerate(self.blocks):
                hidden_new, new_cache = block(
                    hidden_with_inject,
                    positions,
                    caches[step][block_idx],
                    use_cache,
                )
                if use_cache:
                    caches[step][block_idx] = new_cache
                hidden = torch.where(step_mask, hidden_new, hidden)

        result = self.norm(hidden)
        return (result, caches) if use_cache else result
