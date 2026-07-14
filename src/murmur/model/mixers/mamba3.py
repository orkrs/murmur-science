"""Thin adapter around the official Mamba 3 implementation."""

import inspect

import torch
import torch.nn as nn

from murmur.model.mlp import SwiGLU
from murmur.model.norms import RMSNorm


class MambaUnavailableError(RuntimeError):
    """Raised when official Mamba 3 is not installed or is incompatible."""


def _official_mamba(**kwargs) -> nn.Module:
    try:
        from mamba_ssm.modules.mamba3 import Mamba3
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        raise MambaUnavailableError("install the pinned official mamba_ssm revision") from exc
    accepted = inspect.signature(Mamba3).parameters
    return Mamba3(**{key: value for key, value in kwargs.items() if key in accepted})


class Mamba3Block(nn.Module):
    """Pre-norm Mamba 3 + SwiGLU block with the TransformerBlock contract."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        d_state: int,
        headdim: int,
        norm_eps: float,
        residual_scale: float,
        layer_idx: int,
        is_mimo: bool = False,
        mimo_rank: int = 1,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.attn_norm = RMSNorm(d_model, norm_eps)
        if mimo_rank < 1:
            raise ValueError("mimo_rank must be positive")
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        # Official Mamba-3 recommends chunk_size=64/rank for MIMO.
        chunk_size = 64 // self.mimo_rank if is_mimo else 64
        self.mixer = _official_mamba(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            is_mimo=is_mimo,
            mimo_rank=self.mimo_rank,
        )
        self.mlp_norm = RMSNorm(d_model, norm_eps)
        self.mlp = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, positions=None, kv_cache=None, use_cache=False):
        if use_cache or kv_cache is not None:
            raise MambaUnavailableError("Mamba 3 cache adapter has not passed parity gate")
        mixed = self.mixer(self.attn_norm(x))
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        x = x + mixed * self.residual_scale
        x = x + self.mlp(self.mlp_norm(x)) * self.residual_scale
        return x, None
