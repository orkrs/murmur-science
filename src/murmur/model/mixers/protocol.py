"""Sequence mixer protocol."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SequenceMixer(Protocol):
    """Protocol for sequence mixing layers.

    All mixers must implement this interface to be interchangeable
    within the recurrent core.
    """

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: dict | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """Mix sequence dimension.

        Args:
            x: Input [batch, seq_len, d_model]
            positions: Position indices
            cache: Optional cache state
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, new_cache)
        """
        ...
