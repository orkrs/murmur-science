"""Stateful token sampler with deterministic shuffle and cursor persistence."""

import random
from typing import Iterator

from torch.utils.data import Sampler

from murmur.data.dataset import PackedTokenDataset


class StatefulTokenSampler(Sampler[int]):
    """Deterministic sampler with saveable state for resume.

    Supports:
        - Deterministic shuffle with fixed seed
        - Cursor persistence for mid-epoch resume
        - Epoch tracking
    """

    def __init__(
        self,
        dataset: PackedTokenDataset,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
    ):
        """Initialize sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            seed: Random seed for shuffle
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.cursor = 0
        self._indices: list[int] | None = None

    def _generate_indices(self) -> list[int]:
        """Generate shuffled or sequential indices for current epoch."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices."""
        if self._indices is None:
            self._indices = self._generate_indices()

        while self.cursor < len(self._indices):
            idx = self._indices[self.cursor]
            self.cursor += 1
            yield idx

        self.epoch += 1
        self.cursor = 0
        self._indices = None

    def __len__(self) -> int:
        return len(self.dataset)

    def state_dict(self) -> dict:
        """Save sampler state for resume.

        Returns:
            Dictionary containing epoch, cursor, and seed
        """
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore sampler state from checkpoint.

        Args:
            state: State dictionary from state_dict()
        """
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self.seed = state["seed"]
        self.shuffle = state["shuffle"]
        self._indices = self._generate_indices()
