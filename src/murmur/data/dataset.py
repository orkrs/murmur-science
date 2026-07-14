"""Memory-mapped token dataset."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Memory-mapped dataset of packed token sequences.

    Each sample is a contiguous chunk of `sample_length` uint16 tokens
    stored in little-endian binary format.
    """

    def __init__(self, shard_paths: list[Path], sample_length: int):
        """Initialize dataset from shard files.

        Args:
            shard_paths: List of paths to .bin shard files
            sample_length: Number of tokens per sample (including label shift)
        """
        self.shard_paths = [Path(p) for p in shard_paths]
        self.sample_length = sample_length
        self._shards: list[np.memmap] = []
        self._shard_offsets: list[int] = []
        self._total_samples = 0

        for shard_path in self.shard_paths:
            idx_path = shard_path.with_suffix(".idx")
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)

            num_samples = idx["num_samples"]
            shard = np.memmap(
                str(shard_path),
                dtype=np.uint16,
                mode="r",
                shape=(num_samples * sample_length,),
            )
            self._shards.append(shard)
            self._shard_offsets.append(self._total_samples)
            self._total_samples += num_samples

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample by global index.

        Args:
            idx: Global sample index

        Returns:
            Tensor of shape [sample_length] with dtype int64
        """
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples})")

        shard_idx = 0
        for i, offset in enumerate(self._shard_offsets):
            if idx < offset + len(self._shards[i]) // self.sample_length:
                shard_idx = i
                break
        else:
            shard_idx = len(self._shards) - 1

        local_idx = idx - self._shard_offsets[shard_idx]
        start = local_idx * self.sample_length
        end = start + self.sample_length
        tokens = self._shards[shard_idx][start:end]
        return torch.from_numpy(tokens.astype(np.int64))
