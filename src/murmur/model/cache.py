"""Inference cache containers for full and recurrent model paths."""

from dataclasses import dataclass

import torch

KV = tuple[torch.Tensor, torch.Tensor]


@dataclass
class MurmurCache:
    """KV/SSM cache split by physical block and recurrent occurrence."""

    prelude: list[KV | None]
    core: list[list[KV | None]]
    coda: list[KV | None]

    @classmethod
    def empty(cls, n_prelude: int, n_core: int, max_depth: int, n_coda: int) -> "MurmurCache":
        return cls(
            prelude=[None] * n_prelude,
            core=[[None] * n_core for _ in range(max_depth)],
            coda=[None] * n_coda,
        )

    def lengths(self) -> dict[str, list[int | None]]:
        def length(cache: KV | None) -> int | None:
            return None if cache is None else int(cache[0].shape[2])

        return {
            "prelude": [length(x) for x in self.prelude],
            "core": [[length(x) for x in row] for row in self.core],
            "coda": [length(x) for x in self.coda],
        }
