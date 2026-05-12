"""Bounded FIFO replay buffer of (chosen, rejected) pairs for OnPolicy-DPO-GTW."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Pair:
    prompt: str
    chosen: str
    rejected: str
    reward_margin: float
    step_added: int


class ReplayBuffer:
    def __init__(self, size: int = 256, seed: int = 42):
        self.size = size
        self._buf: deque[Pair] = deque(maxlen=size)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, pair: Pair) -> None:
        self._buf.append(pair)

    def push_many(self, pairs: list[Pair]) -> None:
        for p in pairs:
            self.push(p)

    def sample(self, k: int) -> list[Pair]:
        if not self._buf:
            return []
        k = min(k, len(self._buf))
        return self._rng.sample(list(self._buf), k=k)

    def maybe_inject(self, batch_size: int, inject_prob: float) -> list[Pair]:
        if not self._buf or inject_prob <= 0:
            return []
        k = sum(1 for _ in range(batch_size) if self._rng.random() < inject_prob)
        return self.sample(k)
