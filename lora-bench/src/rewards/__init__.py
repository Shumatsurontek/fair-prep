"""Reward registry. Add new rewards to REWARD_FUNCS."""
from __future__ import annotations

from .gsm8k import format_reward, gsm8k_accuracy_reward


REWARD_FUNCS = {
    "gsm8k_exact_match": gsm8k_accuracy_reward,
    "format_think": format_reward,
}


def get_reward_fn(name: str):
    if name not in REWARD_FUNCS:
        raise KeyError(f"unknown reward {name!r}; available: {list(REWARD_FUNCS)}")
    return REWARD_FUNCS[name]
