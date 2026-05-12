"""GSM8K rewards for on-policy RL.

Signature: fn(prompts, completions, ground_truth) -> list[float] in [0,1].
"""
from __future__ import annotations

import re
from typing import Sequence

from ..data import normalize_number, parse_predicted_answer


def gsm8k_accuracy_reward(
    prompts: Sequence,
    completions: Sequence,
    ground_truth: Sequence[str] | None = None,
    **kw,
) -> list[float]:
    """1.0 if extracted answer matches gold; else 0.0."""
    if ground_truth is None:
        raise ValueError("gsm8k_accuracy_reward requires `ground_truth`")
    out = []
    for comp, gold in zip(completions, ground_truth):
        text = comp if isinstance(comp, str) else (
            comp[-1]["content"] if isinstance(comp, list) and comp else ""
        )
        pred = normalize_number(parse_predicted_answer(text))
        gold_n = normalize_number(gold)
        out.append(1.0 if (pred is not None and gold_n is not None and pred == gold_n) else 0.0)
    return out


_THINK_FORMAT = re.compile(r"<think>.*?</think>.*?(\\boxed\{.*\}|####\s*-?\d)", re.DOTALL)


def format_reward(prompts, completions, **kw) -> list[float]:
    """Bonus for `<think>…</think>` + final answer marker."""
    out = []
    for comp in completions:
        text = comp if isinstance(comp, str) else (
            comp[-1]["content"] if isinstance(comp, list) and comp else ""
        )
        out.append(1.0 if _THINK_FORMAT.search(text) else 0.0)
    return out
