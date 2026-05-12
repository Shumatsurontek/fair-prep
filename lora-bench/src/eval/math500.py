"""MATH-500 scorer + dataset binding."""
from __future__ import annotations

import re

from ..data.math500 import load_math500

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_RE = re.compile(r"####\s*(.+?)(?:\n|$)")


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    m = HASH_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def check_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if pred.strip() == gold.strip():
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


def score(text: str, example: dict) -> tuple[str | None, bool]:
    pred = extract_boxed(text)
    return pred, check_equal(pred, example["gold"])


def load_dataset(cfg: dict, tokenizer, max_samples: int | None = None):
    # cfg unused; signature kept consistent with eval/gsm8k.load_dataset
    return load_math500(tokenizer, max_samples=max_samples)
