"""Categorize MATH-500 failures: format / reasoning / symbolic / truncation."""
from __future__ import annotations

import json
import re
from collections import Counter


SYM_TOKENS = re.compile(r"\\(?:frac|sqrt|pi|sin|cos|tan|log|sum|int|left|right|begin|cdot)")


def has_boxed(text: str | None) -> bool:
    if not text:
        return False
    return r"\boxed{" in text or "####" in text


def is_symbolic_gold(gold: str) -> bool:
    return bool(SYM_TOKENS.search(gold)) or "/" in gold or "(" in gold


def is_pure_number(s: str | None) -> bool:
    if s is None:
        return False
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", s.strip()))


def categorize(sample: dict, full_output: str | None = None) -> str:
    if sample["ok"]:
        return "correct"
    pred = sample.get("pred")
    gold = sample["gold"]
    out = full_output if full_output is not None else sample.get("output", "")

    if out is None:
        if pred is None:
            return "format"
        if is_symbolic_gold(gold) and is_pure_number(pred):
            return "symbolic_mismatch"
        if is_pure_number(pred) and is_pure_number(gold):
            return "reasoning_numeric"
        return "reasoning_symbolic"

    if not has_boxed(out):
        return "truncation"
    if pred is None:
        return "format"
    if is_symbolic_gold(gold) and is_pure_number(pred):
        return "symbolic_mismatch"
    if is_pure_number(pred) and is_pure_number(gold):
        return "reasoning_numeric"
    return "reasoning_symbolic"


def analyze(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    samples = d["samples"]
    counts = Counter(categorize(s) for s in samples)
    total = len(samples)
    return {
        "task": d.get("task", "MATH-500"),
        "adapter": d.get("adapter"),
        "accuracy": d["accuracy"],
        "n_inspected": total,
        "n_total": d["n"],
        "categories": dict(counts),
        "pct": {k: f"{100*v/total:.0f}%" for k, v in counts.items()},
    }
