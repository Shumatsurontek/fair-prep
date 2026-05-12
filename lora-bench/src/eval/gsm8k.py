"""GSM8K scorer + dataset binding."""
from __future__ import annotations

from ..data import normalize_number, parse_predicted_answer
from ..data.gsm8k import load_gsm8k_for_eval


def score(text: str, example: dict) -> tuple[str | None, bool]:
    pred = normalize_number(parse_predicted_answer(text))
    gold = normalize_number(example["gold"])
    ok = pred is not None and gold is not None and pred == gold
    return pred, ok


load_dataset = load_gsm8k_for_eval
