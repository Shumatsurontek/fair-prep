"""Datasets + canonical answer parsing helpers.

Shared helpers (regex, normalize, chat template wrappers) live here so all
consumers (eval, rewards, trainers) import from a single canonical place.
"""
from __future__ import annotations

import re
from typing import Optional


GSM8K_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_gsm8k_answer(answer_field: str) -> tuple[str, str]:
    """Split GSM8K 'answer' field into (cot, final_number_str)."""
    m = GSM8K_ANSWER_RE.search(answer_field)
    if not m:
        return answer_field.strip(), ""
    return answer_field[: m.start()].strip(), m.group(1).replace(",", "")


def parse_predicted_answer(text: str) -> Optional[str]:
    """Extract '#### N' from model output; fallback to last number."""
    m = GSM8K_ANSWER_RE.search(text)
    if not m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else None
    return m.group(1).replace(",", "")


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else f"{f}"
    except ValueError:
        return s


def format_qwen_prompt_only(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question.strip()}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
