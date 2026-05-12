"""Hendrycks MATH per-domain SFT loader (for teacher 4B training).

Dataset: EleutherAI/hendrycks_math (configs: algebra, geometry, number_theory,
intermediate_algebra, prealgebra, precalculus, counting_and_probability).
Each example: {problem, level, type, solution}. Solution contains \\boxed{}.
"""
from __future__ import annotations

import re

from datasets import load_dataset

BOXED_FINAL = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_boxed(text: str) -> str | None:
    m = BOXED_FINAL.search(text)
    return m.group(1).strip() if m else None


def format_math_sft(example: dict, tokenizer) -> dict:
    question = example["problem"].strip()
    solution = example["solution"]
    final = _extract_boxed(solution) or ""
    cot = solution
    assistant = f"<think>\n{cot}\n</think>\n\nThe answer is: \\boxed{{{final}}}"
    messages = [
        {"role": "user", "content": question + "\n\nReason step by step; put final in \\boxed{}."},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def load_math_subset_for_sft(cfg: dict, tokenizer, with_eval: bool = False):
    dcfg = cfg["dataset"]
    raw = load_dataset(dcfg["name"], dcfg["config"], split=dcfg["train_split"])
    if dcfg.get("max_train_samples"):
        raw = raw.select(range(min(dcfg["max_train_samples"], len(raw))))

    if with_eval and dcfg.get("eval_holdout"):
        k = min(dcfg["eval_holdout"], len(raw) // 10)
        if k == 0:
            fmt = raw.map(
                lambda ex: format_math_sft(ex, tokenizer),
                remove_columns=raw.column_names, desc="Formatting MATH",
            )
            return fmt, None
        train_part = raw.select(range(len(raw) - k))
        eval_part = raw.select(range(len(raw) - k, len(raw)))
        train_fmt = train_part.map(
            lambda ex: format_math_sft(ex, tokenizer),
            remove_columns=train_part.column_names, desc="Formatting MATH train",
        )
        eval_fmt = eval_part.map(
            lambda ex: format_math_sft(ex, tokenizer),
            remove_columns=eval_part.column_names, desc="Formatting MATH eval",
        )
        return train_fmt, eval_fmt

    fmt = raw.map(
        lambda ex: format_math_sft(ex, tokenizer),
        remove_columns=raw.column_names, desc="Formatting MATH",
    )
    return fmt, None
