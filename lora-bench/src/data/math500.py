"""MATH-500 eval dataset loader (HuggingFaceH4/MATH-500)."""
from __future__ import annotations

from datasets import load_dataset

from . import format_qwen_prompt_only


def load_math500(tokenizer, max_samples: int | None = None):
    raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _proc(ex):
        prompt = format_qwen_prompt_only(
            ex["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
            tokenizer,
        )
        return {"prompt": prompt, "gold": str(ex["answer"]), "question": ex["problem"]}

    return raw.map(_proc, remove_columns=raw.column_names, desc="Formatting MATH-500")
