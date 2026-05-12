"""GSM8K dataset formatting for SFT + eval.

Qwen3 thinking-mode format: assistant response wraps CoT in <think>...</think>
followed by `#### N` answer marker so eval extraction stays trivial.
"""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset

from . import extract_gsm8k_answer, format_qwen_prompt_only


def format_qwen_chat_sft(example: dict, tokenizer) -> dict:
    question = example["question"].strip()
    cot, final = extract_gsm8k_answer(example["answer"])
    assistant = f"<think>\n{cot}\n</think>\n\nThe answer is: #### {final}"
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def load_gsm8k_for_sft(cfg: dict, tokenizer, with_eval: bool = False):
    """Returns (train_ds, eval_ds_or_None). Holdout carved from train tail."""
    dcfg = cfg["dataset"]
    raw = load_dataset(dcfg["name"], dcfg["config"], split=dcfg["train_split"])
    if dcfg.get("max_train_samples"):
        raw = raw.select(range(min(dcfg["max_train_samples"], len(raw))))

    if with_eval and dcfg.get("eval_holdout"):
        k = min(dcfg["eval_holdout"], len(raw) // 10)
        train_part = raw.select(range(len(raw) - k))
        eval_part = raw.select(range(len(raw) - k, len(raw)))
        train_fmt = train_part.map(
            lambda ex: format_qwen_chat_sft(ex, tokenizer),
            remove_columns=train_part.column_names, desc="Formatting SFT train",
        )
        eval_fmt = eval_part.map(
            lambda ex: format_qwen_chat_sft(ex, tokenizer),
            remove_columns=eval_part.column_names, desc="Formatting SFT eval",
        )
        return train_fmt, eval_fmt

    fmt = raw.map(
        lambda ex: format_qwen_chat_sft(ex, tokenizer),
        remove_columns=raw.column_names, desc="Formatting SFT",
    )
    return fmt, None


def load_gsm8k_for_eval(cfg: dict, tokenizer, max_samples: Optional[int] = None) -> Dataset:
    dcfg = cfg["dataset"]
    raw = load_dataset(dcfg["name"], dcfg["config"], split=dcfg["eval_split"])
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _proc(ex):
        prompt = format_qwen_prompt_only(ex["question"], tokenizer)
        _, final = extract_gsm8k_answer(ex["answer"])
        return {"prompt": prompt, "gold": final, "question": ex["question"]}

    return raw.map(_proc, remove_columns=raw.column_names, desc="Formatting eval")
