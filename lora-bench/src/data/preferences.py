"""DPO preference dataset: (prompt, chosen, rejected) format for TRL."""
from __future__ import annotations

from datasets import load_dataset

from . import format_qwen_prompt_only


def load_preference_dataset(cfg: dict, tokenizer):
    dcfg = cfg["dataset"]
    raw = load_dataset(dcfg["name"], split=dcfg["train_split"])
    if dcfg.get("max_train_samples"):
        raw = raw.select(range(min(dcfg["max_train_samples"], len(raw))))

    p_field = dcfg["prompt_field"]
    c_field = dcfg["chosen_field"]
    r_field = dcfg["rejected_field"]
    holdout = dcfg.get("eval_holdout", 0)

    def _proc(ex):
        question = ex[p_field]
        if isinstance(question, list):
            rendered = tokenizer.apply_chat_template(
                question, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = format_qwen_prompt_only(str(question), tokenizer)
        chosen = ex[c_field]
        rejected = ex[r_field]
        if isinstance(chosen, list):
            chosen = chosen[-1]["content"] if chosen else ""
        if isinstance(rejected, list):
            rejected = rejected[-1]["content"] if rejected else ""
        return {"prompt": rendered, "chosen": chosen, "rejected": rejected}

    processed = raw.map(_proc, remove_columns=raw.column_names, desc="Formatting DPO")
    if holdout and holdout < len(processed):
        train_part = processed.select(range(len(processed) - holdout))
        eval_part = processed.select(range(len(processed) - holdout, len(processed)))
        return train_part, eval_part
    return processed, None
