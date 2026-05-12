"""Smoke test: gsm8k_accuracy_reward on a few real GSM8K samples."""
from __future__ import annotations

from datasets import load_dataset

from ..gsm8k import gsm8k_accuracy_reward
from ...data import extract_gsm8k_answer


def main(n: int = 20) -> None:
    raw = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    prompts, gold = [], []
    completions_correct, completions_wrong = [], []
    for ex in raw:
        _, final = extract_gsm8k_answer(ex["answer"])
        prompts.append(ex["question"])
        gold.append(final)
        completions_correct.append(f"some reasoning #### {final}")
        completions_wrong.append("garbage answer #### 99999999")

    r_ok = gsm8k_accuracy_reward(prompts, completions_correct, gold)
    r_no = gsm8k_accuracy_reward(prompts, completions_wrong, gold)
    assert all(r == 1.0 for r in r_ok), f"correct completions scored {r_ok}"
    assert all(r == 0.0 for r in r_no), f"wrong completions scored {r_no}"
    print(f"[OK] {n} correct → {sum(r_ok):.0f}, {n} wrong → {sum(r_no):.0f}")


if __name__ == "__main__":
    main()
