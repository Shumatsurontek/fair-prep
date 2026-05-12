"""Generic eval loop: greedy generate + per-sample scoring + memory log."""
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from ..core.utils import Timer, peak_memory_mb, reset_peak_memory


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    eval_ds,
    device: str,
    score_fn: Callable[[str, dict], tuple[str | None, bool]],
    max_new_tokens: int = 512,
    sample_limit_full_output: int = 5,
) -> dict:
    """Generic eval loop.

    score_fn(generated_text, example) → (pred_str_or_None, is_correct).
    """
    model.eval()
    n_correct = 0
    n_total = 0
    total_tokens = 0
    samples = []
    reset_peak_memory(device)

    with Timer("gen") as t:
        for ex in tqdm(eval_ds, desc="eval"):
            inputs = tokenizer(ex["prompt"], return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_tokens = out[0, inputs["input_ids"].shape[1]:]
            total_tokens += int(gen_tokens.shape[0])
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred, ok = score_fn(text, ex)
            n_correct += int(ok)
            n_total += 1
            keep_full = len(samples) < sample_limit_full_output
            samples.append({
                "question": ex.get("question") if keep_full else None,
                "gold": ex.get("gold"),
                "pred": pred,
                "ok": ok,
                "output": text[:600] if keep_full else None,
            })

    return {
        "accuracy": n_correct / max(1, n_total),
        "n": n_total,
        "tokens_per_sec": total_tokens / max(t.elapsed, 1e-6),
        "wall_seconds": t.elapsed,
        "peak_memory_mb": peak_memory_mb(device),
        "samples": samples,
    }
