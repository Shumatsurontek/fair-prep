"""Generic eval loop with batched generation."""
from __future__ import annotations

from contextlib import nullcontext
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
    batch_size: int = 8,
) -> dict:
    """Batched generation eval loop.

    score_fn(generated_text, example) → (pred_str_or_None, is_correct).
    """
    model.eval()
    # KV-cache massively speeds up autoregressive generation.
    if hasattr(model, "config"):
        model.config.use_cache = True
    # Left-pad so generation slicing is consistent across batch.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    n_correct = 0
    n_total = 0
    total_tokens = 0
    samples: list[dict] = []
    reset_peak_memory(device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    n = len(eval_ds)
    indices = list(range(n))

    with Timer("gen") as t:
        pbar = tqdm(total=n, desc="eval")
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = [eval_ds[i] for i in batch_idx]
            prompts = [ex["prompt"] for ex in batch]

            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(device)

            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else nullcontext())
            with ctx:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            prompt_len = enc["input_ids"].shape[1]
            gen_only = out[:, prompt_len:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for ex, gen_ids, text in zip(batch, gen_only, texts):
                total_tokens += int((gen_ids != pad_id).sum())
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

            pbar.update(len(batch))
            pbar.set_postfix(acc=f"{n_correct/max(n_total,1):.3f}")
        pbar.close()

    tokenizer.padding_side = original_padding_side
    return {
        "accuracy": n_correct / max(1, n_total),
        "n": n_total,
        "tokens_per_sec": total_tokens / max(t.elapsed, 1e-6),
        "wall_seconds": t.elapsed,
        "peak_memory_mb": peak_memory_mb(device),
        "batch_size": batch_size,
        "samples": samples,
    }
