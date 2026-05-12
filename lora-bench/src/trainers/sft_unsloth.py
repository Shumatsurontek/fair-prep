"""Unsloth-accelerated SFT (2-4× faster + lower VRAM).

Drop-in alternative to `trainers.sft.run`. Selected via `lb train sft --fast`.
Auto-raises ImportError if unsloth not installed → caller should fall back.

Unsloth requires CUDA (Ampere or newer). MPS / CPU → use `trainers.sft.run`.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from ..core.utils import count_trainable, ensure_dir, log_env, save_json
from ..data.gsm8k import load_gsm8k_for_sft
from ..data.math_subset import load_math_subset_for_sft


def _select_loader(cfg: dict):
    name = cfg["dataset"]["name"]
    if "gsm8k" in name:
        return load_gsm8k_for_sft
    if "hendrycks_math" in name or "MATH" in name:
        return load_math_subset_for_sft
    raise ValueError(f"no SFT loader for dataset {name!r}")


def run(
    cfg: dict,
    device: str,
    max_steps: int = -1,
    max_train_samples: int | None = None,
    load_in_4bit: bool = True,
) -> Path:
    """SFT with unsloth's FastLanguageModel. Falls back on ImportError."""
    if device != "cuda":
        raise RuntimeError("unsloth requires CUDA. Use `lb train sft` (no --fast).")
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "unsloth not installed. Install via "
            '`uv pip install "unsloth[cu121-torch24] @ git+https://github.com/unslothai/unsloth.git"` '
            "or run colab/setup.sh."
        ) from e

    if max_train_samples is not None:
        cfg["dataset"]["max_train_samples"] = max_train_samples

    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["run_name"])
    save_json({"env": log_env(device), "cfg": cfg, "backend": "unsloth"},
              run_dir / "run_meta.json")

    model_name = cfg["model"]["name"]
    max_seq = int(cfg["tokenizer"]["max_length"])
    # T4 / older = Turing → bf16 unsupported. Detect & fall back to fp16.
    bf16_ok = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"[unsloth] bf16_supported={bf16_ok}  dtype={dtype}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lcfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lcfg["r"],
        target_modules=lcfg["target_modules"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        bias=lcfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 42),
    )

    trainable, total = count_trainable(model)
    print(f"[unsloth-lora] trainable={trainable:,} / total={total:,}  ({100*trainable/total:.3f}%)")

    loader = _select_loader(cfg)
    train_ds, eval_ds = loader(cfg, tokenizer, with_eval=True)
    print(f"[data] train_size={len(train_ds)}  eval_size={len(eval_ds) if eval_ds else 0}")

    has_eval = eval_ds is not None
    train_args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"].get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_steps=cfg["train"].get("warmup_steps", 0),
        weight_decay=cfg["train"]["weight_decay"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        optim=cfg["train"].get("optim", "adamw_8bit"),
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=cfg["train"].get("eval_strategy", "no") if has_eval else "no",
        eval_steps=cfg["train"].get("eval_steps", 200),
        save_strategy=cfg["train"].get("save_strategy", "steps"),
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", False) and has_eval,
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg["train"].get("greater_is_better", False),
        bf16=bf16_ok,
        fp16=not bf16_ok,
        report_to=cfg["train"].get("report_to", "tensorboard"),
        max_steps=max_steps,
        dataset_text_field="text",
        max_length=max_seq,
        packing=cfg["train"].get("packing", False),
        seed=cfg["seed"],
        run_name=cfg["run_name"] + "_unsloth",
    )

    callbacks = []
    es = cfg.get("early_stopping")
    if es and has_eval:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=es.get("patience", 3),
            early_stopping_threshold=es.get("threshold", 0.0),
        ))

    trainer = SFTTrainer(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, callbacks=callbacks,
    )
    trainer.train()
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] unsloth saved → {final_dir}")
    return final_dir
