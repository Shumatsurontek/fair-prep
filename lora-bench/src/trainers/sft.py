"""SFT LoRA training entry-point (no CLI). Driven by config dict."""
from __future__ import annotations

from pathlib import Path

from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from ..core.model import apply_lora, load_base_model, load_tokenizer
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


def run(cfg: dict, device: str, max_steps: int = -1, max_train_samples: int | None = None) -> Path:
    if max_train_samples is not None:
        cfg["dataset"]["max_train_samples"] = max_train_samples

    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["run_name"])
    save_json({"env": log_env(device), "cfg": cfg}, run_dir / "run_meta.json")

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg, device)
    model = apply_lora(model, cfg)

    if cfg["train"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainable, total = count_trainable(model)
    print(f"[lora] trainable={trainable:,} / total={total:,}  ({100*trainable/total:.3f}%)")

    loader = _select_loader(cfg)
    train_ds, eval_ds = loader(cfg, tokenizer, with_eval=True)
    print(f"[data] train_size={len(train_ds)}  eval_size={len(eval_ds) if eval_ds else 0}")

    bf16 = bool(cfg["train"].get("bf16", False)) and device == "cuda"
    fp16 = bool(cfg["train"].get("fp16", False)) and device == "cuda" and not bf16
    if device == "mps":
        bf16 = fp16 = False

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
        optim=cfg["train"]["optim"],
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=cfg["train"].get("eval_strategy", "no") if has_eval else "no",
        eval_steps=cfg["train"].get("eval_steps", 200),
        save_strategy=cfg["train"].get("save_strategy", "steps"),
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", False) and has_eval,
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg["train"].get("greater_is_better", False),
        bf16=bf16,
        fp16=fp16,
        report_to=cfg["train"].get("report_to", "tensorboard"),
        max_steps=max_steps,
        dataset_text_field="text",
        max_length=cfg["tokenizer"]["max_length"],
        packing=cfg["train"].get("packing", False),
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", False),
        seed=cfg["seed"],
        run_name=cfg["run_name"],
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
    print(f"[done] saved to {final_dir}")
    return final_dir
