"""Standard DPO training (TRL DPOTrainer), init from SFT LoRA checkpoint."""
from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

from ..core.model import load_base_model, load_tokenizer
from ..core.utils import count_trainable, ensure_dir, log_env, save_json
from ..data.preferences import load_preference_dataset


def run(
    cfg: dict,
    device: str,
    sft_checkpoint: str,
    max_steps: int = -1,
    max_train_samples: int | None = None,
) -> Path:
    if max_train_samples is not None:
        cfg["dataset"]["max_train_samples"] = max_train_samples

    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["run_name"])
    save_json({"env": log_env(device), "cfg": cfg, "sft_ckpt": sft_checkpoint},
              run_dir / "run_meta.json")

    tokenizer = load_tokenizer(cfg)
    base_policy = load_base_model(cfg, device)
    policy = PeftModel.from_pretrained(base_policy, sft_checkpoint, is_trainable=True)

    if cfg["train"].get("gradient_checkpointing", False):
        policy.gradient_checkpointing_enable()
        policy.enable_input_require_grads()

    trainable, total = count_trainable(policy)
    print(f"[lora-policy] trainable={trainable:,} / total={total:,}")

    bf16 = bool(cfg["train"].get("bf16", False)) and device == "cuda"
    fp16 = bool(cfg["train"].get("fp16", False)) and device == "cuda" and not bf16
    if device == "mps":
        bf16 = fp16 = False

    train_ds, eval_ds = load_preference_dataset(cfg, tokenizer)
    print(f"[data] dpo_train_size={len(train_ds)}  eval_size={len(eval_ds) if eval_ds else 0}")

    has_eval = eval_ds is not None
    dpo_args = DPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"].get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_steps=cfg["train"].get("warmup_steps", 0),
        weight_decay=cfg["train"]["weight_decay"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        optim=cfg["train"]["optim"],
        logging_steps=cfg["train"]["logging_steps"],
        eval_strategy=cfg["train"].get("eval_strategy", "no") if has_eval else "no",
        eval_steps=cfg["train"].get("eval_steps", 100),
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
        beta=cfg["dpo"]["beta"],
        loss_type=cfg["dpo"]["loss_type"],
        max_length=cfg["dpo"]["max_length"],
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

    trainer = DPOTrainer(
        model=policy, ref_model=None, args=dpo_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer, callbacks=callbacks,
    )
    trainer.train()
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] saved to {final_dir}")
    return final_dir
