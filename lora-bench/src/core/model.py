"""Model & tokenizer loading. Applies LoRA or attaches existing adapter."""
from __future__ import annotations

from typing import Optional

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_torch_dtype


def load_tokenizer(cfg: dict):
    tok_cfg = cfg["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg["name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        padding_side=tok_cfg.get("padding_side", "right"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: dict, device: str):
    dtype = get_torch_dtype(device, cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        dtype=dtype,
        attn_implementation=cfg["model"].get("attn_implementation", "eager"),
    )
    model.to(device)
    if device == "cuda":
        model.config.use_cache = False
    return model


def apply_lora(model, cfg: dict):
    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )
    return get_peft_model(model, peft_cfg)


def load_model_with_lora(cfg: dict, device: str, adapter_path: Optional[str] = None):
    model = load_base_model(cfg, device)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        model = apply_lora(model, cfg)
    return model


def merge_and_unload(peft_model):
    return peft_model.merge_and_unload()
