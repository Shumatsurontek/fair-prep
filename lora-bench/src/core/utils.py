"""Device, dtype, seeding, timing, memory."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_torch_dtype(device: str, base_cfg: dict) -> torch.dtype:
    key = f"dtype_{device}"
    name = base_cfg["model"].get(key, "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[name]
    # T4 / pre-Ampere: bf16 weights load but no Tensor Cores → slow.
    # Auto-fallback to fp16 (Tensor-Core-accelerated on Turing).
    if device == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[model] bf16 not supported on this GPU → using fp16")
        return torch.float16
    return dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


@dataclass
class Timer:
    name: str = "task"

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def peak_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            return None
    return None


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_env(device: str) -> dict:
    info = {"device": device, "torch": torch.__version__}
    if device == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info
