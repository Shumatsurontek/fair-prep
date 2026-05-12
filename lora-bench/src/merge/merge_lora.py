"""Merge k PEFT LoRA adapters into one. Methods from arXiv:2509.24244.

Task vector per adapter = effective ΔW = B @ A (per target module). We merge
the ΔW task vectors then refactor back to (A, B) via truncated SVD at rank r.

Methods:
  - average: ΔW_m = mean_i ΔW_i
  - ta (task arithmetic): ΔW_m = Σ_i α_i · ΔW_i  (α uniform by default)
  - ties: trim small magnitudes per-param, resolve sign conflicts, mean kept
  - dare: drop with prob p, rescale 1/(1-p), then average

CLI lives in `src.cli.merge`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file


def _load_adapter(path: str | Path) -> tuple[LoraConfig, dict[str, torch.Tensor]]:
    p = Path(path)
    cfg = LoraConfig.from_pretrained(str(p))
    sd_path = p / "adapter_model.safetensors"
    if not sd_path.exists():
        sd_path = p / "adapter_model.bin"
        state = torch.load(sd_path, map_location="cpu")
    else:
        state = load_file(str(sd_path))
    return cfg, state


def _pair_keys(state: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
    """Pair lora_A / lora_B keys by their common module prefix."""
    pairs: dict[str, list[str]] = {}
    for k in state:
        if "lora_A" in k:
            base = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
            pairs.setdefault(base, []).append(k)
        elif "lora_B" in k:
            base = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
            pairs.setdefault(base, []).append(k)
    out = {}
    for base, ks in pairs.items():
        a = next((k for k in ks if "lora_A" in k), None)
        b = next((k for k in ks if "lora_B" in k), None)
        if a and b:
            out[base] = (a, b)
    return out


def _delta(state: dict[str, torch.Tensor], a_key: str, b_key: str) -> torch.Tensor:
    A = state[a_key].float()
    B = state[b_key].float()
    return B @ A  # (d_out, d_in)


def _refactor(delta: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated SVD back to (A, B) with rank r. ΔW ≈ B @ A."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    # ΔW ≈ U_r · diag(S_r) · Vh_r  →  B = U_r · diag(S_r), A = Vh_r
    B = U_r * S_r.unsqueeze(0)
    A = Vh_r
    return A, B


def _ties_trim(deltas: list[torch.Tensor], top_frac: float = 0.2) -> list[torch.Tensor]:
    """Keep top |x| values per delta (per-tensor)."""
    out = []
    for d in deltas:
        flat = d.abs().flatten()
        if flat.numel() == 0:
            out.append(d)
            continue
        k = max(1, int(top_frac * flat.numel()))
        thresh = torch.topk(flat, k, largest=True).values.min()
        mask = d.abs() >= thresh
        out.append(d * mask)
    return out


def _ties_resolve(deltas: list[torch.Tensor]) -> torch.Tensor:
    """Sign-conflict resolution then mean-of-kept."""
    stack = torch.stack(deltas, dim=0)            # (k, d_out, d_in)
    sign_sum = stack.sum(dim=0).sign()            # (d_out, d_in)
    keep = (stack.sign() == sign_sum.unsqueeze(0)).float()
    num = (stack * keep).sum(dim=0)
    den = keep.sum(dim=0).clamp_min(1.0)
    return num / den


def _dare_drop(deltas: list[torch.Tensor], p: float = 0.5, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in deltas:
        mask = (torch.rand(d.shape, generator=g) > p).float()
        out.append(d * mask / (1.0 - p))
    return out


def merge_deltas(deltas: list[torch.Tensor], method: str, **kw) -> torch.Tensor:
    if method == "average" or method == "ta":
        weights = kw.get("weights") or [1.0 / len(deltas)] * len(deltas)
        out = torch.zeros_like(deltas[0])
        for w, d in zip(weights, deltas):
            out += w * d
        return out
    if method == "ties":
        trimmed = _ties_trim(deltas, top_frac=kw.get("top_frac", 0.2))
        return _ties_resolve(trimmed)
    if method == "dare":
        dropped = _dare_drop(deltas, p=kw.get("p", 0.5), seed=kw.get("seed", 0))
        return torch.stack(dropped, dim=0).mean(dim=0)
    raise ValueError(f"unknown method: {method}")


def merge_adapters(
    adapter_paths: list[str],
    out_dir: str,
    method: str = "average",
    **kw,
) -> None:
    if not adapter_paths:
        raise ValueError("no adapters provided")
    cfg0, state0 = _load_adapter(adapter_paths[0])
    pairs0 = _pair_keys(state0)
    states = [state0] + [_load_adapter(p)[1] for p in adapter_paths[1:]]

    merged_state: dict[str, torch.Tensor] = {}
    for base, (a_key, b_key) in pairs0.items():
        deltas = []
        for s in states:
            if a_key not in s or b_key not in s:
                continue
            deltas.append(_delta(s, a_key, b_key))
        if not deltas:
            continue
        merged = merge_deltas(deltas, method, **kw)
        A_m, B_m = _refactor(merged, r=cfg0.r)
        merged_state[a_key] = A_m.to(state0[a_key].dtype).contiguous()
        merged_state[b_key] = B_m.to(state0[b_key].dtype).contiguous()

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    save_file(merged_state, str(out_p / "adapter_model.safetensors"))
    cfg0.save_pretrained(str(out_p))
    with open(out_p / "merge_info.json", "w") as f:
        json.dump({"method": method, "k": len(adapter_paths), "adapters": adapter_paths, "kwargs": kw}, f, indent=2)


