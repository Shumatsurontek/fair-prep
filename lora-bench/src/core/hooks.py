"""Generic forward-hook activation collection.

Captures inputs (X) and outputs (Y) of named submodules across a batch.
Token activations are flattened to (N_tokens, d) and stored on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn


@dataclass
class ActivationStore:
    inputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    outputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def add(self, name: str, x: torch.Tensor, y: torch.Tensor) -> None:
        # Keep activations in their native dtype (typically bf16) on CPU to
        # bound RAM; OLS solver upcasts to fp32 locally per module.
        self.inputs.setdefault(name, []).append(x.detach().to("cpu"))
        self.outputs.setdefault(name, []).append(y.detach().to("cpu"))

    def stack(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {n: (torch.cat(self.inputs[n], dim=0), torch.cat(self.outputs[n], dim=0))
                for n in self.inputs}


def _flatten(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    return t


def register_hooks(model: nn.Module, target_names: Iterable[str], store: ActivationStore):
    target_set = set(target_names)
    handles = []
    for name, module in model.named_modules():
        if name in target_set:
            def make_hook(n):
                def _hook(_mod, args, output):
                    x = args[0] if isinstance(args, tuple) else args
                    store.add(n, _flatten(x), _flatten(output))
                return _hook
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def collect_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    target_names: list[str],
    device: str,
    max_length: int = 512,
    batch_size: int = 4,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run prompts through model; return {module_name: (X_in, Y_out)} on CPU."""
    from tqdm import tqdm

    store = ActivationStore()
    handles = register_hooks(model, target_names, store)
    model.eval()
    try:
        with torch.no_grad():
            bar = tqdm(range(0, len(prompts), batch_size), desc="calib forward")
            for i in bar:
                batch = prompts[i : i + batch_size]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()
    return store.stack()
