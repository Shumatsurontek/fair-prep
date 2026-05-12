"""YAML config loading + nested merge."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_cfg(base: dict, override: dict) -> dict:
    """Nested-dict merge (override wins). Lists replaced entirely."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def load_run_cfg(base_path: str, override_path: str | None = None) -> dict:
    base = load_yaml(base_path)
    if override_path:
        return merge_cfg(base, load_yaml(override_path))
    return base
