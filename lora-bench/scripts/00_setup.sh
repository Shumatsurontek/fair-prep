#!/usr/bin/env bash
# Setup: install deps via uv, pre-download model & datasets to local cache.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not installed. https://docs.astral.sh/uv/" >&2
    exit 1
fi

cd ..
echo "==> uv sync (root project)"
uv sync
cd lora-bench

# Pre-download model + tokenizer (avoid first-run delay)
echo "==> pre-downloading Qwen3-0.6B"
uv run python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen3-0.6B"
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
print("ok")
PY

echo "==> pre-downloading GSM8K"
uv run python - <<'PY'
from datasets import load_dataset
load_dataset("openai/gsm8k", "main")
print("ok")
PY

echo "==> pre-downloading Math-Step-DPO-10K (preferences)"
uv run python - <<'PY'
from datasets import load_dataset
try:
    load_dataset("xinlai/Math-Step-DPO-10K")
    print("ok")
except Exception as e:
    print(f"WARN: could not preload preference dataset: {e}")
PY

echo "Done."
