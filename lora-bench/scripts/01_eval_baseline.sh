#!/usr/bin/env bash
# Evaluate Qwen3-0.6B (no adapter) on GSM8K test.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"

mkdir -p results
./lb eval gsm8k \
    --cfg configs/sft_gsm8k.yaml \
    ${DEVICE:+--device "$DEVICE"} \
    --max-samples "$MAX_SAMPLES" \
    --out results/baseline.json
