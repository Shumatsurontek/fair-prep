#!/usr/bin/env bash
# Evaluate SFT-LoRA checkpoint on GSM8K test.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
ADAPTER="${ADAPTER:-./runs/qwen3_06b_sft_gsm8k/final}"

mkdir -p results
./lb eval gsm8k \
    --cfg configs/sft_gsm8k.yaml \
    --adapter "$ADAPTER" \
    ${DEVICE:+--device "$DEVICE"} \
    --max-samples "$MAX_SAMPLES" \
    --out results/sft.json
