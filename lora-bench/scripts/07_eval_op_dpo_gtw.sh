#!/usr/bin/env bash
# Eval GTW checkpoint on GSM8K.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
ADAPTER="${ADAPTER:-./runs/qwen3_06b_op_dpo_gtw/final}"

mkdir -p results
./lb eval gsm8k \
    --cfg configs/sft_gsm8k.yaml \
    --adapter "$ADAPTER" \
    ${DEVICE:+--device "$DEVICE"} \
    --max-samples "$MAX_SAMPLES" \
    --out results/op_dpo_gtw.json
