#!/usr/bin/env bash
# Train SFT LoRA on GSM8K.
# Env vars: DEVICE, MAX_STEPS, MAX_TRAIN_SAMPLES
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

./lb train sft \
    --cfg configs/sft_gsm8k.yaml \
    ${DEVICE:+--device "$DEVICE"} \
    --max-steps "$MAX_STEPS" \
    ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"}
