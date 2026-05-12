#!/usr/bin/env bash
# DPO training, init from SFT LoRA checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
SFT_CKPT="${SFT_CKPT:-./runs/qwen3_06b_sft_gsm8k/final}"

./lb train dpo \
    --cfg configs/dpo_pref.yaml \
    --sft-checkpoint "$SFT_CKPT" \
    ${DEVICE:+--device "$DEVICE"} \
    --max-steps "$MAX_STEPS" \
    ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"}
