#!/usr/bin/env bash
# OnPolicy-DPO-GTW custom trainer (research). Run AFTER SFT.
# Env vars: DEVICE, MAX_STEPS, MAX_TRAIN_PROMPTS, NUM_GEN
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_TRAIN_PROMPTS="${MAX_TRAIN_PROMPTS:-}"
SFT_CKPT="${SFT_CKPT:-./runs/qwen3_06b_sft_gsm8k/final}"

./lb train gtw \
    --cfg configs/op_dpo_gtw.yaml \
    --sft-checkpoint "$SFT_CKPT" \
    ${DEVICE:+--device "$DEVICE"} \
    ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
    ${MAX_TRAIN_PROMPTS:+--max-train-prompts "$MAX_TRAIN_PROMPTS"}
