#!/usr/bin/env bash
# lm-evaluation-harness: gsm8k_cot, minerva_math, mmlu_pro
# Compares DPO (final) vs GTW (checkpoint-100) on Qwen3-0.6B.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-Qwen/Qwen3-0.6B}"
DEVICE="${DEVICE:-mps}"
LIMIT="${LIMIT:-500}"
TASKS="${TASKS:-gsm8k_cot,minerva_math,mmlu_pro}"
BATCH="${BATCH:-1}"

ADAPTER="${ADAPTER:?adapter path required, e.g. ./runs/qwen3_06b_dpo_math/final}"
OUT="${OUT:?output path required, e.g. results/lm_eval_dpo}"

mkdir -p "$OUT"

uv run lm_eval \
    --model hf \
    --model_args "pretrained=${BASE},peft=${ADAPTER},trust_remote_code=True,dtype=float16" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size "$BATCH" \
    --limit "$LIMIT" \
    --output_path "$OUT" \
    --log_samples
