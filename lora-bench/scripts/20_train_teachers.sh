#!/usr/bin/env bash
# Train Qwen3-4B teacher LoRA experts, one per MATH domain.
# DO NOT run while another GPU job is active (check `nvidia-smi`).
# Env vars: DEVICE, MAX_STEPS, MAX_TRAIN_SAMPLES, DOMAINS (space-separated)
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
DOMAINS="${DOMAINS:-algebra geometry number_theory}"

for d in $DOMAINS; do
    cfg="configs/teacher_4b_${d}.yaml"
    if [[ ! -f "$cfg" ]]; then
        echo "[skip] $cfg not found — create it (template: teacher_4b_algebra.yaml)"
        continue
    fi
    echo "=== teacher: $d ==="
    ./lb train sft \
        --cfg "$cfg" \
        ${DEVICE:+--device "$DEVICE"} \
        --max-steps "$MAX_STEPS" \
        ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"}
done
