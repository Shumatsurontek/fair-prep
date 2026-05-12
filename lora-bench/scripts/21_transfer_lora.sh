#!/usr/bin/env bash
# Project teacher LoRA (Qwen3-4B) → student (Qwen3-0.6B) hidden space.
# DO NOT run while another GPU job is active.
# Env vars: DEVICE, N_CALIB, RIDGE, DOMAINS, CALIB (path)
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
N_CALIB="${N_CALIB:-256}"
RIDGE="${RIDGE:-0.0001}"
DOMAINS="${DOMAINS:-algebra geometry number_theory}"
CALIB="${CALIB:-data/calib_math.jsonl}"
TEACHER="${TEACHER:-Qwen/Qwen3-4B-Instruct-2507}"
STUDENT="${STUDENT:-Qwen/Qwen3-0.6B}"

if [[ ! -f "$CALIB" ]]; then
    echo "[error] calibration file missing: $CALIB"
    echo "  build via: uv run python -m src.transfer.calibration_build (TODO) or hand-curate"
    exit 1
fi

for d in $DOMAINS; do
    adapter="runs/teacher_4b/${d}/adapter"
    out="runs/transferred/${d}/adapter"
    if [[ ! -d "$adapter" ]]; then
        echo "[skip] $adapter missing — run 20_train_teachers.sh first"
        continue
    fi
    echo "=== transfer: $d ==="
    ./lb transfer \
        --teacher "$TEACHER" \
        --student "$STUDENT" \
        --adapter "$adapter" \
        --calib "$CALIB" \
        --out "$out" \
        --device "$DEVICE" \
        --n-calib "$N_CALIB" \
        --ridge "$RIDGE"
done
