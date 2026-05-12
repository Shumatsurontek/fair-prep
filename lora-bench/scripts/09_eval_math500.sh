#!/usr/bin/env bash
# MATH-500 eval: compare adapters on out-of-domain math.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
ADAPTER="${ADAPTER:?adapter path required}"
OUT="${OUT:?output JSON path required}"

mkdir -p results
./lb eval math500 \
    --cfg configs/sft_gsm8k.yaml \
    --adapter "$ADAPTER" \
    ${DEVICE:+--device "$DEVICE"} \
    --max-samples "$MAX_SAMPLES" \
    --out "$OUT"
