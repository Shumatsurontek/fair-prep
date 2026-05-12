#!/usr/bin/env bash
# Run ONLY OnPolicy-DPO-GTW (train + eval) and regenerate REPORT.md.
# Does NOT touch baseline/sft/dpo .json files — only adds op_dpo_gtw.json.
#
# Env vars: DEVICE, MAX_STEPS, MAX_TRAIN_PROMPTS, MAX_SAMPLES, SFT_CKPT
set -euo pipefail
cd "$(dirname "$0")"

bash 06_train_op_dpo_gtw.sh
bash 07_eval_op_dpo_gtw.sh

echo "==> regenerating REPORT.md (existing baseline/sft/dpo preserved)"
cd ..
./lb report
