#!/usr/bin/env bash
# Full pipeline: baseline eval -> SFT train -> SFT eval -> DPO train -> DPO eval
#                                          \-> GTW train -> GTW eval -> report
# Disable GTW stage with SKIP_GTW=1.
set -euo pipefail
cd "$(dirname "$0")"

bash 01_eval_baseline.sh
bash 02_train_sft.sh
bash 03_eval_sft.sh
bash 04_train_dpo.sh
bash 05_eval_dpo.sh

if [ "${SKIP_GTW:-0}" != "1" ]; then
    bash 06_train_op_dpo_gtw.sh
    bash 07_eval_op_dpo_gtw.sh
fi

echo "==> generating report"
cd ..
./lb report
