#!/usr/bin/env bash
# Merge k transferred LoRAs (varying k & method), eval, fit scaling law.
# DO NOT run while another GPU job is active.
# Env vars: METHODS, KS, DOMAINS
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS="${METHODS:-average ties}"
KS="${KS:-1 2 4 6 7}"
ALL_DOMAINS=(algebra geometry number_theory intermediate_algebra prealgebra precalculus counting_and_probability)

mkdir -p runs/merged results/scaling

for method in $METHODS; do
    pairs_k=()
    pairs_L=()
    for k in $KS; do
        adapters=()
        for i in $(seq 0 $((k - 1))); do
            d="${ALL_DOMAINS[$i]}"
            p="runs/transferred/${d}/adapter"
            [[ -d "$p" ]] && adapters+=("$p")
        done
        if [[ ${#adapters[@]} -lt $k ]]; then
            echo "[skip] k=$k method=$method — only ${#adapters[@]} adapters available"
            continue
        fi
        out="runs/merged/${method}_k${k}/adapter"
        echo "=== merge k=$k method=$method ==="
        adapter_args=()
        for a in "${adapters[@]}"; do adapter_args+=(--adapter "$a"); done
        ./lb merge --method "$method" "${adapter_args[@]}" --out "$out"
        eval_out="results/scaling/${method}_k${k}.json"
        ./lb eval math500 --adapter "$out" --out "$eval_out" || true
        # Parse CE; expect 'loss' or 'ce' field — fallback to script's own extractor downstream
        L=$(python -c "import json,sys;d=json.load(open('$eval_out'));print(d.get('loss', d.get('ce', d.get('eval_loss', 0.0))))" 2>/dev/null || echo "0.0")
        pairs_k+=("$k")
        pairs_L+=("$L")
    done
    # Write k_L pairs for scaling-law fit
    python -c "
import json
ks = [${pairs_k[*]/%/,}]
ls = [${pairs_L[*]/%/,}]
json.dump({'k': ks, 'loss': ls}, open('results/scaling/${method}_pairs.json','w'), indent=2)
"
    ./lb fit k \
        --pairs "results/scaling/${method}_pairs.json" \
        --out "results/scaling/${method}_fit.json"
done
