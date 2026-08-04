#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Project root is the parent of this script's folder
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)

plan_name="district"
n_steps=1000
seed=42
tol=0.01
pop_col="TOTPOP"

json_file="${PROJECT_ROOT}/JSON_dualgraphs/gerrymandria.json"
output_dir="${PROJECT_ROOT}/chain_outputs"

if [[ ! -f "$json_file" ]]; then
    echo "Could not find graph JSON at: $json_file" >&2
    exit 1
fi

mkdir -p "$output_dir"
final_output_file="${output_dir}/gerrymandria_chain_${n_steps}_steps.jsonl.ben"

frcw \
    --assignment-col $plan_name \
    --graph-json "$json_file" \
    --n-steps $n_steps \
    --pop-col $pop_col \
    --rng-seed $seed \
    --tol $tol \
    --variant district-pairs-rmst \
    --writer ben \
    --batch-size 1 \
    --n-threads 1 \
    --output-file "${final_output_file}"
