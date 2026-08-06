#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Project root is the parent of this script's folder
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)

n_steps=100000
rng_seed=(42 43)
tol=0.01
assignment_col="seed_plan"
pop_col="total_pop_20"

graph_json="${PROJECT_ROOT}/JSON_dualgraphs/pa_dualgraph.json"
output_dir="${PROJECT_ROOT}/chain_outputs"
log_dir="${PROJECT_ROOT}/chain_logs"
output_file="${output_dir}/VANILLA_PA__STEPS_${n_steps}__RNGSEED_${rng_seed}__TOL_${tol}.bendl"

if [[ ! -f "$graph_json" ]]; then
    echo "Could not find graph JSON at: $graph_json" >&2
    exit 1
fi

# Make the output and log directories if they don't exist
mkdir -p "$output_dir"
mkdir -p "$log_dir"

for seed in "${rng_seed[@]}"; do
    output_file="${output_dir}/VANILLA_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol/./p}.bendl"

    echo "Running rustrecom chain with seed: $seed ..."

    rustrecom chain \
        --assignment-col "$assignment_col" \
        --graph-json "$graph_json" \
        --n-steps "$n_steps" \
        --pop-col "$pop_col" \
        --rng-seed "$seed" \
        --tol "$tol" \
        --variant district-pairs-mst \
        --writer bendl \
        --output-file "$output_file" \
        --overwrite-output \
        --show-progress
done

# ———————————————————————————————————————————————————————————————
# UNCOMMENT THE FOLLOWING SECTION TO RUN CHAINS IN THE BACKGROUND
# ———————————————————————————————————————————————————————————————

# for seed in "${rng_seed[@]}"; do
#     output_file="${output_dir}/VANILLA_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol/./p}.bendl"
#     log_file="${log_dir}/VANILLA_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol/./p}.log"
#
#     echo "Running rustrecom chain with seed: $seed in the background..."
#
#     rustrecom chain \
#         --assignment-col "$assignment_col" \
#         --graph-json "$graph_json" \
#         --n-steps "$n_steps" \
#         --pop-col "$pop_col" \
#         --rng-seed "$seed" \
#         --tol "$tol" \
#         --variant district-pairs-mst \
#         --writer bendl \
#         --output-file "$output_file" \
#         --overwrite-output > "$log_file" 2>&1 &
# done
