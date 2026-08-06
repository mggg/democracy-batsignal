#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Project root is the parent of this script's folder
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)

n_steps=1000
rng_seed=(42 43)
tol=0.01
assignment_col="seed_plan"
pop_col="total_pop_20"

OBJECTIVE_FILE="${PROJECT_ROOT}/pipeline_scripts/rustrecom_objectives/gingles_partial.json"

graph_json="${PROJECT_ROOT}/JSON_dualgraphs/pa_dualgraph.json"
output_dir="${PROJECT_ROOT}/chain_outputs"
log_dir="${PROJECT_ROOT}/chain_logs"

if [[ ! -f "$graph_json" ]]; then
    echo "Could not find graph JSON at: $graph_json" >&2
    exit 1
fi

# Make the output and log directories if they don't exist
mkdir -p "$output_dir"
mkdir -p "$log_dir"

for seed in "${rng_seed[@]}"; do
    prefix="GINGLES_PARTIAL_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol}"
    output_file="${output_dir}/${prefix}.bendl"
    log_file="${log_dir}/${prefix}.log"

    echo "Running rustrecom tilted with seed: $seed ..."

    rustrecom tilted \
        --assignment-col "$assignment_col" \
        --graph-json "$graph_json" \
        --n-steps "$n_steps" \
        --pop-col "$pop_col" \
        --rng-seed "$seed" \
        --tol "$tol" \
        --objective "${OBJECTIVE_FILE}" \
        --maximize true \
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
#     prefix="GINGLES_PARTIAL_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol}"
#     output_file="${output_dir}/${prefix}.bendl"
#     log_file="${log_dir}/${prefix}.log"
#
#     echo "Running rustrecom tilted with seed: $seed in the background..."
#
#     rustrecom tilted \
#         --assignment-col "$assignment_col" \
#         --graph-json "$graph_json" \
#         --n-steps "$n_steps" \
#         --pop-col "$pop_col" \
#         --rng-seed "$seed" \
#         --tol "$tol" \
#         --objective "${OBJECTIVE_FILE}" \
#         --maximize true \
#         --variant district-pairs-mst \
#         --writer bendl \
#         --output-file "$output_file" \
#         --overwrite-output > "$log_file" 2>&1 &
# done
