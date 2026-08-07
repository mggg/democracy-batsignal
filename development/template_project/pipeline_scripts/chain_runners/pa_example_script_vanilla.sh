#!/usr/bin/env bash

# This is a direct RustReCom CLI reference. Use pipeline_scripts/run_chains.py for normal batches.
# `chain` samples ordinary ReCom plans from the assignment stored on each graph node.
# Input flags identify the adjacency-data graph and its assignment and population columns.
# Chain flags set the seed, number of steps, population tolerance, and ReCom proposal variant.
# Output flags record the graph, metadata, and assignment stream together in a BENDL file.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

n_steps=1000
rng_seed=(42 43)
tol=0.01
assignment_col="seed_plan"
pop_col="total_pop_20"

graph_json="${PROJECT_ROOT}/JSON_dualgraphs/pa_dualgraph.json"
output_dir="${PROJECT_ROOT}/chain_outputs"

if [[ ! -f "$graph_json" ]]; then
    echo "Could not find graph JSON at: $graph_json" >&2
    exit 1
fi

mkdir -p "$output_dir"

for seed in "${rng_seed[@]}"; do
    bendl_file="${output_dir}/VANILLA_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol/./p}.bendl"

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
        --output-file "$bendl_file" \
        --overwrite-output \
        --show-progress
done
