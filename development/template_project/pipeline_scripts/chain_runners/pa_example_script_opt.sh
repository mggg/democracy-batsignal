#!/usr/bin/env bash

# This is a direct RustReCom CLI reference. Use pipeline_scripts/run_chains.py for normal batches.
# `tilted` runs ReCom while favoring proposals that improve the selected objective score.
# Input and chain flags have the same meaning as in the ordinary `chain` example.
# `--objective` loads the score definition, and `--maximize true` makes larger scores preferable.
# The BENDL file records plans; the companion CSV records objective values for analysis.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

n_steps=1000
rng_seed=(42 43)
tol=0.01
assignment_col="seed_plan"
pop_col="total_pop_20"

OBJECTIVE_FILE="${PROJECT_ROOT}/pipeline_scripts/chain_runners/rustrecom_objectives/gingles_partial.json"

graph_json="${PROJECT_ROOT}/JSON_dualgraphs/pa_dualgraph.json"
output_dir="${PROJECT_ROOT}/chain_outputs"

if [[ ! -f "$graph_json" ]]; then
    echo "Could not find graph JSON at: $graph_json" >&2
    exit 1
fi

mkdir -p "$output_dir"
tol_label=${tol/./p}

for seed in "${rng_seed[@]}"; do
    prefix="GINGLES_PARTIAL_PA__STEPS_${n_steps}__RNGSEED_${seed}__TOL_${tol_label}"
    bendl_file="${output_dir}/${prefix}.bendl"
    scores_file="${output_dir}/${prefix}_scores.csv"

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
        --output-file "$bendl_file" \
        --scores-output-file "$scores_file" \
        --overwrite-output \
        --show-progress
done
