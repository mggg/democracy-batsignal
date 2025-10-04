#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
# Change this as needed to get the top level directory of the repo
TOPDIR=$(realpath "${SCRIPT_DIR}")

export PYTHONHASHSEED=0
# source .env # <- This will also work

rng_seeds=(42 43 44)
n_steps=1000

for seed in "${rng_seeds[@]}"; do
    uv run "${TOPDIR}/pipeline_scripts/example_cli.py" \
        --graph-path "${TOPDIR}/JSON_dualgraphs/gerrymandria.json" \
        --output-path "${TOPDIR}/chain_outputs/gerrymandria_chain_${n_steps}_steps_seed${seed}.jsonl" \
        --starting-plan "district" \
        --pop-col "TOTPOP" \
        --rng-seed $seed \
        --population-tolerance 0.01 \
        --total-steps $n_steps > "./chain_logs/log_simple_rng_seed_$seed.log" 2>&1
done
