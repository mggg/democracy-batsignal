#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

plan_name="district"
n_steps=1000
seed=42
target_pop=8
tol=0.01
pop_col="TOTPOP"
json_dir="JSON_dualgraphs"
output_dir="chain_outputs"

json_file="$(realpath ./${json_dir}/gerrymandria.json)"
final_output_file="$(realpath ./${output_dir}/gerrymandria_chain_${n_steps}_steps.jsonl.ben)"

frcw \
    --assignment-col $plan_name \
    --graph-json $json_file \
    --n-steps $n_steps \
    --pop-col $pop_col \
    --rng-seed $seed \
    --tol $tol \
    --variant district-pairs-rmst \
    --writer ben \
    --batch-size 1 \
    --n-threads 1 \
    --output-file "${final_output_file}"
