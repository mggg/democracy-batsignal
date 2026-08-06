#!/usr/bin/env bash

set -euo pipefail

DEVELOPMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "$DEVELOPMENT_DIR/.." && pwd -P)
IMAGE_NAME=${BATSIGNAL_TEST_IMAGE:-democracy-batsignal-smoke}

if [[ "${BATSIGNAL_IN_CONTAINER:-}" != "1" ]]; then
    docker build --tag "$IMAGE_NAME" --file "$DEVELOPMENT_DIR/Dockerfile" "$ROOT_DIR"
    docker run --rm "$IMAGE_NAME"
    exit
fi

BASH_PROJECT=/projects/bash_project
POWERSHELL_PROJECT=/projects/powershell_project

function require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Expected a non-empty file at: $1" >&2
        exit 1
    fi
}

function reject_source_suffix() {
    local project=$1 suffix=$2
    if find "$project" -path "$project/.venv" -prune \
        -o -type f -name "*$suffix" -print -quit | grep -q .; then
        echo "Project $project unexpectedly contains a $suffix source file." >&2
        exit 1
    fi
}

echo "Checking generated project layouts..."
reject_source_suffix "$BASH_PROJECT" ".ps1"
reject_source_suffix "$POWERSHELL_PROJECT" ".sh"
require_file "$BASH_PROJECT/pipeline_scripts/pa_example_script_vanilla.sh"
require_file "$POWERSHELL_PROJECT/pipeline_scripts/pa_example_script_vanilla.ps1"

echo "Reducing Bash example workloads for the smoke test..."
simple="$BASH_PROJECT/batch_example_python_cli_simple.sh"
parallel="$BASH_PROJECT/batch_example_python_cli_parallel.sh"
vanilla="$BASH_PROJECT/pipeline_scripts/pa_example_script_vanilla.sh"
optimized="$BASH_PROJECT/pipeline_scripts/pa_example_script_opt.sh"

grep -q '^rng_seeds=(42 43 44)$' "$simple"
grep -q '^n_steps=1000$' "$simple"
grep -q '^n_steps=100000$' "$simple"
sed -i \
    -e 's/^rng_seeds=(42 43 44)$/rng_seeds=(42)/' \
    -e 's/^n_steps=1000$/n_steps=2/' \
    -e 's/^n_steps=100000$/n_steps=2/' \
    "$simple"

grep -q '^rng_seeds=({1..50})$' "$parallel"
grep -q '^n_steps=1000$' "$parallel"
sed -i \
    -e 's/^rng_seeds=({1..50})$/rng_seeds=(44 45)/' \
    -e 's/^n_steps=1000$/n_steps=2/' \
    "$parallel"

grep -q '^n_steps=100000$' "$vanilla"
grep -q '^rng_seed=(42 43)$' "$vanilla"
sed -i \
    -e 's/^n_steps=100000$/n_steps=2/' \
    -e 's/^rng_seed=(42 43)$/rng_seed=(46)/' \
    "$vanilla"

grep -q '^n_steps=1000$' "$optimized"
grep -q '^rng_seed=(42 43)$' "$optimized"
sed -i \
    -e 's/^n_steps=1000$/n_steps=2/' \
    -e 's/^rng_seed=(42 43)$/rng_seed=(47)/' \
    "$optimized"

echo "Running every Bash helper..."
bash "$simple"
bash "$parallel"
bash "$vanilla"
bash "$optimized"

require_file "$BASH_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed42.bendl"
require_file "$BASH_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed44.bendl"
require_file "$BASH_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed45.bendl"
require_file "$BASH_PROJECT/chain_outputs/PA_chain_2_steps_seed42.bendl"
BASH_RUN=VANILLA_PA__STEPS_2__RNGSEED_46__TOL_0p01
require_file "$BASH_PROJECT/chain_outputs/${BASH_RUN}.bendl"
require_file \
    "$BASH_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_47__TOL_0.01.bendl"

echo "Running every PowerShell helper..."
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/batch_example_python_cli_simple.ps1' \
        -RngSeeds @(52) -TotalSteps 2 -RngSeeds2 @(53) -TotalSteps2 2"
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/batch_example_python_cli_parallel.ps1' \
        -MaxJobs 2 -RngSeeds @(54, 55) -TotalSteps 2"
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/pipeline_scripts/pa_example_script_vanilla.ps1' \
        -NSteps 2 -RngSeeds @(56)"
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/pipeline_scripts/pa_example_script_opt.ps1' \
        -NSteps 2 -RngSeeds @(57)"

require_file "$POWERSHELL_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed52.bendl"
require_file "$POWERSHELL_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed54.bendl"
require_file "$POWERSHELL_PROJECT/chain_outputs/gerrymandria_chain_2_steps_seed55.bendl"
require_file "$POWERSHELL_PROJECT/chain_outputs/PA_chain_2_steps_seed53.bendl"
require_file \
    "$POWERSHELL_PROJECT/chain_outputs/VANILLA_PA__STEPS_2__RNGSEED_56__TOL_0p01.bendl"
require_file \
    "$POWERSHELL_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_57__TOL_0.01.bendl"

echo "Running the shared Python metrics and figure pipeline..."
cd "$BASH_PROJECT"
uv run pipeline_scripts/metrics/collect_data_vanilla_pa.py
uv run pipeline_scripts/figure_generators/base_plan_figures.py
uv run pipeline_scripts/figure_generators/cut_edges_histogram.py
uv run pipeline_scripts/figure_generators/disprop_scatter.py
uv run pipeline_scripts/figure_generators/reock_boxplot.py

require_file "$BASH_PROJECT/stats/$BASH_RUN/manifest__${BASH_RUN}.json"
require_file "$BASH_PROJECT/figures/plan_maps/pa_original_plan.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_original_plan_philadelphia.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_pres_20_partisan_choropleth.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_pres_20_partisan_choropleth_philadelphia.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_recom_district_comparison.png"
require_file "$BASH_PROJECT/figures/$BASH_RUN/cut_edges_histogram_${BASH_RUN}.png"
require_file "$BASH_PROJECT/figures/$BASH_RUN/disprop_scatter_${BASH_RUN}.png"
require_file "$BASH_PROJECT/figures/$BASH_RUN/reock_boxplots_${BASH_RUN}.png"

echo "All generated-project smoke tests passed."
