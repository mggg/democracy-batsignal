#!/usr/bin/env bash

set -eu

DEVELOPMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "$DEVELOPMENT_DIR/.." && pwd -P)
IMAGE_NAME=${BATSIGNAL_TEST_IMAGE:-democracy-batsignal-smoke}

if [[ "${BATSIGNAL_IN_CONTAINER:-}" != "1" ]]; then
    python3 "$DEVELOPMENT_DIR/clean_notebooks.py" --check
    docker build --tag "$IMAGE_NAME" --file "$DEVELOPMENT_DIR/Dockerfile" "$ROOT_DIR"
    docker run --rm "$IMAGE_NAME"
    exit
fi

BASH_PROJECT=/projects/bash_project
POWERSHELL_PROJECT=/projects/powershell_project
SMOKE_TAG=smoke
SMOKE_DATE=2026-01-02

function require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Expected a non-empty file at: $1" >&2
        exit 1
    fi
}

function scheduler_stem() {
    local engine=$1 prefix=$2 seed=$3 plan=$4
    printf '%s%s\n' \
        "${engine}_${prefix}__STEPS_2__RNGSEED_${seed}__TOL_0p01" \
        "__SEEDPLN__${plan}__TAG_${SMOKE_TAG}__DATE_${SMOKE_DATE}"
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
require_file "$BASH_PROJECT/pipeline_scripts/chain_runners/pa_example_script_vanilla.sh"
require_file "$POWERSHELL_PROJECT/pipeline_scripts/chain_runners/pa_example_script_vanilla.ps1"

echo "Reducing example workloads for the smoke test..."
python_runner="$BASH_PROJECT/run_chains.py"
vanilla="$BASH_PROJECT/pipeline_scripts/chain_runners/pa_example_script_vanilla.sh"
optimized="$BASH_PROJECT/pipeline_scripts/chain_runners/pa_example_script_opt.sh"

grep -q '^ENGINE: ReComEngine = "gerrychain"$' "$python_runner"
grep -q '^RNG_SEEDS = (42,)$' "$python_runner"
grep -q '^TOTAL_STEPS = 100$' "$python_runner"
sed -i \
    -e 's/^TOTAL_STEPS = 100$/TOTAL_STEPS = 2/' \
    -e 's/^EXPERIMENT_TAG = "quickstart"$/EXPERIMENT_TAG = "smoke"/' \
    -e 's/^RUN_DATE = .*$/RUN_DATE = "2026-01-02"/' \
    "$python_runner"

grep -q '^n_steps=100000$' "$vanilla"
grep -q '^rng_seed=(42 43)$' "$vanilla"
sed -i \
    -e 's/^n_steps=100000$/n_steps=2/' \
    -e 's/^rng_seed=(42 43)$/rng_seed=(46 48)/' \
    "$vanilla"

grep -q '^n_steps=1000$' "$optimized"
grep -q '^rng_seed=(42 43)$' "$optimized"
sed -i \
    -e 's/^n_steps=1000$/n_steps=2/' \
    -e 's/^rng_seed=(42 43)$/rng_seed=(47)/' \
    "$optimized"

echo "Running the Python-first GerryChain workflow..."
uv run --project "$BASH_PROJECT" "$python_runner"

sed -i \
    -e 's/^OUTPUT_PREFIX = "VANILLA_PA"$/OUTPUT_PREFIX = "PARALLEL_PA"/' \
    -e 's/^RNG_SEEDS = (42,)$/RNG_SEEDS = (44, 45)/' \
    -e 's/^MAX_WORKERS = 1$/MAX_WORKERS = 2/' \
    "$python_runner"
uv run --project "$BASH_PROJECT" "$python_runner"

echo "Running the direct RustReCom CLI references..."
bash "$vanilla"
bash "$optimized"

require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY VANILLA_PA 42 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_logs/$(scheduler_stem PY VANILLA_PA 42 seed_plan).log"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY PARALLEL_PA 44 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY PARALLEL_PA 45 seed_plan).bendl"
BASH_RUN=VANILLA_PA__STEPS_2__RNGSEED_46__TOL_0p01
require_file "$BASH_PROJECT/chain_outputs/${BASH_RUN}.bendl"
require_file \
    "$BASH_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_47__TOL_0p01.bendl"
require_file \
    "$BASH_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_47__TOL_0p01_scores.csv"

echo "Checking that a failed child makes the shared runner fail..."
if uv run --project "$BASH_PROJECT" \
    "$BASH_PROJECT/pipeline_scripts/run_parallel_chains.py" \
    --graph-path "$BASH_PROJECT/JSON_dualgraphs/gerrymandria.json" \
    --output-prefix expected_failure \
    --starting-plan missing_attribute \
    --pop-col TOTPOP \
    --rng-seed 99 \
    --total-steps 2 \
    --tag "$SMOKE_TAG" \
    --run-date "$SMOKE_DATE"; then
    echo "The shared runner unexpectedly accepted a failed child." >&2
    exit 1
fi
require_file \
    "$BASH_PROJECT/chain_logs/$(scheduler_stem PY expected_failure 99 missing_attribute).log"

echo "Running the documented RustReCom short-bursts workflow..."
rustrecom short-bursts \
    --graph-json "$BASH_PROJECT/JSON_dualgraphs/pa_dualgraph.json" \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 2 \
    --burst-length 1 \
    --tol 0.01 \
    --rng-seed 49 \
    --objective \
        "$BASH_PROJECT/pipeline_scripts/chain_runners/rustrecom_objectives/gingles_partial.json" \
    --maximize true \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file "$BASH_PROJECT/chain_outputs/gingles_short_bursts.bendl" \
    --scores-output-file "$BASH_PROJECT/chain_outputs/gingles_short_bursts_scores.csv" \
    --overwrite-output
require_file "$BASH_PROJECT/chain_outputs/gingles_short_bursts.bendl"
require_file "$BASH_PROJECT/chain_outputs/gingles_short_bursts_scores.csv"

echo "Running the Python-first workflow from the PowerShell-generated project..."
powershell_python_runner="$POWERSHELL_PROJECT/run_chains.py"
sed -i \
    -e 's/^RNG_SEEDS = (42,)$/RNG_SEEDS = (52,)/' \
    -e 's/^TOTAL_STEPS = 100$/TOTAL_STEPS = 2/' \
    -e 's/^EXPERIMENT_TAG = "quickstart"$/EXPERIMENT_TAG = "smoke"/' \
    -e 's/^RUN_DATE = .*$/RUN_DATE = "2026-01-02"/' \
    "$powershell_python_runner"
uv run --project "$POWERSHELL_PROJECT" "$powershell_python_runner"

echo "Running the PowerShell RustReCom CLI references..."
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/pipeline_scripts/chain_runners/pa_example_script_vanilla.ps1' \
        -NSteps 2 -RngSeeds @(56)"
pwsh -NoProfile -Command \
    "& '$POWERSHELL_PROJECT/pipeline_scripts/chain_runners/pa_example_script_opt.ps1' \
        -NSteps 2 -RngSeeds @(57)"

require_file "$POWERSHELL_PROJECT/chain_outputs/$(scheduler_stem PY VANILLA_PA 52 seed_plan).bendl"
require_file "$POWERSHELL_PROJECT/chain_logs/$(scheduler_stem PY VANILLA_PA 52 seed_plan).log"
require_file \
    "$POWERSHELL_PROJECT/chain_outputs/VANILLA_PA__STEPS_2__RNGSEED_56__TOL_0p01.bendl"
require_file \
    "$POWERSHELL_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_57__TOL_0p01.bendl"
require_file \
    "$POWERSHELL_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_57__TOL_0p01_scores.csv"

echo "Running ordinary RustReCom chains through the Python-first workflow..."
sed -i \
    -e 's/^ENGINE: ReComEngine = "gerrychain"$/ENGINE: ReComEngine = "rustrecom-chain"/' \
    -e 's/^OUTPUT_PREFIX = "PARALLEL_PA"$/OUTPUT_PREFIX = "VANILLA_PA"/' \
    -e 's/^RNG_SEEDS = (44, 45)$/RNG_SEEDS = (58, 59)/' \
    -e 's#^REGION_WEIGHTS:.*#REGION_WEIGHTS: dict[str, float] | None = {"boundary_node": 1.0}#' \
    "$python_runner"
uv run --project "$BASH_PROJECT" "$python_runner"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem RUST VANILLA_PA 58 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem RUST VANILLA_PA 59 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_logs/$(scheduler_stem RUST VANILLA_PA 58 seed_plan).log"

echo "Running tilted RustReCom chains through the Python-first workflow..."
sed -i \
    -e 's/^ENGINE: ReComEngine = "rustrecom-chain"$/ENGINE: ReComEngine = "rustrecom-tilted"/' \
    -e 's/^OUTPUT_PREFIX = "VANILLA_PA"$/OUTPUT_PREFIX = "GINGLES_PARTIAL_PA"/' \
    -e 's/^RNG_SEEDS = (58, 59)$/RNG_SEEDS = (60, 61)/' \
    -e 's#^OBJECTIVE_FILE:.*#OBJECTIVE_FILE: Path | None = OBJECTIVES_DIR / "gingles_partial.json"#' \
    "$python_runner"
uv run --project "$BASH_PROJECT" "$python_runner"
RUST_TILTED_60=$(scheduler_stem RUST GINGLES_PARTIAL_PA 60 seed_plan)
RUST_TILTED_61=$(scheduler_stem RUST GINGLES_PARTIAL_PA 61 seed_plan)
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_60}.bendl"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_61}.bendl"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_60}_scores.csv"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_61}_scores.csv"
require_file "$BASH_PROJECT/chain_logs/${RUST_TILTED_60}.log"

echo "Running the shared Python metrics and figure pipeline..."
cd "$BASH_PROJECT"
uv run ruff check pipeline_scripts
uv run ruff format --check pipeline_scripts
uv run ty check pipeline_scripts

echo "Running the GerryChain walkthrough notebook cells..."
uv run env MPLBACKEND=Agg python - <<'PY'
import json
import os
from pathlib import Path

project_root = Path("/projects/bash_project")
notebook_path = project_root / "notebooks/gerrychain_cut_edges_walkthrough.ipynb"
notebook = json.loads(notebook_path.read_text())
namespace = {"__name__": "__main__"}

os.chdir(notebook_path.parent)
for cell_number, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path.name}:cell-{cell_number}", "exec"), namespace)

assert namespace["OUTPUT_PATH"].is_file()
assert namespace["STATS_DIR"].is_dir()
PY

uv run pipeline_scripts/metrics/collect_data_vanilla_pa.py --max-workers 2 --batch-size 2
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
