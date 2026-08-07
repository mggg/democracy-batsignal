#!/usr/bin/env bash

set -eu

DEVELOPMENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "$DEVELOPMENT_DIR/.." && pwd -P)
IMAGE_NAME=${BATSIGNAL_TEST_IMAGE:-democracy-batsignal-smoke}

if [[ "${BATSIGNAL_IN_CONTAINER:-}" != "1" ]]; then
    python3 "$DEVELOPMENT_DIR/clean_notebooks.py" --check
    python3 "$DEVELOPMENT_DIR/generate_installers.py" --check
    python3 -m unittest discover -s "$DEVELOPMENT_DIR" -p 'test_*.py'
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
require_file "$BASH_PROJECT/pipeline_scripts/run_chains.py"
require_file "$BASH_PROJECT/pipeline_scripts/run_data_collection_scripts.py"
require_file "$BASH_PROJECT/pipeline_scripts/run_figure_generation_scripts.py"
require_file "$BASH_PROJECT/pipeline_scripts/chain_runners/batch_runner.py"
require_file "$BASH_PROJECT/pipeline_scripts/chain_runners/gerrychain_cli.py"
require_file "$POWERSHELL_PROJECT/pipeline_scripts/run_chains.py"
require_file "$POWERSHELL_PROJECT/pipeline_scripts/chain_runners/batch_runner.py"
require_file "$POWERSHELL_PROJECT/pipeline_scripts/chain_runners/gerrychain_cli.py"
grep -q '^prompt = bash_project$' "$BASH_PROJECT/.venv/pyvenv.cfg"
grep -q '^prompt = powershell_project$' "$POWERSHELL_PROJECT/.venv/pyvenv.cfg"

echo "Checking that both installers emitted the same shared project files..."
uv run --project "$BASH_PROJECT" python - <<'PY'
from pathlib import Path


def shared_files(project: Path) -> dict[Path, Path]:
    return {
        path.relative_to(project): path
        for path in project.rglob("*")
        if path.is_file()
        and ".venv" not in path.relative_to(project).parts
        and path.suffix not in {".ps1", ".sh"}
    }


bash_files = shared_files(Path("/projects/bash_project"))
powershell_files = shared_files(Path("/projects/powershell_project"))
assert bash_files.keys() == powershell_files.keys()
for relative_path, bash_path in bash_files.items():
    assert bash_path.read_bytes() == powershell_files[relative_path].read_bytes(), relative_path
PY

echo "Checking supported Python versions..."
for python_version in 3.12 3.13 3.14; do
    project_name="python_${python_version//./_}_smoke"
    printf '%s\n' "$project_name" n "$python_version" \
        | pwsh -NoProfile -File /installers/democracy-batsignal.ps1
    uv run --project "/projects/$project_name" python -c \
        "import binary_ensemble, geopandas, gerrychain, gerrytools, matplotlib"
done

echo "Checking GerryChain assignment-label normalization..."
uv run --project "$BASH_PROJECT" python - <<'PY'
import sys
from pathlib import Path

import networkx as nx
from click import ClickException

sys.path.insert(0, str(Path("/projects/bash_project")))
from pipeline_scripts.chain_runners.gerrychain_cli import integer_assignment


def assignment(labels):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, {"district": label}) for node, label in enumerate(labels)
    )
    return integer_assignment(graph, "district")


assert assignment([1, 2, 1]) == {0: 1, 1: 2, 2: 1}
assert assignment(["A", "B", "A"]) == {0: 0, 1: 1, 2: 0}
assert assignment(["01", "1"]) == {0: 0, 1: 1}
assert assignment([1.2, 1.8, 1.2]) == {0: 0, 1: 1, 2: 0}
assert assignment([7, "A", 7]) == {0: 0, 1: 1, 2: 0}
assert assignment([-1, 0, -1]) == {0: 0, 1: 1, 2: 0}
assert assignment([65_535, 0]) == {0: 65_535, 1: 0}
assert assignment([65_536, 0, 65_536]) == {0: 0, 1: 1, 2: 0}

for invalid_labels in (
    [None, 1],
    [float("nan"), 1],
    [float("inf"), 1],
    [([],), 1],
):
    try:
        assignment(invalid_labels)
    except ClickException:
        pass
    else:
        raise AssertionError(f"Invalid labels were accepted: {invalid_labels!r}")

try:
    assignment(range(65_537))
except ClickException:
    pass
else:
    raise AssertionError("More than 65,536 distinct labels were accepted.")
PY

echo "Reducing example workloads for the smoke test..."
python_runner="$BASH_PROJECT/pipeline_scripts/run_chains.py"
vanilla="$BASH_PROJECT/pipeline_scripts/chain_runners/pa_example_script_vanilla.sh"
optimized="$BASH_PROJECT/pipeline_scripts/chain_runners/pa_example_script_opt.sh"

grep -q '^ENGINE: ReComEngine = "gerrychain"$' "$python_runner"
grep -q '^STARTING_PLANS = ("seed_plan",)$' "$python_runner"
grep -q '^RNG_SEEDS = (42,)$' "$python_runner"
grep -q '^TOTAL_STEPS = 100$' "$python_runner"
uv run --project "$BASH_PROJECT" python - <<'PY'
import json
from pathlib import Path

graph_path = Path("/projects/bash_project/JSON_dualgraphs/pa_dualgraph.json")
graph = json.loads(graph_path.read_text())
alternate_plan = json.loads(
    Path("/projects/bash_project/data/alt_plan_pa.json").read_text()
)
assert len(graph["nodes"]) == len(alternate_plan)
for node, district in zip(graph["nodes"], alternate_plan, strict=True):
    node["second_plan"] = district + 1
assert any(node["second_plan"] != node["seed_plan"] for node in graph["nodes"])
graph_path.write_text(json.dumps(graph))
PY
sed -i \
    -e 's/^STARTING_PLANS = ("seed_plan",)$/STARTING_PLANS = ("seed_plan", "second_plan")/' \
    -e 's/^TOTAL_STEPS = 100$/TOTAL_STEPS = 2/' \
    -e 's/^EXPERIMENT_TAG = "quickstart"$/EXPERIMENT_TAG = "smoke"/' \
    -e 's/^RUN_DATE = .*$/RUN_DATE = "2026-01-02"/' \
    "$python_runner"

grep -q '^n_steps=1000$' "$vanilla"
grep -q '^rng_seed=(42 43)$' "$vanilla"
sed -i \
    -e 's/^n_steps=1000$/n_steps=2/' \
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
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY VANILLA_PA 42 second_plan).bendl"
require_file "$BASH_PROJECT/chain_logs/$(scheduler_stem PY VANILLA_PA 42 second_plan).log"
uv run --project "$BASH_PROJECT" python - <<'PY'
from pathlib import Path

from binary_ensemble import BendlDecoder

outputs = Path("/projects/bash_project/chain_outputs")
seed_plan = BendlDecoder(next(outputs.glob("PY_VANILLA_PA*SEEDPLN__seed_plan*.bendl")))
second_plan = BendlDecoder(next(outputs.glob("PY_VANILLA_PA*SEEDPLN__second_plan*.bendl")))

assert second_plan.read_metadata()["starting_plan"] == "second_plan"
assert seed_plan.lookup(0) != second_plan.lookup(0)
PY
uv run --project "$BASH_PROJECT" python - <<'PY'
import sys

sys.path.insert(0, "/projects/bash_project/pipeline_scripts")
import run_chains

run_chains.EXPERIMENT_TAG = "bad/name"
try:
    run_chains.settings_for_run(42, "seed_plan")
except ValueError as error:
    assert "Invalid tag" in str(error)
else:
    raise AssertionError("The Python-first settings accepted a path separator in EXPERIMENT_TAG.")
PY
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY PARALLEL_PA 44 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem PY PARALLEL_PA 45 seed_plan).bendl"
BASH_RUN=VANILLA_PA__STEPS_2__RNGSEED_46__TOL_0p01
require_file "$BASH_PROJECT/chain_outputs/${BASH_RUN}.bendl"
require_file \
    "$BASH_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_47__TOL_0p01.bendl"
require_file \
    "$BASH_PROJECT/chain_outputs/GINGLES_PARTIAL_PA__STEPS_2__RNGSEED_47__TOL_0p01_scores.csv"

echo "Checking that a failed child makes the batch runner fail..."
uv run --project "$BASH_PROJECT" python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "/projects/bash_project/pipeline_scripts")
from chain_runners.batch_runner import ChainSettings, run_chains

project_root = Path("/projects/bash_project")
settings = ChainSettings(
    engine="gerrychain",
    graph_path=project_root / "JSON_dualgraphs" / "gerrymandria.json",
    output_prefix="expected_failure",
    starting_plan="missing_attribute",
    pop_col="TOTPOP",
    rng_seed=99,
    total_steps=2,
    population_tolerance=0.01,
    recom_variant="district-pairs-mst",
    run_date="2026-01-02",
    output_dir=project_root / "chain_outputs",
    log_dir=project_root / "chain_logs",
    tag="smoke",
)
try:
    run_chains([settings])
except RuntimeError as error:
    assert "1 chain(s) failed" in str(error)
else:
    raise AssertionError("The batch runner accepted a failed child process.")
PY
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

echo "Checking every supplied RustReCom objective..."
objective_dir="$BASH_PROJECT/pipeline_scripts/chain_runners/rustrecom_objectives"
for objective_file in "$objective_dir"/*.json; do
    objective_name=$(basename "$objective_file" .json)
    maximize=true
    if [[ "$objective_name" == "minimize_bvap_target_deviation" ]]; then
        maximize=false
    fi
    objective_bendl="$BASH_PROJECT/chain_outputs/objective_smoke_${objective_name}.bendl"
    objective_scores="$BASH_PROJECT/chain_outputs/objective_smoke_${objective_name}_scores.csv"
    rustrecom tilted \
        --graph-json "$BASH_PROJECT/JSON_dualgraphs/pa_dualgraph.json" \
        --assignment-col seed_plan \
        --pop-col total_pop_20 \
        --n-steps 2 \
        --tol 0.01 \
        --rng-seed 50 \
        --objective "$objective_file" \
        --maximize "$maximize" \
        --variant district-pairs-mst \
        --writer bendl \
        --output-file "$objective_bendl" \
        --scores-output-file "$objective_scores" \
        --overwrite-output
    require_file "$objective_bendl"
    require_file "$objective_scores"
done

echo "Running the Python-first workflow from the PowerShell-generated project..."
powershell_python_runner="$POWERSHELL_PROJECT/pipeline_scripts/run_chains.py"
sed -i \
    -e 's/^RNG_SEEDS = (42,)$/RNG_SEEDS = (52,)/' \
    -e 's/^TOTAL_STEPS = 100$/TOTAL_STEPS = 2/' \
    -e 's/^EXPERIMENT_TAG = "quickstart"$/EXPERIMENT_TAG = "smoke"/' \
    -e 's/^RUN_DATE = .*$/RUN_DATE = "2026-01-02"/' \
    "$powershell_python_runner"
uv run --project "$POWERSHELL_PROJECT" "$powershell_python_runner"
sed -i \
    -e 's/^RUN_DATE = "2026-01-02"$/RUN_DATE = datetime.now().astimezone().date().isoformat()/' \
    "$powershell_python_runner"

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
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem RUST_CHAIN VANILLA_PA 58 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_outputs/$(scheduler_stem RUST_CHAIN VANILLA_PA 59 seed_plan).bendl"
require_file "$BASH_PROJECT/chain_logs/$(scheduler_stem RUST_CHAIN VANILLA_PA 58 seed_plan).log"

echo "Running tilted RustReCom chains through the Python-first workflow..."
sed -i \
    -e 's/^ENGINE: ReComEngine = "rustrecom-chain"$/ENGINE: ReComEngine = "rustrecom-tilted"/' \
    -e 's#^OBJECTIVE_FILE:.*#OBJECTIVE_FILE: Path | None = OBJECTIVES_DIR / "gingles_partial.json"#' \
    "$python_runner"
uv run --project "$BASH_PROJECT" "$python_runner"
RUST_TILTED_58=$(scheduler_stem RUST_TILTED VANILLA_PA 58 seed_plan)
RUST_TILTED_59=$(scheduler_stem RUST_TILTED VANILLA_PA 59 seed_plan)
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_58}.bendl"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_59}.bendl"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_58}_scores.csv"
require_file "$BASH_PROJECT/chain_outputs/${RUST_TILTED_59}_scores.csv"
require_file "$BASH_PROJECT/chain_logs/${RUST_TILTED_58}.log"

sed -i \
    -e 's/^RUN_DATE = "2026-01-02"$/RUN_DATE = datetime.now().astimezone().date().isoformat()/' \
    "$python_runner"
echo "Running the shared Python metrics and figure pipeline..."
cd "$BASH_PROJECT"
# Keep the walkthrough's deliberate cell layout; lint and type-check it without reformatting it.
uv run ruff check pipeline_scripts notebooks
uv run ruff format --check pipeline_scripts
uv run ty check pipeline_scripts notebooks

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

assert namespace["BENDL_PATH"].is_file()
assert namespace["STATS_DIR"].is_dir()
PY

collector=pipeline_scripts/metrics/collect_data_vanilla_pa.py
grep -q '^INPUT_GLOB = "\*VANILLA_PA\*.bendl"$' "$collector"
sed -i \
    -e 's/^MAX_WORKERS = 1$/MAX_WORKERS = 2/' \
    -e 's/^BATCH_SIZE = 256$/BATCH_SIZE = 2/' \
    "$collector"
uv run pipeline_scripts/run_data_collection_scripts.py
uv run pipeline_scripts/run_figure_generation_scripts.py

require_file "$BASH_PROJECT/figures/plan_maps/pa_original_plan.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_original_plan_philadelphia.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_pres_20_partisan_choropleth.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_pres_20_partisan_choropleth_philadelphia.png"
require_file "$BASH_PROJECT/figures/plan_maps/pa_recom_district_comparison.png"
PY_RUN=$(scheduler_stem PY VANILLA_PA 42 seed_plan)
RUST_RUN=$(scheduler_stem RUST_CHAIN VANILLA_PA 58 seed_plan)
for run in "$BASH_RUN" "$PY_RUN" "$RUST_RUN"; do
    require_file "$BASH_PROJECT/stats/$run/manifest__${run}.json"
    require_file "$BASH_PROJECT/figures/$run/cut_edges_histogram_${run}.png"
    require_file "$BASH_PROJECT/figures/$run/disprop_scatter_${run}.png"
    require_file "$BASH_PROJECT/figures/$run/reock_boxplots_${run}.png"
done

echo "All generated-project smoke tests passed."
