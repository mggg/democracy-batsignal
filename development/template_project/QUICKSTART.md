# ReCom quickstart

This guide gets an existing dual graph into a recorded ReCom batch. The included Pennsylvania
settings are runnable as written and can be replaced with another graph and its node columns.

## 1. Check the project

Open a terminal in this project and run:

```bash
uv run python --version
uv run python -c "import gerrychain, gerrytools, binary_ensemble; print('ready')"
```

The same commands work in PowerShell. If the second command prints `ready`, the Python environment
is usable. `rustrecom --version` reports whether the separate RustReCom executable is available.

Windows PowerShell sessions that run the included `.ps1` references may require:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

The process scope lasts only for the current PowerShell session.

## 2. Edit one settings block

Open `pipeline_scripts/run_chains.py`. The block headed `Experiment settings` configures the
standard batch workflow. `settings_for_run()` translates those settings into one chain
configuration, and `main()` submits the batch.

These settings describe the included Pennsylvania example:

```python
ENGINE = "gerrychain"
GRAPH_PATH = PROJECT_ROOT / "JSON_dualgraphs" / "pa_dualgraph.json"
OUTPUT_PREFIX = "VANILLA_PA"
STARTING_PLANS = ("seed_plan",)
POPULATION_COLUMN = "total_pop_20"
RNG_SEEDS = (42,)
TOTAL_STEPS = 100
POPULATION_TOLERANCE = 0.01
RECOM_VARIANT = "district-pairs-mst"
EXPERIMENT_TAG = "quickstart"
RUN_DATE = datetime.now().astimezone().date().isoformat()
MAX_WORKERS = 1
```

For your data, change:

| Setting                | Meaning                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `GRAPH_PATH`           | Path to the dual graph. Keep the path inside quotes and join folders with `/`.         |
| `OUTPUT_PREFIX`        | Short analysis name used in every output filename.                                     |
| `STARTING_PLANS`       | Node columns containing starting assignments. One item is written as `("seed_plan",)`. |
| `POPULATION_COLUMN`    | Node column containing the population used to balance districts.                       |
| `RNG_SEEDS`            | Reproducible seeds combined with every starting plan. `(42,)` has one item.            |
| `TOTAL_STEPS`          | Number of recorded positions per plan-and-seed run. The example uses `100`.            |
| `POPULATION_TOLERANCE` | Allowed fractional deviation from ideal population. `0.01` is 1 percent.               |
| `RECOM_VARIANT`        | District-pair and spanning-tree sampling rule.                                         |
| `EXPERIMENT_TAG`       | Name of the experiment, such as `"baseline"` or `"county-splits"`.                     |
| `RUN_DATE`             | Date included in filenames. The example evaluates the current local date.              |
| `MAX_WORKERS`          | Maximum independent chains running at once.                                            |

The graph must be NetworkX adjacency-data JSON. Every node needs the starting-plan and population
columns named above. Objective files and region-aware proposals may require additional columns.
The full README explains graph preparation and how to inspect available node columns.

Every starting plan is combined with every random seed. For example:

```python
STARTING_PLANS = ("enacted_plan", "alternate_plan")
RNG_SEEDS = (42, 43)
```

This configuration runs four chains. The starting-plan column is included in each output filename.

## 3. Run the batch

From the project root:

```bash
uv run pipeline_scripts/run_chains.py
```

The runner starts one child process per starting-plan and seed combination, limits simultaneous
chains with `MAX_WORKERS`, and writes each child's console output to its own file in `chain_logs/`.
It exits unsuccessfully if any chain fails and tells you which log to inspect.

Successful chains produce BENDL recordings in `chain_outputs/`. A filename records the engine,
step count, random seed, population tolerance, starting-plan column, experiment tag, and date.

## 4. Choose an engine

Change `ENGINE` to switch between ordinary Python and RustReCom runs:

| `ENGINE` value       | Workflow                                                            |
| -------------------- | ------------------------------------------------------------------- |
| `"gerrychain"`       | Python GerryChain example for modifying constraints or proposals.   |
| `"rustrecom-chain"`  | Fast ordinary RustReCom chain. Requires the `rustrecom` executable. |
| `"rustrecom-tilted"` | Objective-guided RustReCom search. Requires an objective file.      |

For an ordinary RustReCom batch, set:

```python
ENGINE = "rustrecom-chain"
OBJECTIVE_FILE = None
```

For a tilted run, also set an objective:

```python
ENGINE = "rustrecom-tilted"
OBJECTIVE_FILE = OBJECTIVES_DIR / "gingles_partial.json"
MAXIMIZE_OBJECTIVE = True
```

Set `MAXIMIZE_OBJECTIVE = False` for objectives where smaller values are better. Tilted chains are
optimization searches, not neutral ensemble samples.

## 5. Scale

A representative 10,000-step chain provides an estimate of runtime and output size for a particular
graph, constraint set, and objective. `STARTING_PLANS` and `RNG_SEEDS` determine the number of runs,
while `TOTAL_STEPS` sets each run's length. `MAX_WORKERS` controls how many chains run concurrently.
The template does not add threads inside an individual chain.

## 6. Collect data

`pipeline_scripts/metrics/collect_data_vanilla_pa.py` evaluates recorded Pennsylvania chains with
GerryTools. Its settings are at the top of the file:

```python
INPUT_GLOB = "*VANILLA_PA*.bendl"
MAX_WORKERS = 1
BATCH_SIZE = 256
```

`INPUT_GLOB` selects BENDL files from `chain_outputs/`. The default matches direct
`VANILLA_PA...` outputs and all Python-runner outputs whose analysis prefix is `VANILLA_PA`,
including the `PY_`, `RUST_CHAIN_`, and `RUST_TILTED_` engines. `MAX_WORKERS` controls how many
recordings are evaluated in separate processes. `BATCH_SIZE` controls how many plans GerryTools
scores together while streaming a recording.

Run the configured data collection scripts from the project root:

```bash
uv run pipeline_scripts/run_data_collection_scripts.py
```

`DATA_COLLECTION_SCRIPTS` at the top of that file lists the collectors to run. The Pennsylvania
collector shown above is included by default.

The supplied evaluator calculates compactness, cut edges, population totals, seats, and election
disproportionality. Each recording produces a reusable GerryTools `EnsembleEvalResult` directory
under `stats/<recording-name>/`.

## 7. Generate figures

Run the configured figure scripts from the project root:

```bash
uv run pipeline_scripts/run_figure_generation_scripts.py
```

`FIGURE_GENERATION_SCRIPTS` at the top of that file lists the scripts and their execution order. It
runs the base-plan maps and the three ensemble figures by default. The ensemble scripts read the
evaluation directories created in the preceding section.

Each ensemble script has a settings block at the top. `STATS_GLOB` selects directories under
`stats/`, `STARTING_PLAN` selects the reference assignment, and the remaining constants control
plot-specific values such as elections, bins, and axis limits. Images are written under
`figures/plan_maps/` or `figures/<recording-name>/`.

`STARTING_PLAN` is one common reference for every directory selected by `STATS_GLOB`. To compare
each starting plan separately, narrow `STATS_GLOB`, change `STARTING_PLAN`, and rerun the script.

## 8. Modifying the code for your project

- `pipeline_scripts/run_chains.py`: Change graph paths, node columns, engines, seeds, and experiment
  settings.
- `pipeline_scripts/run_data_collection_scripts.py`: Select the data collectors to run.
- `pipeline_scripts/run_figure_generation_scripts.py`: Select and order the figure scripts.
- `notebooks/gerrychain_cut_edges_walkthrough.ipynb`: Start here for an interactive GerryChain
  example that runs a chain and plots a result.
- `pipeline_scripts/chain_runners/gerrychain_cli.py`: Change GerryChain proposals, constraints,
  acceptance rules, updaters, or recording behavior.
- `pipeline_scripts/chain_runners/rustrecom_objectives/`: Add or modify RustReCom objective JSON.
- `pipeline_scripts/metrics/`: Change the statistics evaluated for each recorded plan.
- `pipeline_scripts/figure_generators/`: Change maps and ensemble plots.
- `pipeline_scripts/chain_runners/batch_runner.py`: Change output naming, logging, or process
  scheduling.
- `pyproject.toml`: Add Python packages, then run `uv sync`.

Open the notebook from the project root in VS Code, JupyterLab, or another notebook editor and
select this project's `.venv` Python kernel.
