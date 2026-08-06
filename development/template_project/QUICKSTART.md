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

## 2. Edit one settings block

Open `run_chains.py`. The block headed `Experiment settings` configures the standard batch workflow.
`settings_for_seed()` translates those settings into one chain configuration, and `main()` submits
the batch.

These settings describe the included Pennsylvania example:

```python
ENGINE = "gerrychain"
GRAPH_PATH = PROJECT_ROOT / "JSON_dualgraphs" / "pa_dualgraph.json"
OUTPUT_PREFIX = "VANILLA_PA"
STARTING_PLAN = "seed_plan"
POPULATION_COLUMN = "total_pop_20"
RNG_SEEDS = (42,)
TOTAL_STEPS = 100
POPULATION_TOLERANCE = 0.01
RECOM_VARIANT = "district-pairs-mst"
EXPERIMENT_TAG = "quickstart"
RUN_DATE = date.today().isoformat()
MAX_WORKERS = 1
```

For your data, change:

| Setting                | Meaning                                                                           |
| ---------------------- | --------------------------------------------------------------------------------- |
| `GRAPH_PATH`           | Path to the dual graph. Keep the path inside quotes and join folders with `/`.    |
| `OUTPUT_PREFIX`        | Short analysis name used in every output filename.                                |
| `STARTING_PLAN`        | Node column containing the starting district assignment.                          |
| `POPULATION_COLUMN`    | Node column containing the population used to balance districts.                  |
| `RNG_SEEDS`            | Reproducible seed for each independent chain. `(42,)` is a one-item Python tuple. |
| `TOTAL_STEPS`          | Number of recorded chain positions per seed. The example uses `100`.              |
| `POPULATION_TOLERANCE` | Allowed fractional deviation from ideal population. `0.01` is 1 percent.          |
| `RECOM_VARIANT`        | District-pair and spanning-tree sampling rule.                                    |
| `EXPERIMENT_TAG`       | Name of the experiment, such as `"baseline"` or `"county-splits"`.                |
| `RUN_DATE`             | Date included in filenames. The example evaluates the current local date.         |
| `MAX_WORKERS`          | Maximum independent chains running at once.                                       |

The graph must be NetworkX adjacency-data JSON. Every node needs the starting-plan and population
columns named above. Objective files and region-aware proposals may require additional columns.
The full README explains graph preparation and how to inspect available node columns.

## 3. Run the batch

From the project root:

```bash
uv run run_chains.py
```

The runner starts one child process per seed, limits simultaneous chains with `MAX_WORKERS`, and
writes each child's console output to its own file in `chain_logs/`. It exits unsuccessfully if any
chain fails and tells you which log to inspect.

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
graph, constraint set, and objective. `RNG_SEEDS` and `TOTAL_STEPS` set the batch size;
`MAX_WORKERS` controls how many independent chains run concurrently. The template does not add
threads inside an individual chain.

Next, use `pipeline_scripts/metrics/collect_data_vanilla_pa.py` as the model for scoring BENDL
recordings and `pipeline_scripts/figure_generators/` as examples for plots.

## 6. Modifying the Code for Your Project

- `run_chains.py`: Change graph paths, node columns, engines, seeds, and experiment settings.
- `notebooks/gerrychain_cut_edges_walkthrough.ipynb`: Start here for an interactive GerryChain
  example that runs a chain and plots a result.
- `pipeline_scripts/chain_runners/example_cli.py`: Change GerryChain proposals, constraints,
  acceptance rules, updaters, or recording behavior.
- `pipeline_scripts/chain_runners/rustrecom_objectives/`: Add or modify RustReCom objective JSON.
- `pipeline_scripts/metrics/`: Change the statistics evaluated for each recorded plan.
- `pipeline_scripts/figure_generators/`: Change maps and ensemble plots.
- `pipeline_scripts/run_parallel_chains.py`: Change output naming, logging, or process scheduling.
- `pyproject.toml`: Add Python packages, then run `uv sync`.

Open the notebook from the project root in VS Code, JupyterLab, or another notebook editor and
select this project's `.venv` Python kernel.
