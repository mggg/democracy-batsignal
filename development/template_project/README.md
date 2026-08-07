# Redistricting Project

This is a runnable example of a redistricting ensemble workflow. It includes two ways to generate
ReCom chains, BENDL recording, GerryTools scoring, and figures built from Pennsylvania data.

- Use **RustReCom** for fast ordinary chains and objective-guided searches.
- Use **GerryChain** when you want to build or modify the proposal, constraints, acceptance rule,
  or updaters in Python.
- Use **binary-ensemble** to inspect BENDL recordings.
- Use **GerryTools** to score recordings and create plots.

All the example files serve as starting points and are intended to be modified later.

## Reference documentation

- [GerryChain documentation](https://gerrychain.readthedocs.io/en/latest/) covers graphs,
  partitions, proposals, constraints, updaters, and the Python ReCom workflow.
- [GerryTools documentation](https://gerrytools.readthedocs.io/en/latest/) covers its data,
  geometry, scoring, and plotting APIs.
- [binary-ensemble documentation](https://binary-ensemble.readthedocs.io/en/latest/) covers BEN,
  XBEN, BENDL, streaming, random access, and the Python API.
- [RustReCom source and README](https://github.com/mggg/rustrecom) supplement the command-specific
  help from `rustrecom <command> --help`.

## Start here

Use [QUICKSTART.md](QUICKSTART.md) for the shortest path from a dual graph to a batch of recorded
ReCom chains. The same Python entry point runs on macOS, Linux, and Windows:

```bash
uv run pipeline_scripts/run_chains.py
```

The included settings run a 100-step Pennsylvania GerryChain example. To use another graph, edit
the `Experiment settings` block near the top of `pipeline_scripts/run_chains.py`. That selects the
engine, graph, node columns, seeds, step count, population tolerance, experiment tag, and
concurrency.

The Democracy Batsignal installer creates and synchronizes the environment. This command reports
whether the Python packages are importable:

```bash
uv run python -c "import gerrychain, gerrytools, binary_ensemble; print('Python tools ready')"
```

If you chose a RustReCom engine, also verify the standalone executable:

```bash
rustrecom --version
```

RustReCom is not installed in the uv environment. It is a Rust executable installed through
Cargo. The GerryChain examples still work if you skipped it.

Windows PowerShell sessions that run the included `.ps1` references may require:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

The process scope lasts only for the current PowerShell session.

If you change `pyproject.toml`, update the environment with:

```bash
uv sync
```

## Where to customize an experiment

Each part of an experiment has one primary customization point:

- `pipeline_scripts/run_chains.py` contains the normal experiment settings: graph, columns, engine,
  seeds, ReCom variant, objective, and concurrency.
- `pipeline_scripts/run_data_collection_scripts.py` selects the data collectors to run.
- `pipeline_scripts/run_figure_generation_scripts.py` selects and orders the figure scripts.
- `notebooks/gerrychain_cut_edges_walkthrough.ipynb` is an interactive GerryChain example from
  graph loading through a 10,000-step cut-edge histogram.
- `pipeline_scripts/chain_runners/gerrychain_cli.py` contains the GerryChain proposal, partition,
  updaters, and BENDL recording. Modify it for new constraints or acceptance rules.
- `pipeline_scripts/chain_runners/rustrecom_objectives/` contains RustReCom optimization
  objectives. Copy the closest JSON example and change its columns or score settings.
- `pipeline_scripts/metrics/` contains ensemble scoring. Add statistics here when the output will
  be reused by multiple figures.
- `pipeline_scripts/figure_generators/` contains maps and plots.
- `pipeline_scripts/chain_runners/batch_runner.py` owns process scheduling, log capture, and
  standardized filenames.
- `pyproject.toml` owns Python dependencies. Run `uv sync` after editing it.

## Project layout

```text
.
├── QUICKSTART.md              # ~10-minute guide for adapting the project
├── JSON_dualgraphs/
│   ├── gerrymandria.json       # small graph for the Python examples
│   └── pa_dualgraph.json       # Pennsylvania graph used by RustReCom and scoring examples
├── data/
│   ├── pa_gdf.parquet          # Pennsylvania geometry used by scoring and figure examples
│   └── alt_plan_pa.json        # one alternate assignment used by a comparison map
├── notebooks/
│   └── gerrychain_cut_edges_walkthrough.ipynb
├── pipeline_scripts/
│   ├── run_chains.py            # edit one settings block, then run this file
│   ├── run_data_collection_scripts.py
│   ├── run_figure_generation_scripts.py
│   ├── chain_runners/           # one-chain CLI and the bounded batch runner
│   ├── metrics/                # GerryTools evaluation
│   └── figure_generators/      # maps and ensemble plots
├── chain_outputs/              # BENDL recordings
├── chain_logs/                 # redirected batch output (where to find error messages)
├── stats/                      # EnsembleEvalResult data (parquet files)
└── figures/                    # generated PNG files
```

Projects created by the Bash installer contain the direct `.sh` RustReCom examples. Projects
created by the PowerShell installer contain equivalent `.ps1` examples. The primary Python
workflow is identical on every platform.

## Choose a chain workflow

The project provides three chain-generation modes through `pipeline_scripts/run_chains.py`:

1. `ENGINE = "gerrychain"` records the Python GerryChain example. Use this route when you want to
   modify the proposal, constraints, acceptance rule, or updaters.
2. `ENGINE = "rustrecom-chain"` launches high-throughput ordinary RustReCom chains while keeping
   experiment configuration and parallelization in Python.
3. `ENGINE = "rustrecom-tilted"` launches objective-guided RustReCom searches from the same Python
   settings file.

RustReCom also provides two objective-guided search commands:

- `rustrecom tilted` runs one chain and probabilistically favors score improvements.
- `rustrecom short-bursts` runs fixed-length bursts and starts each new burst from the best plan
  found in the preceding burst.

An objective-guided run is not a neutral ensemble sample. It is a search for plans that score well
under the selected objective.

The Bash project contains `.sh` files in `pipeline_scripts/chain_runners/`; the PowerShell project
contains the equivalent `.ps1` files. They intentionally expose the raw RustReCom CLI.
`pipeline_scripts/run_chains.py` adds Python-based plan and seed lists, concurrency, logs, and
failure handling around the same commands.

## Preparing a dual graph

RustReCom reads a NetworkX adjacency-data JSON graph. Each node represents one geographic unit,
and an edge joins units that are adjacent for redistricting purposes. Before running a chain,
every node must have:

- an assignment attribute naming its district in the starting plan;
- the population attribute passed to `--pop-col`; and
- every node-level attribute referenced by an objective or requested statistic.

Compactness objectives may also require node area, node boundary perimeter, and edge shared
perimeter attributes. Column names are case-sensitive. The starting assignment must define a
contiguous plan, and the graph must support population-balanced recombinations.

The available graph columns can be inspected with:

```bash
uv run python - <<'PY'
from gerrychain import Graph

graph = Graph.from_json("JSON_dualgraphs/pa_dualgraph.json")
node = next(iter(graph.nodes))
edge = next(iter(graph.edges))

print("Example node:", node)
print("Node columns:", sorted(graph.node_data(node)))
print("Example edge:", edge)
print("Edge columns:", sorted(graph.edge_data(edge)))
PY
```

In PowerShell, put the Python portion in a temporary `.py` file and run it with `uv run python`.

## RustReCom ordinary chains

The complete Pennsylvania example is
`pipeline_scripts/chain_runners/pa_example_script_vanilla.sh` in a Bash project or
`pipeline_scripts/chain_runners/pa_example_script_vanilla.ps1` in a PowerShell project. Its central
command is:

```bash
rustrecom chain \
    --graph-json JSON_dualgraphs/pa_dualgraph.json \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 1000 \
    --tol 0.01 \
    --rng-seed 42 \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file chain_outputs/pa_example.bendl \
    --overwrite-output \
    --show-progress
```

`rustrecom` names the executable and `chain` selects ordinary ReCom sampling. Everything after the
subcommand is an option-value pair or a switch. The backslashes only continue a Bash command onto
the next line; the PowerShell reference uses backticks for the same purpose. Option order does not
change the run.

One call produces one independently seeded chain and one output file. The reference scripts loop
over two seeds to demonstrate reproducible independent runs. `pipeline_scripts/run_chains.py`
manages plan-and-seed combinations, concurrency, logs, and failures in Python.

### Core chain options

- `--graph-json` points to a NetworkX adjacency-data JSON dual graph. RustReCom does not accept
  node-link JSON.
- `--assignment-col` names the node attribute containing the starting district labels.
- `--pop-col` names the node population attribute used to balance districts.
- `--n-steps` is the requested chain length. Runtime and output size grow with it.
- `--tol` is the allowed fractional deviation from target district population. A value of `0.01`
  means one percent.
- `--rng-seed` controls the pseudorandom proposal stream. Use a different seed for each independent
  run.
- `--variant` selects how district pairs and spanning trees are sampled.
- `--writer bendl` stores assignments together with the graph and metadata.
- `--output-file` is required for BENDL output.
- `--overwrite-output` replaces an existing output instead of failing.
- `--show-progress` displays progress without changing the recording.
- `--sample-interval K` records the seed and every Kth chain position, reducing output size without
  reducing the work performed by the chain.

Run `rustrecom chain --help` for the complete option list and the installed version's defaults.

### ReCom variants

The four common variants combine two district-pair rules with two spanning-tree rules:

| Variant              | District pair                    | Spanning tree         |
| -------------------- | -------------------------------- | --------------------- |
| `cut-edges-mst`      | Select through a cut edge        | Minimum spanning tree |
| `district-pairs-mst` | Select an adjacent district pair | Minimum spanning tree |
| `cut-edges-ust`      | Select through a cut edge        | Uniform spanning tree |
| `district-pairs-ust` | Select an adjacent district pair | Uniform spanning tree |

The ordinary `chain` command also exposes region-aware and reversible variants, documented by
`rustrecom chain --help`. Their assumptions and tuning differ from the four examples above. A
variant changes the proposal distribution, not merely the runtime.

### Constraints and additional columns

RustReCom can load additional settings from the command line or a versioned JSON config:

- `--constraint` accepts inline JSON or a path to constraint JSON.
- `--sum-cols` asks RustReCom to maintain district totals for extra node attributes.
- `--partial-sum-cols` is available to the optimization commands when an attribute may legitimately
  be absent on some nodes. Missing values are treated as zero.
- `--region-weights` and `--edge-weight-keys` influence supported weighted or region-aware tree
  sampling variants.
- `--config` accepts a JSON string, a JSON file path, or `-` for standard input. Its fields mirror
  the command-line arguments.

A versioned JSON file can record many non-default settings alongside the results needed to
reconstruct a run.

## BENDL recordings

BENDL is the default format in this project because one file carries:

- the compressed assignment stream;
- the dual graph used by the run; and
- metadata such as population column, tolerance, seed, variant, and step count.

Use `binary_ensemble.BendlDecoder` to inspect a recording

```python
from binary_ensemble import BendlDecoder

recording = BendlDecoder("chain_outputs/VANILLA_PA__STEPS_1000__RNGSEED_42__TOL_0p01.bendl")

print(recording.count_samples())
print(recording.read_metadata())

graph = recording.read_graph()
first_assignment = recording.lookup(0)
```

Other useful methods include `list_assets()`, `subsample_every()`, `subsample_range()`, and
`extract_stream()`. An assignment returned by `lookup()` uses the recording's node order.

## RustReCom optimization objectives

An objective converts a districting plan into a numeric score. `rustrecom tilted` uses that score
to favor some valid ReCom proposals over others.

The Pennsylvania-ready objective files are in
`pipeline_scripts/chain_runners/rustrecom_objectives/`. Pass one to `rustrecom tilted` with
`--objective`:

```bash
rustrecom tilted \
    --graph-json JSON_dualgraphs/pa_dualgraph.json \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 1000 \
    --tol 0.01 \
    --rng-seed 42 \
    --objective pipeline_scripts/chain_runners/rustrecom_objectives/gingles_partial.json \
    --maximize true \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file chain_outputs/gingles_example.bendl \
    --scores-output-file chain_outputs/gingles_example_scores.csv \
    --overwrite-output \
    --show-progress
```

`pipeline_scripts/chain_runners/pa_example_script_opt.sh` in a Bash project or the corresponding
`.ps1` file in a PowerShell project contains a complete two-seed example. `--objective` also accepts
inline JSON, but a file is easier to inspect, reuse, and preserve with the results.

### Tilted acceptance

For each valid proposal, RustReCom compares the current and proposed plan scores. Score
improvements are accepted. The selected acceptance rule determines the probability of accepting a
worse score, which allows the search to move away from local optima. The direction and worse-plan
behavior are controlled by these options:

- `--maximize true`: Higher scores are improvements. Use this for all supplied examples except
  target deviation.
- `--maximize false`: Lower scores are improvements. Use this for
  `minimize_bvap_target_deviation.json`.

- Acceptance rules `--accept-rule <RULE>`:
  - `fixed`: Accept any worse plan with the probability from `--accept-worse-prob`.
  - `linear`: Accept a worse plan with probability `max(0, 1 - beta * score_loss)`.
  - `exponential`: Accept a worse plan with probability `exp(-beta * score_loss)`.

- `--acceptance-beta`: Control how strongly `linear` or `exponential` rejects worse scores. Larger
  values make the search greedier. The default is `1.0`.
- `--accept-worse-prob`: Set the probability from `0` to `1` used only by the fixed rule. `0` is
  hill climbing; `1` accepts every valid proposal.
- `--scores-output-file`: Write objective scores and per-district scores to CSV.
- `--write-improved-scores-only`: Write only new global-best rows to the score CSV.

### Short bursts

`rustrecom short-bursts` uses the same graph, population, objective, direction, variant, writer,
and output options as `tilted`. Add `--burst-length` to select the number of accepted steps in each
burst:

```bash
rustrecom short-bursts \
    --graph-json JSON_dualgraphs/pa_dualgraph.json \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 1000 \
    --burst-length 10 \
    --tol 0.01 \
    --rng-seed 42 \
    --objective pipeline_scripts/chain_runners/rustrecom_objectives/gingles_partial.json \
    --maximize true \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file chain_outputs/gingles_short_bursts.bendl \
    --scores-output-file chain_outputs/gingles_short_bursts_scores.csv \
    --overwrite-output \
    --show-progress
```

### Objective JSON settings

Every objective file contains an `objective` field selecting the scoring function. Other fields
set numeric targets or name attributes in the dual graph. Attribute names are case-sensitive.
Population and election columns used by these objectives must contain non-null data on every
node unless the command explicitly treats a column as partial.

The supplied files use columns from `JSON_dualgraphs/pa_dualgraph.json`. When adapting an objective
to another graph, update every column-name setting.

### Gingles partial

File: `gingles_partial.json`. Use `--maximize true`.

This objective rewards minority opportunity districts while retaining a gradient toward the next
one. Each district at or above `threshold` contributes `1`. The highest-share district below the
threshold contributes `minority_share / threshold`. All other below-threshold districts contribute
`0`.

For example, three districts at or above 50% and a best remaining district at 45% produce a score
of `3 + 0.45 / 0.50 = 3.9`.

- `objective`: Must be `"gingles_partial"`.
- `threshold`: Minority share required for a full point. It must be between `0` and `1`.
- `min_pop`: Node column containing the minority population. The example uses `bvap_20`.
- `total_pop`: Node column used as the denominator. The example uses `total_vap_20`.

This is a plan-search metric, not a legal determination that a district satisfies the Gingles
preconditions or other Voting Rights Act requirements.

### Banded Gingles partial

File: `banded_gingles_partial.json`. Use `--maximize true`.

This variant rewards minority shares inside a target band instead of rewarding every share above
one threshold. A district inside the inclusive band contributes `1`. The best district below the
band contributes `share / lower_threshold`. Each district above the band contributes
`upper_threshold / share`, so increasingly large overshoots receive progressively less credit.

- `objective`: Must be `"banded_gingles_partial"`.
- `lower_threshold`: Lower edge of the target band, between `0` and `1`.
- `upper_threshold`: Upper edge of the band. It must be at least `lower_threshold` and less than
  `1`.
- `min_pop`: Node column containing the minority population.
- `total_pop`: Node column used as the denominator.

The supplied example targets BVAP shares from 50% through 60%.

### Election wins

File: `democratic_election_wins.json`. Usually use `--maximize true`.

For each election, this objective counts districts won by the selected side. It adds a fractional
tiebreaker for the closest losing district, giving the optimizer useful score changes between
whole-number seat gains. Scores from multiple elections are combined using `aggregation`.

- `objective`: Must be `"election_wins"`.
- `elections`: List of elections. Each entry names the two node-level vote columns with `votes_a`
  and `votes_b`.
- `target`: Side whose wins are scored, either `"a"` or `"b"`.
- `aggregation`: How to combine election scores:
  - `"mean"` averages them.
  - `"min"` uses the weakest election, favoring plans that perform well across all listed
    elections.
  - `"sum"` adds them.

The supplied example targets Democratic wins (`"a"`) using the 2020 presidential and 2018 U.S.
Senate vote columns. Use `--maximize false` if the goal is to reduce the target side's wins.

### Polsby-Popper compactness

File: `polsby_popper_mean.json`. Use `--maximize true`.

This objective computes `4 * pi * area / perimeter^2` for each district. Values closer to `1` are
more compact. District scores are combined using `aggregation`.

- `objective`: Must be `"polsby_popper"`.
- `area_col`: Node column containing precinct area.
- `shared_perim_col`: Edge column containing the shared perimeter between adjacent precincts.
- `boundary_perim_col`: Node column containing the portion of a precinct boundary on the exterior
  of the state. RustReCom combines it with shared perimeters to derive total precinct perimeter.
- `perim_col`: Alternative node column containing total precinct perimeter. One of `perim_col` or
  `boundary_perim_col` is required.
- `aggregation`: `"mean"`, `"min"`, or `"sum"` across district compactness scores. Maximizing
  `"min"` specifically improves the least-compact district.

The supplied example uses the Pennsylvania graph's `area`, `shared_perim`, and `boundary_perim`
attributes and maximizes mean compactness.

### BVAP target deviation

File: `minimize_bvap_target_deviation.json`. Use `--maximize false`.

This objective calculates a population-of-interest share for every district, then matches the
requested targets to distinct districts. Its score is the smallest possible sum of absolute
differences between matched district shares and targets (note: this is NOT $L^1$ unless a score
for every district is provided). Lower is better.

The supplied `[0.5, 0.5]` target asks for two distinct districts whose BVAP shares are as close as
possible to 50%.

- `objective`: `"by_district_abs_deviation"`, or the shorter alias `"abs_deviation"` when using
  the shorthand settings below.
- `target_values`: List of target shares. Each target is matched to a different district.
- `target` and `n_target_districts`: Shorthand for repeating one target. For example,
  `"target": 0.5` and `"n_target_districts": 2` are equivalent to `[0.5, 0.5]`.
- `pov_counts_col`: Node column containing the population of interest. Despite the setting name,
  it can represent any population; the example uses `bvap_20`.
- `total_counts_col`: Node column used as the denominator for each district. The example uses
  `total_vap_20`.
- `total_count`: Alternative fixed global denominator. Use it instead of `total_counts_col`.

## GerryChain recording workflow

`pipeline_scripts/chain_runners/gerrychain_cli.py` shows the current GerryChain and GerryTools
recording pattern:

1. load a `gerrychain.Graph`;
2. construct a `gerrytools.ben.RecordedChain`;
3. assign its initial `Partition`, including the population updater;
4. select one of GerryChain 1.0's four standard `ReCom` proposal variants; and
5. iterate the chain to write a BENDL recording.

The batch runner starts every chain as a child process. This CLI gives GerryChain a stable
one-chain command, so the batch runner can schedule Python and RustReCom runs without
changing its process, logging, or failure-handling syntax. Normal experiment settings remain in
`pipeline_scripts/run_chains.py`.

Select this implementation through the editable Python settings in
`pipeline_scripts/run_chains.py`:

```python
ENGINE = "gerrychain"
GRAPH_PATH = PROJECT_ROOT / "JSON_dualgraphs" / "pa_dualgraph.json"
STARTING_PLANS = ("seed_plan",)
POPULATION_COLUMN = "total_pop_20"
RNG_SEEDS = (42,)
TOTAL_STEPS = 100
```

Each item in `STARTING_PLANS` is a node attribute name, not a path to an assignment file. Every
starting plan is combined with every value in `RNG_SEEDS`. The GerryChain implementation converts
assignment labels to BENDL-compatible integer district IDs. Integer-like labels retain their values
when conversion is one-to-one and the IDs are between 0 and 65,535; other hashable labels receive
stable IDs based on graph iteration order. Missing and non-finite labels are rejected, normalization
never merges distinct labels, and a BENDL recording can contain at most 65,536 distinct districts.

The `RecordedChain` metadata stores the starting-plan column, population column, tolerance, and
seed. The `--recom-variant` choices match the four common variants in the table above. Add your own
constraints, updaters, and acceptance rule in `chain_runners/gerrychain_cli.py` when adapting the
workflow.

The two MST variants also accept region-column surcharges. For example,
`--region-weights '{"county_id": 1.0}'` makes spanning-tree edges that cross a `county_id` boundary
more expensive. Every named column must exist on the graph nodes. UST variants do not use region
weights.

### Python batch configuration

`pipeline_scripts/run_chains.py` is the user-facing experiment file. Each starting-plan column is
combined with each random seed. `MAX_WORKERS` sets the maximum number that may run simultaneously:

```python
STARTING_PLANS = ("enacted_plan", "alternate_plan")
RNG_SEEDS = (42, 43, 44, 45)
MAX_WORKERS = 2
```

The same settings work for GerryChain and both supported RustReCom modes. For an ordinary
RustReCom ensemble, set:

```python
ENGINE = "rustrecom-chain"
OBJECTIVE_FILE = None
```

For objective-guided search, select the tilted engine and a checked-in objective file:

```python
ENGINE = "rustrecom-tilted"
OBJECTIVE_FILE = OBJECTIVES_DIR / "gingles_partial.json"
MAXIMIZE_OBJECTIVE = True
```

Each tilted run writes both a BENDL recording and a `_scores.csv` file. RustReCom remains a
standalone executable; the Python scheduler starts `rustrecom chain` or `rustrecom tilted` as a
child process for each starting-plan and seed combination.

The scheduler shows one aggregate spinner while chains run. Output from GerryChain and RustReCom,
including their progress indicators, goes to the per-run files in `chain_logs/` instead of being
interleaved in the calling terminal.

The scheduler accepts the same `REGION_WEIGHTS` dictionary for GerryChain and both RustReCom modes.
It forwards the dictionary as GerryChain's region surcharge or RustReCom's region weights, so one
experiment specification can use either engine. Region weights require an MST variant.

### Output names and experiment tags

`EXPERIMENT_TAG` is the short name of the experiment. Use the same tag for runs that belong to one
analysis, such as `baseline`, `beta-0p5`, or `county-split-test`. Tags may contain letters, numbers,
periods, underscores, and hyphens. `pipeline_scripts/run_chains.py` uses the experimenter's current
local date. Values written into filenames cannot contain the reserved `__` field delimiter.

The scheduler adds `PY_` for GerryChain, `RUST_CHAIN_` for ordinary RustReCom, or `RUST_TILTED_`
for tilted RustReCom, then writes:

```text
<ENGINE>_<OUTPUT_PREFIX>__STEPS_<steps>__RNGSEED_<seed>__TOL_<tol>__SEEDPLN__<plan_name>__TAG_<tag>__DATE_<date>.bendl
```

`<plan_name>` is one of the node attributes in `STARTING_PLANS`. The log has the identical stem
with `.log` and records the engine, seed, child output, and exit code. Tilted RustReCom adds
`_scores.csv` to the stem. For example:

```text
PY_VANILLA_PA__STEPS_1000__RNGSEED_42__TOL_0p01__SEEDPLN__seed_plan__TAG_baseline__DATE_2026-08-06.bendl
```

`OUTPUT_PREFIX` supplies `VANILLA_PA`; do not include an automatic engine prefix.

The runner exits nonzero when any run fails and reports the corresponding log path.

### Resource budget

`MAX_WORKERS` controls how many child processes run at once. CPU, memory, and I/O requirements
scale with the number of concurrent processes.

## Scoring a Pennsylvania ensemble

`pipeline_scripts/metrics/collect_data_vanilla_pa.py` demonstrates streaming BENDL evaluation with
`gerrytools.scoring.PlanEvaluator`. It computes:

- district Polsby-Popper and Reock scores;
- cut edges;
- total population, Black population, total voting-age population, and BVAP by district;
- 2020 presidential Democratic seats;
- aggregate Democratic seats across several elections; and
- election-specific disproportionality.

Run the configured data collection stage from the project root with:

```bash
uv run pipeline_scripts/run_data_collection_scripts.py
```

`DATA_COLLECTION_SCRIPTS` at the top of that file lists the collectors to run. The Pennsylvania
collector is included by default, and its own settings select inputs and evaluation batch size:

```python
INPUT_GLOB = "*VANILLA_PA*.bendl"
MAX_WORKERS = 1
BATCH_SIZE = 256
```

The default glob matches direct `VANILLA_PA...` outputs and all Python-runner outputs whose analysis
prefix is `VANILLA_PA`, including the `PY_`, `RUST_CHAIN_`, and `RUST_TILTED_` engines.
`MAX_WORKERS` evaluates independent BENDL files in separate processes, while `BATCH_SIZE` controls
how many plans are scored together while streaming one recording. Results are stored under
`stats/<recording-name>/` and can be opened with `gerrytools.scoring.EnsembleEvalResult`.

Generation and scoring are separate stages. A high-throughput workflow consists of:

1. generate chains for the configured starting plans and seeds;
2. collect the completed recordings and their logs;
3. score the completed files, using `MAX_WORKERS` for process-level parallelism;
4. open the reusable `EnsembleEvalResult` directories in the plotting scripts.

Evaluation is valid when the graph, geometry, assignment order, and column names describe the same
geographic units. The evaluator assumes that the graph, assignment vector, and GeoDataFrame use the
same unit ordering.

The example reads the graph embedded in each BENDL file and uses the matching Pennsylvania
geometry in `data/pa_gdf.parquet`.

## Generating figures

The figure scripts save PNG files and do not require an interactive Matplotlib window.

Run the configured figure generation stage:

```bash
uv run pipeline_scripts/run_figure_generation_scripts.py
```

`FIGURE_GENERATION_SCRIPTS` at the top of that file lists the scripts and their execution order. It
runs the starting-plan maps and the three ensemble figures by default. Run data collection first so
the ensemble figures have evaluation directories to read.

Each ensemble script exposes `STATS_GLOB`, `STARTING_PLAN`, and plot-specific settings at the top of
the file. The default `*VANILLA_PA*` glob matches evaluation directories produced from direct or
Python-scheduled runs. Images appear under `figures/<recording-name>/`; base-plan maps appear under
`figures/plan_maps/`.

`STARTING_PLAN` is one common reference for every directory selected by `STATS_GLOB`. To compare
each starting plan separately, narrow `STATS_GLOB`, change `STARTING_PLAN`, and rerun the script.

## Adapting the project to another state

Adapting the project changes these state-specific inputs:

1. Replace the graph and geometry with data whose node identifiers describe the same geographic
   units.
2. Set the starting-plan and population columns.
3. Update graph paths, output prefixes, assignment columns, population columns, and tolerances in
   the chain scripts.
4. Update every column named by an objective JSON file.
5. Update the metrics list and geometry path in the evaluator.
6. Update the file globs used by the evaluator and ensemble figure scripts.
7. Replace Pennsylvania-specific election columns, county masks, projections, labels, and map
   extents in the figure scripts.

## Troubleshooting

### `uv`, `cargo`, or `rustrecom` is not found

Restart the terminal after installation. uv commonly installs under `~/.local/bin`, and Cargo
installs Rust executables under `~/.cargo/bin`. Confirm those directories are on `PATH`.

### RustReCom reports a missing column

Column names are case-sensitive. Inspect the graph as shown in "Preparing a dual graph," then
check `--assignment-col`, `--pop-col`, every objective field ending in `_col`, and any `--sum-cols`.

### A run refuses to replace an output

Choose a new output path or add `--overwrite-output`. The provided scripts intentionally include
the overwrite flag so rerunning the same seed replaces its earlier example.

### The metrics or ensemble figure command prints nothing

Check `INPUT_GLOB` at the top of the evaluator and `STATS_GLOB` at the top of each ensemble figure
script. Their defaults match names containing `VANILLA_PA`.

### Plotting reports a non-GUI backend

That is expected when no display is attached. Look for the saved path printed by the script under
`figures/`.

### A long run appears stuck

First reproduce the command with a much smaller `--n-steps`. Use `--show-progress`, inspect the
per-run log for batch jobs, and check CPU use and available storage. Tight population tolerances
and the chosen graph or proposal variant can materially affect proposal time.

### A parallel batch reports failed runs

Open the reported file under `chain_logs/`. The runner preserves each child's complete output and
returns a failing status instead of silently continuing.
