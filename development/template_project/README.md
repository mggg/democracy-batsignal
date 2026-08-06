# Redistricting Project

This is a runnable example of a redistricting ensemble workflow. It includes two ways to generate
ReCom chains, BENDL recording, GerryTools scoring, and figures built from Pennsylvania data.

- Use **RustReCom** for fast ordinary chains and objective-guided searches.
- Use **GerryChain** when you want to build or modify the proposal, constraints, acceptance rule,
  or updaters in Python.
- Use **binary-ensemble** to inspect BENDL recordings.
- Use **GerryTools** to score recordings and create plots.

The examples are starting points. Read through a script before increasing its step count or using
it with another graph.

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

The Democracy Batsignal installer has already created and synchronized the environment. From the
project root, verify it with:

```bash
uv run python --version
uv run python -c "import gerrychain, gerrytools, binary_ensemble; print('Python tools ready')"
```

If you change `pyproject.toml`, update the environment with:

```bash
uv sync
```

If you chose to install RustReCom, verify the standalone executable:

```bash
rustrecom --version
rustrecom --help
```

RustReCom is not installed in the uv environment. It is a Rust executable installed through
Cargo. The GerryChain examples still work if you skipped it.

### A small first run

The RustReCom scripts default to substantial runs. Start with 100 steps and one seed.

On macOS or Linux, edit `n_steps` and `rng_seed` near the top of
`pipeline_scripts/pa_example_script_vanilla.sh`, then run:

```bash
bash pipeline_scripts/pa_example_script_vanilla.sh
```

On Windows:

```powershell
.\pipeline_scripts\pa_example_script_vanilla.ps1 -NSteps 100 -RngSeeds 42
```

The resulting BENDL recording appears in `chain_outputs/`. Once that works, increase the step
count and run multiple seeds for the analysis you actually need.

## Project layout

```text
.
├── JSON_dualgraphs/
│   ├── gerrymandria.json       # small graph for the Python examples
│   └── pa_dualgraph.json       # Pennsylvania graph used by RustReCom and scoring
├── data/
│   ├── pa_gdf.parquet          # Pennsylvania geometry used by scoring and figures
│   └── alt_plan_pa.json        # one alternate assignment used by a comparison map
├── pipeline_scripts/
│   ├── example_cli.py          # GerryChain RecordedChain CLI
│   ├── pa_example_script_vanilla.*
│   ├── pa_example_script_opt.*
│   ├── rustrecom_objectives/   # example objective JSON files
│   ├── metrics/                # GerryTools evaluation
│   └── figure_generators/      # maps and ensemble plots
├── batch_example_python_cli_simple.*
├── batch_example_python_cli_parallel.*
├── chain_outputs/              # BENDL recordings
├── chain_logs/                 # redirected batch output
├── stats/                      # EvaluationRun data
└── figures/                    # generated PNG files
```

Projects created by the Bash installer contain `.sh` helpers. Projects created by the PowerShell
installer contain equivalent `.ps1` helpers. All Python scripts are cross-platform.

## Choose a chain workflow

The project provides two chain-generation routes:

1. `rustrecom chain` is the quickest route for high-throughput ordinary ReCom ensembles.
2. `pipeline_scripts/example_cli.py` records a GerryChain ReCom chain from Python. Use this route
   when the Python API is more important than RustReCom's speed.

RustReCom also provides two objective-guided search commands:

- `rustrecom tilted` runs one chain and probabilistically favors score improvements.
- `rustrecom short-bursts` runs fixed-length bursts and starts each new burst from the best plan
  found in the preceding burst.

An objective-guided run is not a neutral ensemble sample. It is a search for plans that score well
under the selected objective.

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

Inspect the available columns before adapting a script:

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

The complete Pennsylvania examples are `pipeline_scripts/pa_example_script_vanilla.sh` and
`pipeline_scripts/pa_example_script_vanilla.ps1`. Their central command is:

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

### Core chain options

- `--graph-json` points to the NetworkX JSON dual graph.
- `--assignment-col` names the node attribute containing the starting district labels.
- `--pop-col` names the node population attribute used to balance districts.
- `--n-steps` is the number of proposals to generate. Runtime and output size grow with it.
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
- `--n-threads` and `--batch-size` control RustReCom's internal parallel proposal generation. Start
  with their defaults and benchmark before changing them.

Run `rustrecom chain --help` for the complete option list and the installed version's defaults.

### ReCom variants

The four common variants combine two district-pair rules with two spanning-tree rules:

| Variant | District pair | Spanning tree |
| --- | --- | --- |
| `cut-edges-mst` | Select through a cut edge | Minimum spanning tree |
| `district-pairs-mst` | Select an adjacent district pair | Minimum spanning tree |
| `cut-edges-ust` | Select through a cut edge | Uniform spanning tree |
| `district-pairs-ust` | Select an adjacent district pair | Uniform spanning tree |

The ordinary `chain` command also exposes region-aware and reversible variants. Their assumptions
and tuning differ from the four examples above, so consult `rustrecom chain --help` before using
them. Do not compare ensembles produced by different variants as though only the runtime changed;
the proposal distribution changed too.

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

Prefer a checked-in JSON file when a run needs many non-default settings. Keep that config with the
results so the run can be reconstructed.

## BENDL recordings

BENDL is the default format in this project because one file carries:

- the compressed assignment stream;
- the dual graph used by the run; and
- metadata such as population column, tolerance, seed, variant, and step count.

Use `binary_ensemble.BendlDecoder` to inspect a recording. binary-ensemble 2.0 is a Python package;
it does not install a `ben` shell command.

```python
from binary_ensemble import BendlDecoder

recording = BendlDecoder("chain_outputs/VANILLA_PA__STEPS_1000__RNGSEED_42__TOL_0p01.bendl")

recording.verify()             # Raises if the bundle or stream is corrupt.
print(recording.count_samples())
print(recording.read_metadata())

graph = recording.read_graph()
first_assignment = recording.lookup(0)
```

Other useful methods include `list_assets()`, `subsample_every()`, `subsample_range()`, and
`extract_stream()`. An assignment returned by `lookup()` uses the recording's node order. Prefer
the included GerryTools streaming evaluator when scoring a full recording rather than manually
materializing every assignment in memory.

## RustReCom optimization objectives

An objective converts a districting plan into a numeric score. `rustrecom tilted` uses that score
to favor some valid ReCom proposals over others. Population balance, contiguity, and other
constraints still determine which proposals are valid; the objective only changes the search
preference among proposals. Optimization therefore biases the search rather than guaranteeing a
particular result. Run multiple seeds when exploring an objective.

The Pennsylvania-ready objective files are in `pipeline_scripts/rustrecom_objectives/`. Pass one
to `rustrecom tilted` with `--objective`:

```bash
rustrecom tilted \
    --graph-json JSON_dualgraphs/pa_dualgraph.json \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 1000 \
    --tol 0.01 \
    --rng-seed 42 \
    --objective pipeline_scripts/rustrecom_objectives/gingles_partial.json \
    --maximize true \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file chain_outputs/gingles_example.bendl \
    --scores-output-file chain_outputs/gingles_example_scores.csv \
    --overwrite-output \
    --show-progress
```

`pipeline_scripts/pa_example_script_opt.sh` or its PowerShell counterpart contains a complete
two-seed example. `--objective` also accepts inline JSON, but a file is easier to inspect, reuse,
and preserve with the results.

### Tilted acceptance

For each valid proposal, RustReCom evaluates the current and proposed plan scores. A score
improvement is accepted. A worse score may still be accepted so the search can leave local optima.
The direction and worse-plan behavior are controlled by these options:

- `--maximize true`: Higher scores are improvements. Use this for all supplied examples except
  target deviation.
- `--maximize false`: Lower scores are improvements. Use this for
  `minimize_bvap_target_deviation.json`.
- `--accept-rule linear`: Accept a worse plan with probability
  `max(0, 1 - beta * score_loss)`. This is the default.
- `--accept-rule exponential`: Accept a worse plan with probability
  `exp(-beta * score_loss)`.
- `--accept-rule fixed`: Accept any worse plan with the probability from `--accept-worse-prob`.
- `--acceptance-beta`: Control how strongly `linear` or `exponential` rejects worse scores. Larger
  values make the search greedier. The default is `1.0`.
- `--accept-worse-prob`: Set the probability from `0` to `1` used only by the fixed rule. `0` is
  hill climbing; `1` accepts every valid proposal.
- `--scores-output-file`: Write objective scores and per-district scores to CSV.
- `--write-improved-scores-only`: Write only new global-best rows to the score CSV.

The useful acceptance strength depends on the scale of the objective. A beta that is gentle for
one score may be nearly deterministic for another. Compare acceptance behavior across several
values and seeds instead of treating the default as a universal calibration.

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
    --objective pipeline_scripts/rustrecom_objectives/gingles_partial.json \
    --maximize true \
    --variant district-pairs-mst \
    --writer bendl \
    --output-file chain_outputs/gingles_short_bursts.bendl \
    --overwrite-output \
    --show-progress
```

Short bursts returns full partitions from its workers. Prefer `assignments`, `canonical`, `ben`, or
`bendl` output; proposal-level writers cannot report the same proposal details for these workers.
Run `rustrecom short-bursts --help` for the exact behavior of the installed version.

### Objective JSON settings

Every objective file contains an `objective` field selecting the scoring function. Other fields
set numeric targets or name attributes in the dual graph. Attribute names are case-sensitive.
Population and election columns used by these objectives must contain integer-valued data on every
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
differences between matched district shares and targets. Lower is better.

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

`pipeline_scripts/example_cli.py` shows the current GerryChain and GerryTools recording pattern:

1. load a `gerrychain.Graph`;
2. construct a `gerrytools.ben.RecordedChain`;
3. assign its initial `Partition`, including the population updater;
4. set its ReCom proposal function; and
5. iterate the chain to write a BENDL recording.

See all CLI options with:

```bash
uv run pipeline_scripts/example_cli.py --help
```

A small Pennsylvania run is:

```bash
uv run pipeline_scripts/example_cli.py \
    --graph-path JSON_dualgraphs/pa_dualgraph.json \
    --output-path chain_outputs/PA_chain_100_steps_seed42.bendl \
    --starting-plan seed_plan \
    --pop-col total_pop_20 \
    --rng-seed 42 \
    --population-tolerance 0.01 \
    --total-steps 100
```

`--starting-plan` is a node attribute name, not a path to an assignment file. The CLI converts its
labels to BENDL-compatible integer district IDs. Existing integer-like labels retain their integer
values; other hashable labels receive stable IDs based on graph iteration order.

The `RecordedChain` metadata stores the starting-plan column, population column, tolerance, and
seed. Add your own constraints, updaters, and acceptance rule in `example_cli.py` when adapting the
workflow.

### Sequential and parallel batches

`batch_example_python_cli_simple` runs several seeds sequentially. It contains a small
Gerrymandria batch followed by a larger Pennsylvania run. Reduce both step-count settings when
testing.

`batch_example_python_cli_parallel` runs independent Gerrymandria seeds concurrently and writes a
separate log for each seed. Its default concurrency is the detected processor count. On a shared
machine, lower `MAX_JOBS` in Bash or pass `-MaxJobs` in PowerShell:

```powershell
.\batch_example_python_cli_parallel.ps1 -MaxJobs 4 -RngSeeds 1,2,3,4 -TotalSteps 100
```

Parallelize independent chains, not steps that must belong to one Markov chain. Assign every run a
different RNG seed and inspect `chain_logs/` if a background job fails.

## Scoring a Pennsylvania ensemble

`pipeline_scripts/metrics/collect_data_vanilla_pa.py` demonstrates streaming BENDL evaluation with
`gerrytools.scoring.PlanEvaluator`. It computes:

- district Polsby-Popper and Reock scores;
- cut edges;
- total population, Black population, total voting-age population, and BVAP by district;
- 2020 presidential Democratic seats;
- aggregate Democratic seats across several elections; and
- election-specific disproportionality.

Run it from anywhere inside the project with:

```bash
uv run pipeline_scripts/metrics/collect_data_vanilla_pa.py
```

The script deliberately processes only files matching `chain_outputs/VANILLA_PA*.bendl`. This
matches the ordinary RustReCom example. If your recordings use another naming convention, change
the glob in `main()`. Results are stored under `stats/<recording-name>/` and can be opened with
`gerrytools.scoring.EvaluationRun`.

When adapting the evaluator, keep its graph, geometry, assignment order, and column names aligned.
The example reads the graph embedded in each BENDL file and uses the matching Pennsylvania
geometry in `data/pa_gdf.parquet`.

## Generating figures

The figure scripts save PNG files and do not require an interactive Matplotlib window.

Generate the starting-plan, Philadelphia, partisan choropleth, and alternate-plan maps:

```bash
uv run pipeline_scripts/figure_generators/base_plan_figures.py
```

After running the metrics collector, generate the ensemble figures:

```bash
uv run pipeline_scripts/figure_generators/cut_edges_histogram.py
uv run pipeline_scripts/figure_generators/disprop_scatter.py
uv run pipeline_scripts/figure_generators/reock_boxplot.py
```

These three scripts process directories matching `stats/VANILLA_PA*`. Their images appear under
`figures/<recording-name>/`. The base-plan maps appear under `figures/plan_maps/`.

A message saying Matplotlib selected the non-GUI `agg` backend is normal in a terminal, remote
session, or container. The scripts call `save()`, so the PNG path printed afterward is the result.

## Adapting the project to another state

Work through these changes in order:

1. Replace the graph and geometry with matching data. Confirm that node identifiers refer to the
   same geographic units in both.
2. Choose and validate the starting-plan and population columns.
3. Update graph paths, output prefixes, assignment columns, population columns, and tolerances in
   the chain scripts.
4. Update every column named by an objective JSON file.
5. Update the metrics list and geometry path in the evaluator.
6. Update the file globs used by the evaluator and ensemble figure scripts.
7. Replace Pennsylvania-specific election columns, county masks, projections, labels, and map
   extents in the figure scripts.
8. Run a very short chain first, verify its BENDL file, score it, and generate every figure before
   starting full runs.

Do not assume that a column with the same name has the same units or definition in another data
source. In particular, distinguish total population from voting-age population and document the
election and demographic vintages used by an analysis.

## Reproducibility and interpretation

- Keep the graph, starting assignment, software versions, seed, population settings, ReCom
  variant, and objective configuration with the results.
- Use multiple independent seeds. A single seed describes one pseudorandom trajectory.
- Do not treat a tilted or short-bursts output as a representative neutral ensemble. Its purpose is
  optimization under a stated score.
- Objective values summarize the implemented formula and supplied columns. They do not establish
  legal compliance, causation, or a unique best plan.

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

Check the input naming convention. The evaluator looks for `chain_outputs/VANILLA_PA*.bendl`, and
the ensemble figures look for `stats/VANILLA_PA*` directories.

### Plotting reports a non-GUI backend

That is expected when no display is attached. Look for the saved path printed by the script under
`figures/`.

### A long run appears stuck

First reproduce the command with a much smaller `--n-steps`. Use `--show-progress`, inspect the
per-seed log for batch jobs, and check CPU use and available storage. Tight population tolerances
and the chosen graph or proposal variant can materially affect proposal time.
