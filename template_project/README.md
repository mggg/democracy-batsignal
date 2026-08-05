# Redistricting Project

This project contains example scripts for running redistricting chains, converting
ensemble files, and calculating common metrics.

Install the Python environment with:

```bash
uv sync
```

The Bash and PowerShell helper scripts are both kept here so the project can be used as
the source for the platform-specific Democracy Batsignal installers.

## RustReCom optimization objectives

An objective converts a districting plan into a numeric score. `rustrecom tilted` uses that
score to favor some valid ReCom proposals over others. Population balance, contiguity, and
other constraints still determine which proposals are valid; the objective only ranks valid
plans. Optimization therefore biases the search rather than guaranteeing a particular result.
Run multiple seeds when exploring an objective.

The PA-ready objective files are in `pipeline_scripts/rustrecom_objectives/`. Pass one to
`rustrecom tilted` with `--objective`:

```bash
rustrecom tilted \
    --graph-json JSON_dualgraphs/pa_dualgraph.json \
    --assignment-col seed_plan \
    --pop-col total_pop_20 \
    --n-steps 1000 \
    --tol 0.01 \
    --rng-seed 42 \
    --objective pipeline_scripts/rustrecom_objectives/gingles_partial.json \
    --maximize true
```

`pipeline_scripts/pa_example_script_opt.sh` contains a complete example with output and
logging. The objective JSON may also be supplied inline, but a file is easier to inspect and
reuse.

### How tilted acceptance works

For each valid proposal, RustReCom evaluates the current and proposed plan scores. A score
improvement is accepted. A worse score may still be accepted so the search can leave local
optima. The direction and worse-plan behavior are controlled by these CLI options:

- `--maximize true`: Higher scores are improvements. Use this for all supplied examples
  except target deviation.
- `--maximize false`: Lower scores are improvements. Use this for
  `minimize_bvap_target_deviation.json`.
- `--accept-rule linear`: Accept a worse plan with probability
  `max(0, 1 - beta * score_loss)`. This is the default.
- `--accept-rule exponential`: Accept a worse plan with probability
  `exp(-beta * score_loss)`.
- `--accept-rule fixed`: Accept any worse plan with the constant probability from
  `--accept-worse-prob`.
- `--acceptance-beta`: Control how strongly `linear` or `exponential` rejects worse scores.
  Larger values make the search greedier. The default is `1.0`.
- `--accept-worse-prob`: Set the probability from `0` to `1` used only by the `fixed` rule.
  `0` is hill climbing; `1` accepts every valid proposal.
- `--scores-output-file`: Write the objective score for each step to CSV.
- `--write-improved-scores-only`: Write only new global-best scores to the score CSV.

The appropriate acceptance strength depends on the scale of the objective. A beta that is
gentle for one score may be nearly deterministic for another. Compare several values and
seeds rather than treating the default as a universal calibration.

### Common objective settings

Every objective file contains an `objective` field that selects the scoring function. Other
fields either set numeric targets or name attributes in the dual graph. Attribute names are
case-sensitive. Population and election columns used by these objectives must contain
integer-valued data on every node.

The supplied files use columns from `JSON_dualgraphs/pa_dualgraph.json`. When adapting an
objective to another graph, update every column-name setting.

### Gingles partial

File: `gingles_partial.json`. Use `--maximize true`.

This objective rewards minority opportunity districts while retaining a gradient toward the
next one. Each district at or above `threshold` contributes `1`. The highest-share district
below the threshold contributes `minority_share / threshold`. All other below-threshold
districts contribute `0`.

For example, three districts at or above 50% and a best remaining district at 45% produce a
score of `3 + 0.45 / 0.50 = 3.9`.

- `objective`: Must be `"gingles_partial"`.
- `threshold`: Minority share required for a full point. It must be between `0` and `1`.
- `min_pop`: Node column containing the minority population. The example uses `bvap_20`.
- `total_pop`: Node column used as the denominator. The example uses `total_vap_20`.

This is a plan-search metric, not a legal determination that a district satisfies the Gingles
preconditions or other Voting Rights Act requirements.

### Banded Gingles partial

File: `banded_gingles_partial.json`. Use `--maximize true`.

This variant rewards minority shares inside a target band instead of rewarding every share
above one threshold. A district inside the inclusive band contributes `1`. The best district
below the band contributes `share / lower_threshold`. Each district above the band contributes
`upper_threshold / share`, so increasingly large overshoots receive progressively less credit.

- `objective`: Must be `"banded_gingles_partial"`.
- `lower_threshold`: Lower edge of the target band, between `0` and `1`.
- `upper_threshold`: Upper edge of the band. It must be at least `lower_threshold` and less
  than `1`.
- `min_pop`: Node column containing the minority population.
- `total_pop`: Node column used as the denominator.

The supplied example targets BVAP shares from 50% through 60%.

### Election wins

File: `democratic_election_wins.json`. Usually use `--maximize true`.

For each election, this objective counts districts won by the selected side. It adds a
fractional tiebreaker for the closest losing district, giving the optimizer useful score
changes between whole-number seat gains. Scores from multiple elections are combined using
`aggregation`.

- `objective`: Must be `"election_wins"`.
- `elections`: List of elections. Each entry names the two node-level vote columns with
  `votes_a` and `votes_b`.
- `target`: Side whose wins are scored, either `"a"` or `"b"`.
- `aggregation`: How to combine election scores:
  - `"mean"` averages them.
  - `"min"` uses the weakest election, favoring plans that perform well across all listed
    elections.
  - `"sum"` adds them.

The supplied example targets Democratic wins (`"a"`) using the 2020 presidential and 2018
U.S. Senate vote columns. Use `--maximize false` if the goal is to reduce the target side's
wins.

### Polsby-Popper compactness

File: `polsby_popper_mean.json`. Use `--maximize true`.

This objective computes `4 * pi * area / perimeter^2` for each district. Values closer to `1`
are more compact. District scores are combined using `aggregation`.

- `objective`: Must be `"polsby_popper"`.
- `area_col`: Node column containing precinct area.
- `shared_perim_col`: Edge column containing the shared perimeter between adjacent precincts.
- `boundary_perim_col`: Node column containing the portion of a precinct boundary on the
  exterior of the state. RustReCom combines it with shared perimeters to derive total
  precinct perimeter.
- `perim_col`: Alternative node column containing total precinct perimeter. One of
  `perim_col` or `boundary_perim_col` is required.
- `aggregation`: `"mean"`, `"min"`, or `"sum"` across district compactness scores. Maximizing
  `"min"` specifically improves the least-compact district.

The supplied example uses the PA graph's `area`, `shared_perim`, and `boundary_perim`
attributes and maximizes mean compactness.

### BVAP target deviation

File: `minimize_bvap_target_deviation.json`. Use `--maximize false`.

This objective calculates a population-of-interest share for every district, then matches the
requested targets to distinct districts. Its score is the smallest possible sum of absolute
differences between matched district shares and targets. Lower is better.

The supplied `[0.5, 0.5]` target asks for two distinct districts whose BVAP shares are as close
as possible to 50%.

- `objective`: `"by_district_abs_deviation"`, or the shorter alias `"abs_deviation"` when
  using the shorthand settings below.
- `target_values`: List of target shares. Each target is matched to a different district.
- `target` and `n_target_districts`: Shorthand for repeating one target. For example,
  `"target": 0.5` and `"n_target_districts": 2` are equivalent to `[0.5, 0.5]`.
- `pov_counts_col`: Node column containing the population of interest. Despite the setting
  name, it can represent any population; the example uses `bvap_20`.
- `total_counts_col`: Node column used as the denominator for each district. The example uses
  `total_vap_20`.
- `total_count`: Alternative fixed global denominator. Use it instead of `total_counts_col`.

### Recording and comparing scores

To inspect optimization behavior, add a score output file:

```bash
--scores-output-file chain_outputs/gingles_scores.csv
```

Add `--write-improved-scores-only` when only the successive global bests are needed. Keep the
RNG seed, objective file, maximize direction, acceptance rule, and beta with the results;
changing any of them changes the search being run.
