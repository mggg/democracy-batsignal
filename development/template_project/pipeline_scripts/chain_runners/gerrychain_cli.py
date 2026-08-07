"""Run one GerryChain recording behind the parallel batch interface.

The batch runner launches each chain as a child process so GerryChain and RustReCom use the
same scheduling, logging, interruption, and failure-handling code. This CLI is that subprocess
boundary for GerryChain. It keeps the orchestrator's one-chain command syntax consistent across
engines; users normally edit ``pipeline_scripts/run_chains.py`` instead.
"""

import json
import math
import sys
from collections.abc import Hashable
from numbers import Number
from operator import eq
from pathlib import Path
from typing import Any

import click
from gerrychain import Graph, Partition
from gerrychain.proposals import ReCom
from gerrychain.updaters import Tally
from gerrytools.ben import RecordedChain

RECOM_VARIANTS = {
    "cut-edges-mst": ReCom.cut_edges_mst,
    "cut-edges-ust": ReCom.cut_edges_ust,
    "district-pairs-mst": ReCom.district_pairs_mst,
    "district-pairs-ust": ReCom.district_pairs_ust,
}
# binary-ensemble 2.0 stores assignment labels as unsigned 16-bit integers.
MAX_BENDL_DISTRICT_ID = (1 << 16) - 1


def node_items(graph: Any) -> list[tuple[Hashable, dict[str, Any]]]:
    """Return node-data pairs from a GerryChain 1.0 or NetworkX graph."""
    if isinstance(graph, Graph):
        return [(node, graph.node_data(node)) for node in graph.nodes]
    return list(graph.nodes(data=True))


def load_graph(graph_path: Path) -> Graph:
    """Load a GerryChain graph from JSON or a GIS file.

    Args:
        graph_path: Path to the graph file.

    Returns:
        The loaded graph.

    Raises:
        click.ClickException: If GerryChain cannot load the graph.
    """
    try:
        if graph_path.suffix.lower() == ".json":
            return Graph.from_json(str(graph_path))
        return Graph.from_file(str(graph_path))
    except Exception as error:
        raise click.ClickException(f"Failed to load graph from {graph_path}: {error}") from error


def integer_assignment(graph: Any, assignment_column: str) -> dict[Hashable, int]:
    """Return a lossless BENDL-compatible assignment from a graph node attribute.

    Note: This function exists primarily as a protective measure against obviously wrong district
    labels so that the pipeline will run. In all likelihood, most users will have districts with
    integer IDs and this function will be unnecessary.

    Integer-like labels retain their values when conversion is one-to-one and the IDs are between
    0 and 65,535. Other hashable labels receive stable integer IDs based on graph iteration order.
    BENDL can represent at most 65,536 distinct labels.

    Args:
        graph: GerryChain or NetworkX graph containing the assignment attribute.
        assignment_column: Node attribute containing each unit's district label.

    Returns:
        Integer district labels keyed by graph node.

    Raises:
        click.ClickException: If a label is missing, non-finite, unhashable, or otherwise invalid.
    """
    raw_assignment: dict[Hashable, Any] = {}
    try:
        for node, data in node_items(graph):
            label = data[assignment_column]
            if label is None:
                raise click.ClickException(
                    f"Starting-plan attribute {assignment_column!r} contains a missing label."
                )
            # Some missing-value sentinels are not numeric but compare unequal to themselves.
            # e.g., numpy.nan, pandas.NA, and pd.NA
            try:
                self_equal = bool(eq(label, label))
            except (TypeError, ValueError) as error:
                raise click.ClickException(
                    f"Starting-plan attribute {assignment_column!r} contains an invalid label: "
                    f"{label!r}."
                ) from error
            if not self_equal:
                raise click.ClickException(
                    f"Starting-plan attribute {assignment_column!r} contains a missing label."
                )
            if isinstance(label, Number):
                try:
                    finite = math.isfinite(label)
                except TypeError as error:
                    raise click.ClickException(
                        f"Starting-plan attribute {assignment_column!r} contains a non-real "
                        f"numeric label: {label!r}."
                    ) from error
                if not finite:
                    raise click.ClickException(
                        f"Starting-plan attribute {assignment_column!r} contains a non-finite "
                        f"numeric label: {label!r}."
                    )
            try:
                hash(label)
            except TypeError as error:
                raise click.ClickException(
                    f"Starting-plan attribute {assignment_column!r} contains an unhashable label: "
                    f"{label!r}."
                ) from error
            raw_assignment[node] = label
    except KeyError as error:
        raise click.ClickException(
            f"Starting-plan attribute {assignment_column!r} is missing from at least one node."
        ) from error

    distinct_label_count = len(set(raw_assignment.values()))
    if distinct_label_count > MAX_BENDL_DISTRICT_ID + 1:
        raise click.ClickException(
            f"Starting-plan attribute {assignment_column!r} contains {distinct_label_count:,} "
            f"distinct labels; BENDL supports at most {MAX_BENDL_DISTRICT_ID + 1:,}."
        )

    try:
        converted = {node: int(label) for node, label in raw_assignment.items()}
    except (OverflowError, TypeError, ValueError):
        converted = None

    if converted is not None:
        numeric_labels_are_exact = all(
            not isinstance(label, Number) or label == converted[node]
            for node, label in raw_assignment.items()
        )
        conversion_is_one_to_one = len(set(converted.values())) == distinct_label_count
        conversion_is_in_range = all(
            0 <= label <= MAX_BENDL_DISTRICT_ID for label in converted.values()
        )
        if numeric_labels_are_exact and conversion_is_one_to_one and conversion_is_in_range:
            return converted

    label_ids: dict[Any, int] = {}
    return {
        node: label_ids.setdefault(label, len(label_ids)) for node, label in raw_assignment.items()
    }


def parse_region_weights(
    _context: click.Context,  # Click supplies context to every option callback.
    parameter: click.Parameter,
    value: str | None,
) -> dict[str, float] | None:
    """Parses region-aware ReCom surcharges from a command-line JSON object.

    Args:
        _context (click.Context): Click command context. It is unused by this parser.
        parameter (click.Parameter): Option being parsed, used in any error message.
        value (str | None): JSON object supplied to ``--region-weights``, or ``None`` when the
            option was omitted.

    Returns:
        dict[str, float] | None: Region columns and their finite numeric surcharges, or ``None``
            when no surcharges were supplied.

    Raises:
        click.BadParameter: If the value is not a JSON object with non-empty string keys and
            finite numeric values.
    """
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise click.BadParameter("must be a JSON object", param=parameter) from error
    if not isinstance(parsed, dict):
        raise click.BadParameter("must be a JSON object", param=parameter)

    weights: dict[str, float] = {}
    for column, weight in parsed.items():
        if not isinstance(column, str) or not column:
            raise click.BadParameter("column names must be non-empty strings", param=parameter)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
        ):
            raise click.BadParameter("weights must be finite numbers", param=parameter)
        weights[column] = float(weight)
    return weights or None


@click.command()
@click.option(
    "--graph-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="GerryChain JSON graph or GIS file.",
)
@click.option(
    "--output-path",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    required=True,
    help="Destination .bendl file. An existing file is replaced after a successful run.",
)
@click.option("--starting-plan", type=str, required=True, help="Starting-plan node attribute.")
@click.option("--pop-col", type=str, required=True, help="Population node attribute.")
@click.option("--rng-seed", type=int, required=True, help="Random seed for the chain.")
@click.option(
    "--recom-variant",
    type=click.Choice(tuple(RECOM_VARIANTS)),
    default="district-pairs-mst",
    show_default=True,
    help="GerryChain 1.0 ReCom proposal variant.",
)
@click.option(
    "--region-weights",
    callback=parse_region_weights,
    help="Region-column surcharges as JSON, for example '{\"COUNTY\": 1.0}'. MST only.",
)
@click.option(
    "--population-tolerance",
    type=click.FloatRange(min=0, max=1),
    default=0.01,
    show_default=True,
    help="Allowed fractional population deviation in each ReCom proposal.",
)
@click.option(
    "--total-steps",
    type=click.IntRange(min=1),
    default=10_000,
    show_default=True,
    help="Number of plans to record, including the initial plan.",
)
def main(
    graph_path: Path,
    output_path: Path,
    starting_plan: str,
    pop_col: str,
    rng_seed: int,
    recom_variant: str,
    region_weights: dict[str, float] | None,
    population_tolerance: float,
    total_steps: int,
) -> None:
    """Run a ReCom chain and record it as a BENDL."""
    if region_weights is not None and not recom_variant.endswith("-mst"):
        raise click.ClickException("--region-weights is only valid with an MST ReCom variant.")

    graph = load_graph(graph_path)
    if any(pop_col not in data for _, data in node_items(graph)):
        raise click.ClickException(
            f"Population attribute {pop_col!r} is missing from at least one node."
        )

    chain = RecordedChain(
        graph,
        output_path=output_path,
        total_steps=total_steps,
        rng=rng_seed,
        metadata={
            "starting_plan": starting_plan,
            "population_column": pop_col,
            "population_tolerance": population_tolerance,
            "recom_variant": recom_variant,
            "region_weights": region_weights,
            "rng_seed": rng_seed,
        },
    )

    chain.initial_partition = Partition(
        chain.graph,
        assignment=integer_assignment(chain.graph, starting_plan),
        updaters={"population": Tally(pop_col, alias="population")},
    )

    ideal_population = sum(chain.initial_partition["population"].values()) / len(
        chain.initial_partition
    )
    if region_weights is None:
        chain.proposal_fn = RECOM_VARIANTS[recom_variant](
            pop_col=pop_col,
            pop_target=ideal_population,
            epsilon=population_tolerance,
        )
    else:
        mst_proposal = (
            ReCom.cut_edges_mst if recom_variant == "cut-edges-mst" else ReCom.district_pairs_mst
        )
        chain.proposal_fn = mst_proposal(
            pop_col=pop_col,
            pop_target=ideal_population,
            epsilon=population_tolerance,
            region_surcharge=region_weights,
        )

    with click.progressbar(
        chain.allow_overwrite(),
        length=total_steps,
        label=f"Recording {output_path.name}",
        file=sys.stderr,
    ) as partitions:
        for _ in partitions:
            pass

    click.echo(f"Recorded {len(chain.recording):,} plans to {output_path}.", err=True)


if __name__ == "__main__":
    main()
