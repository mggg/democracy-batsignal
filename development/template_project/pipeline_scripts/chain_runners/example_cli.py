import json
import math
import sys
from collections.abc import Hashable
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
    """Return a BENDL-compatible integer assignment from a graph node attribute."""
    try:
        raw_assignment: dict[Hashable, Any] = {
            node: data[assignment_column] for node, data in node_items(graph)
        }
    except KeyError as error:
        raise click.ClickException(
            f"Starting-plan attribute {assignment_column!r} is missing from at least one node."
        ) from error
    try:
        return {node: int(label) for node, label in raw_assignment.items()}
    except (TypeError, ValueError):
        label_ids: dict[Hashable, int] = {}
        return {
            node: label_ids.setdefault(label, len(label_ids))
            for node, label in raw_assignment.items()
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
