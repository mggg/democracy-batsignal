"""Run and record a GerryChain ReCom chain."""

import sys
from collections.abc import Hashable
from functools import partial
from pathlib import Path
from typing import Any

import click
from gerrychain import Graph, Partition
from gerrychain.proposals import recom
from gerrychain.updaters import Tally
from gerrytools.ben import RecordedChain


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
    raw_assignment: dict[Hashable, Any] = {
        node: data[assignment_column] for node, data in graph.nodes(data=True)
    }
    try:
        return {node: int(label) for node, label in raw_assignment.items()}
    except (TypeError, ValueError):
        label_ids: dict[Hashable, int] = {}
        return {
            node: label_ids.setdefault(label, len(label_ids))
            for node, label in raw_assignment.items()
        }


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
    population_tolerance: float,
    total_steps: int,
) -> None:
    """Run a ReCom chain and record it as a BENDL."""

    graph = load_graph(graph_path)

    chain = RecordedChain(
        graph,
        output_path=output_path,
        total_steps=total_steps,
        rng=rng_seed,
        metadata={
            "starting_plan": starting_plan,
            "population_column": pop_col,
            "population_tolerance": population_tolerance,
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
    chain.proposal_fn = partial(
        recom,
        pop_col=pop_col,
        pop_target=ideal_population,
        epsilon=population_tolerance,
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
