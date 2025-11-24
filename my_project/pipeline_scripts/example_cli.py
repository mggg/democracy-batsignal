from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally
from gerrychain.accept import always_accept
from gerrychain.proposals.tree_proposals import recom
from functools import partial
import random
import jsonlines as jl
import click
import numpy as np
from pathlib import Path
from pyben import PyBenEncoder
import sys


@click.command()
@click.option("--graph-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(writable=True, dir_okay=False))
@click.option("--starting-plan", type=str)
@click.option("--pop-col", type=str)
@click.option("--rng-seed", type=int)
@click.option("--population-tolerance", type=float, default=0.01)
@click.option("--total-steps", type=int, default=10_000)
@click.option("--writeas", type=click.Choice(["jsonl", "ben"]), default="ben")
def main(
    graph_path,
    output_path,
    starting_plan,
    pop_col,
    rng_seed,
    population_tolerance,
    total_steps,
    writeas,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        if graph_path.endswith(".json"):
            graph = Graph.from_json(graph_path)
        else:
            graph = Graph.from_file(graph_path)
    except Exception as e:
        raise ValueError(f"Failed to load graph from {graph_path}: {e}")

    initial_partition = Partition(
        graph,
        assignment=starting_plan,
        updaters={"population": Tally(pop_col, alias="population")},
    )

    ideal_pop = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(
        recom,
        pop_col=pop_col,
        pop_target=ideal_pop,
        epsilon=population_tolerance,
        node_repeats=1,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[],
        initial_state=initial_partition,
        total_steps=total_steps,
        accept=always_accept,
    )

    graph_node_order = list(graph.nodes)

    # This will print to the standard error stream so that logging does not interfere with the
    # standard output.
    print(
        f"Writing output to '{Path(output_path).name}' in '{writeas.upper()}' format.",
        file=sys.stderr,
        flush=True,
    )
    match writeas:
        case "jsonl":
            with jl.open(output_path, "w") as writer:
                for i, partition in enumerate(chain.with_progress_bar()):
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    writer.write(
                        {
                            "assignment": ordered_assignment,
                            "sample": i + 1,
                        }
                    )

        case "ben":
            with PyBenEncoder(output_path, overwrite=True) as encoder:
                for partition in chain.with_progress_bar():
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    encoder.write(ordered_assignment)

        case _:
            raise ValueError(f"Unsupported writeas format: {writeas}")


if __name__ == "__main__":
    main()
