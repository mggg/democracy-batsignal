from pathlib import Path

import numpy as np
import pandas as pd
from gerrychain import Graph, Partition
from gerrytools.plotting import ScatterPlot
from gerrytools.scoring import EnsembleEvalResult, disproportionality

ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"


ELECTIONS: tuple[str, ...] = (
    "ag_16",
    "ag_20",
    "aud_16",
    "aud_20",
    "gov_18",
    "pres_16",
    "pres_20",
)


def original_disprop() -> tuple[float, float]:
    """Calculates seed-plan disproportionality across the configured elections.

    Returns:
        tuple[float, float]: Mean disproportionality followed by its variance.
    """
    graph = Graph.from_json(str(ROOT_DIR / "JSON_dualgraphs" / "pa_dualgraph.json"))

    partition = Partition(graph, assignment="seed_plan")

    disprop_values: list[float] = []
    for election in ELECTIONS:
        disprop_values.append(
            disproportionality(
                partition, party_vote_attr=f"{election}_dem", opposition_vote_attr=f"{election}_rep"
            )
        )

    return float(np.mean(disprop_values)), float(np.var(disprop_values))


def create_disprop_scatter(stats_dir: Path) -> Path:
    """Saves the across-election mean/variance scatter for one evaluation run.

    Args:
        stats_dir (Path): Directory containing a completed GerryTools evaluation run.

    Returns:
        Path: Location of the saved scatter plot.

    Raises:
        TypeError: If an election's disproportionality metric is not a pandas Series.
    """
    run = EnsembleEvalResult.open(stats_dir)

    disprop_series: list[pd.Series] = []
    for election in ELECTIONS:
        values = run.read(f"{election}_disprop", expand_repetitions=True)
        if not isinstance(values, pd.Series):
            raise TypeError(f"{election}_disprop did not produce a Series")
        disprop_series.append(values)

    disprop_array = np.array(disprop_series)

    mean_disprop = np.mean(disprop_array, axis=0)
    var_disprop = np.var(disprop_array, axis=0)

    scatter = ScatterPlot()

    scatter.add_series(mean_disprop, var_disprop, markerfacecolor="default_grey", markersize=3)
    x, y = original_disprop()
    scatter.add_point(
        x,
        y,
        markerfacecolor="cherryblossompink",
        markersize=10,
        markeredgecolor="black",
        name="Original Plan",
    )

    scatter.add_vertical_lines(0)

    scatter.set_xlim(-0.2, 0.2)
    scatter.set_ylim(0, 0.015)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / stats_dir.name / f"disprop_scatter_{stats_dir.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scatter.save(str(output_path))
    return output_path


def main() -> None:
    """Generates a scatter plot for every vanilla Pennsylvania evaluation run."""
    stats_base_dir = ROOT_DIR / "stats"
    for stats_dir in stats_base_dir.glob("VANILLA_PA*"):
        print(f"Processing '{stats_dir.name}' ...")

        output_path = create_disprop_scatter(stats_dir)
        print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
