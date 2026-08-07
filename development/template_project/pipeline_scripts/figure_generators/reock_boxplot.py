from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from gerrychain import Graph, Partition
from gerrytools.plotting import BoxPlot
from gerrytools.scoring import EnsembleEvalResult, reock

ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"

# Figure settings: edit this block, then run this file with `uv run`.
STATS_GLOB = "*VANILLA_PA*"
STARTING_PLAN = "seed_plan"


def original_reock() -> pd.Series:
    """Calculates seed-plan Reock scores ordered from least to most compact district.

    Returns:
        pd.Series: Ordered district Reock scores with a zero-based index.
    """
    graph = Graph.from_json(str(ROOT_DIR / "JSON_dualgraphs" / "pa_dualgraph.json"))
    gdf = gpd.read_parquet(ROOT_DIR / "data" / "pa_gdf.parquet").to_crs("EPSG:5070")

    partition = Partition(graph, assignment=STARTING_PLAN)

    vals = reock(partition, geometry=gdf).sort_values().reset_index(drop=True)
    return vals


def create_reock_boxes(stats_dir: Path) -> Path:
    """Saves Reock distributions by within-plan district rank for one evaluation run.

    Args:
        stats_dir (Path): Directory containing a completed GerryTools evaluation run.

    Returns:
        Path: Location of the saved box plot.

    """
    run = EnsembleEvalResult.open(stats_dir)
    reock_scores = run.read("reock", expand_repetitions=True, return_type="dataframe")
    scores_array = np.sort(reock_scores.to_numpy().T, axis=0)

    bp = BoxPlot(figure_size=(30, 10))

    bp.add_dataset(scores_array)
    district_ranks = range(1, len(scores_array) + 1)
    bp.set_xticks(district_ranks, labels=[str(rank) for rank in district_ranks])
    bp.set_tick_style("x", size=6)

    bp.add_pointset(
        original_reock(),
        facecolor="cherryblossompink",
        markersize=5,
        markeredgecolor="black",
        name="Original Plan",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / stats_dir.name / f"reock_boxplots_{stats_dir.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bp.save(str(output_path))
    return output_path


def main() -> None:
    """Generates a Reock box plot for every vanilla Pennsylvania evaluation run."""
    stats_base_dir = ROOT_DIR / "stats"
    stats_dirs = sorted(stats_base_dir.glob(STATS_GLOB))
    if not stats_dirs:
        raise FileNotFoundError(f"No directories in {stats_base_dir} match {STATS_GLOB!r}.")

    for stats_dir in stats_dirs:
        print(f"Processing '{stats_dir.name}' ...")

        output_path = create_reock_boxes(stats_dir)
        print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
