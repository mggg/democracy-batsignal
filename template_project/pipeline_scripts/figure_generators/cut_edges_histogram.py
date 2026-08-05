from pathlib import Path

import pandas as pd
from gerrychain import Graph, Partition
from gerrytools.plotting import Histogram
from gerrytools.scoring import EnsembleEvalResult, cut_edges

ROOT_DIR = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "figures"


def original_cut_edges() -> int:
    """Calculates the cut-edge count for the graph's seed plan.

    Returns:
        int: Number of cut edges in the seed plan.
    """
    graph = Graph.from_json(str(ROOT_DIR / "JSON_dualgraphs" / "pa_dualgraph.json"))

    partition = Partition(graph, assignment="seed_plan")

    cut_edge_count = cut_edges(partition)
    if not isinstance(cut_edge_count, (int, float)):
        raise TypeError("cut_edges did not produce an int")
    return int(cut_edge_count)


def create_cut_edge_hist(stats_dir: Path) -> Path:
    """Saves a cut-edge histogram for one streamed evaluation run.

    Args:
        stats_dir (Path): Directory containing a completed GerryTools evaluation run.

    Returns:
        Path: Location of the saved histogram.

    Raises:
        TypeError: If the cut-edge metric is not stored as a pandas Series.
    """
    run = EnsembleEvalResult.open(stats_dir)

    cut_edge_scores = run.read("cut_edges", expand_repetitions=True)
    if not isinstance(cut_edge_scores, pd.Series):
        raise TypeError("cut_edges did not produce a Series")

    hist = Histogram()
    hist.add_dataset(cut_edge_scores)

    hist.add_vertical_lines([original_cut_edges()], linecolor="cherryblossompink", linewidth=2)

    hist.set_bin_widths(10)
    hist.set_xlim(2750, 3550)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / stats_dir.name / f"cut_edges_histogram_{stats_dir.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hist.save(str(output_path))
    return output_path


def main() -> None:
    """Generates a histogram for every vanilla Pennsylvania evaluation run."""
    stats_base_dir = ROOT_DIR / "stats"
    for stats_dir in stats_base_dir.glob("VANILLA_PA*"):
        print(f"Processing '{stats_dir.name}' ...")

        output_path = create_cut_edge_hist(stats_dir)
        print(f"Saved to '{output_path}'")


if __name__ == "__main__":
    main()
