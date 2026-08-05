from pathlib import Path
from gerrytools.scoring import (
    AggregateSeats,
    Disproportionality,
    PlanEvaluator,
    PolsbyPopper,
    Reock,
    Tally,
    CutEdges,
    Seats,
)
from binary_ensemble import BendlDecoder
import geopandas as gpd


ROOT_DIR = Path(__file__).resolve().parents[2]


def collect_results(bendl_file: Path, stats_dir: Path):
    """
    Collects results from the specified directories and saves them to a JSONL file.

    Args:
        bendl_dir (Path): Directory containing .bendl files.
        stats_dir (Path): Directory containing .jsonl files with statistics.
    """
    decoder = BendlDecoder(bendl_file)
    graph = decoder.read_graph()

    if graph is None:
        raise ValueError(f"Graph could not be read from {bendl_file}")

    gdf = gpd.read_parquet(ROOT_DIR / "data" / "pa_gdf.parquet").to_crs("EPSG:5070")

    evaluator = PlanEvaluator(graph=graph, geometry=gdf)

    evaluator.add_metrics(
        PolsbyPopper(),
        Reock(),
        CutEdges(),
        Tally("total_pop_20", "bpop_20", result_name="base_populations"),
        Tally("total_vap_20", "bvap_20", result_name="base_voting_age_populations"),
        Seats("pres_20_dem", "pres_20_rep", result_name="pres_20_seats"),
        AggregateSeats(
            (
                "ag_16_dem",
                "ag_20_dem",
                "aud_16_dem",
                "aud_20_dem",
                "gov_18_dem",
                "pres_16_dem",
                "pres_20_dem",
            ),
            (
                "ag_16_rep",
                "ag_20_rep",
                "aud_16_rep",
                "aud_20_rep",
                "gov_18_rep",
                "pres_16_rep",
                "pres_20_rep",
            ),
        ),
        Disproportionality("ag_16_dem", "ag_16_rep", result_name="ag_16_disprop"),
        Disproportionality("ag_20_dem", "ag_20_rep", result_name="ag_20_disprop"),
        Disproportionality("aud_16_dem", "aud_16_rep", result_name="aud_16_disprop"),
        Disproportionality("aud_20_dem", "aud_20_rep", result_name="aud_20_disprop"),
        Disproportionality("gov_18_dem", "gov_18_rep", result_name="gov_18_disprop"),
        Disproportionality("pres_16_dem", "pres_16_rep", result_name="pres_16_disprop"),
        Disproportionality("pres_20_dem", "pres_20_rep", result_name="pres_20_disprop"),
    )

    out_dir = stats_dir / bendl_file.stem

    evaluator.evaluate_stream(bendl_file, out_dir, progress=True, update=True)


def main():
    chain_dir = ROOT_DIR / "chain_outputs"
    stats_dir = ROOT_DIR / "stats"

    stats_dir.mkdir(exist_ok=True, parents=True)

    for bendl_file in chain_dir.glob("VANILLA_PA*.bendl"):
        print(f"Processing {bendl_file.name}...")
        collect_results(bendl_file, stats_dir)


if __name__ == "__main__":
    main()
