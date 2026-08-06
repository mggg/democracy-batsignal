from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import geopandas as gpd
from binary_ensemble import BendlDecoder
from gerrytools.scoring import (
    AggregateSeats,
    CutEdges,
    Disproportionality,
    PlanEvaluator,
    PolsbyPopper,
    Reock,
    Seats,
    Tally,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def collect_results(
    bendl_file: Path,
    stats_dir: Path,
    batch_size: int,
    show_progress: bool,
) -> Path:
    """Evaluate one BENDL file and return its statistics directory.

    Args:
        bendl_file: Recorded ensemble to evaluate.
        stats_dir: Parent directory for evaluation results.
        batch_size: Plans scored together in each streaming batch.
        show_progress: Whether to display GerryTools progress.

    Returns:
        Directory containing the streamed metric results.
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

    output_dir = stats_dir / bendl_file.stem
    evaluator.evaluate_stream(
        bendl_file,
        output_dir,
        batch_size=batch_size,
        progress=show_progress,
        update=True,
    )
    return output_dir


@click.command()
@click.option(
    "--input-glob",
    default="VANILLA_PA*.bendl",
    show_default=True,
    help="Filename pattern to select from chain_outputs.",
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum BENDL files to evaluate at once.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Plans scored together in each GerryTools streaming batch.",
)
def main(input_glob: str, max_workers: int, batch_size: int) -> None:
    """Evaluate matching Pennsylvania chains and write reusable metric results."""
    chain_dir = ROOT_DIR / "chain_outputs"
    stats_dir = ROOT_DIR / "stats"
    bendl_files = sorted(chain_dir.glob(input_glob))
    if not bendl_files:
        raise click.ClickException(f"No files in {chain_dir} match {input_glob!r}.")

    stats_dir.mkdir(exist_ok=True, parents=True)
    if max_workers == 1:
        for bendl_file in bendl_files:
            click.echo(f"Processing {bendl_file.name}...")
            collect_results(bendl_file, stats_dir, batch_size, show_progress=True)
        return

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(bendl_files))) as executor:
        future_files = {
            executor.submit(collect_results, path, stats_dir, batch_size, False): path
            for path in bendl_files
        }
        for future in as_completed(future_files):
            bendl_file = future_files[future]
            try:
                output_dir = future.result()
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                failures.append(f"{bendl_file.name}: {error}")
            else:
                click.echo(f"Finished {bendl_file.name}: {output_dir}")

    if failures:
        details = "\n".join(f"  {failure}" for failure in failures)
        raise click.ClickException(f"Evaluation failed:\n{details}")


if __name__ == "__main__":
    main()
