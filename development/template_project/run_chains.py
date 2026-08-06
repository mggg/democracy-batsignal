"""Configure and run an independently seeded batch of ReCom chains."""

from datetime import date
from pathlib import Path

from pipeline_scripts.run_parallel_chains import (
    ChainSettings,
    ReComEngine,
    ReComVariant,
    run_chains,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OBJECTIVES_DIR = PROJECT_ROOT / "pipeline_scripts" / "chain_runners" / "rustrecom_objectives"

# Experiment settings: edit this block, then run `uv run run_chains.py`.
ENGINE: ReComEngine = "gerrychain"
GRAPH_PATH = PROJECT_ROOT / "JSON_dualgraphs" / "pa_dualgraph.json"
OUTPUT_PREFIX = "VANILLA_PA"
STARTING_PLAN = "seed_plan"
POPULATION_COLUMN = "total_pop_20"
RNG_SEEDS = (42,)
TOTAL_STEPS = 100
POPULATION_TOLERANCE = 0.01
RECOM_VARIANT: ReComVariant = "district-pairs-mst"
EXPERIMENT_TAG = "quickstart"
RUN_DATE = date.today().isoformat()  # noqa: DTZ011 - use the experimenter's local date.
MAX_WORKERS = 1

# Tilted RustReCom requires an objective file. Ordinary GerryChain and RustReCom runs use None.
OBJECTIVE_FILE: Path | None = None
MAXIMIZE_OBJECTIVE = True

# Region weights are optional and require an MST variant. Example: {"county_id": 1.0}
REGION_WEIGHTS: dict[str, float] | None = None


def settings_for_seed(rng_seed: int) -> ChainSettings:
    """Builds the complete settings for one independently seeded chain.

    Args:
        rng_seed (int): Random seed that identifies and reproduces this chain.

    Returns:
        ChainSettings: Engine, graph, ReCom, and output settings for the chain.
    """
    return ChainSettings(
        engine=ENGINE,
        graph_path=GRAPH_PATH,
        output_prefix=OUTPUT_PREFIX,
        starting_plan=STARTING_PLAN,
        pop_col=POPULATION_COLUMN,
        rng_seed=rng_seed,
        total_steps=TOTAL_STEPS,
        population_tolerance=POPULATION_TOLERANCE,
        recom_variant=RECOM_VARIANT,
        run_date=RUN_DATE,
        output_dir=PROJECT_ROOT / "chain_outputs",
        log_dir=PROJECT_ROOT / "chain_logs",
        tag=EXPERIMENT_TAG,
        objective_file=OBJECTIVE_FILE,
        maximize=MAXIMIZE_OBJECTIVE,
        region_weights=REGION_WEIGHTS,
    )


def main() -> None:
    """Runs every configured seed and reports the location of each per-chain log."""
    chains = [settings_for_seed(seed) for seed in RNG_SEEDS]
    run_chains(chains, max_workers=MAX_WORKERS)


if __name__ == "__main__":
    main()
