import json
import math
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from itertools import cycle
from pathlib import Path
from re import fullmatch
from threading import Event, Lock
from typing import Literal, cast

import click

ROOT_DIR = Path(__file__).resolve().parents[1]
ACTIVE_PROCESSES: set[subprocess.Popen[bytes]] = set()
ACTIVE_PROCESSES_LOCK = Lock()
STOP_REQUESTED = Event()


RECOM_ENGINES = ("gerrychain", "rustrecom-chain", "rustrecom-tilted")
RECOM_VARIANTS = (
    "cut-edges-mst",
    "cut-edges-ust",
    "district-pairs-mst",
    "district-pairs-ust",
)

ReComEngine = Literal["gerrychain", "rustrecom-chain", "rustrecom-tilted"]
ReComVariant = Literal[
    "cut-edges-mst",
    "cut-edges-ust",
    "district-pairs-mst",
    "district-pairs-ust",
]


@dataclass(frozen=True, slots=True)
class ChainSettings:
    """Describes the inputs and destinations for one independent ReCom chain.

    Attributes:
        engine (ReComEngine): Program used to generate the chain. ``gerrychain`` runs the
            Python example, ``rustrecom-chain`` samples an ordinary RustReCom chain, and
            ``rustrecom-tilted`` applies a RustReCom optimization objective.
        graph_path (Path): Dual graph supplied to the selected engine.
        output_prefix (str): Analysis name included in output filenames, such as
            ``VANILLA_PA`` or ``GINGLES_PARTIAL_PA``.
        starting_plan (str): Node attribute containing each unit's starting district.
        pop_col (str): Node attribute containing the population used for balance constraints.
        rng_seed (int): Seed that makes this chain's random proposal stream reproducible.
        total_steps (int): Number of chain positions requested from the selected engine.
        population_tolerance (float): Maximum fractional population deviation allowed by ReCom.
            For example, ``0.01`` permits one percent deviation from ideal population.
        recom_variant (ReComVariant): Rule used to select adjacent districts and sample a
            spanning tree for each ReCom proposal.
        run_date (str): Experiment date written into the output filenames in ``YYYY-MM-DD`` form.
        output_dir (Path): Directory for BENDL recordings and tilted score files.
        log_dir (Path): Directory for one child-process log per random seed.
        tag (str): Short experiment name included in every output filename.
        objective_file (Path | None): RustReCom objective JSON used for a tilted chain.
        maximize (bool): Whether a tilted chain searches for larger or smaller objective values.
        region_weights (dict[str, float] | None): Surcharges for cutting named region columns.
            Region-aware proposals require an MST ReCom variant.
    """

    engine: ReComEngine
    graph_path: Path
    output_prefix: str
    starting_plan: str
    pop_col: str
    rng_seed: int
    total_steps: int
    population_tolerance: float
    recom_variant: ReComVariant
    run_date: str
    output_dir: Path
    log_dir: Path
    tag: str = "baseline"
    objective_file: Path | None = None
    maximize: bool = True
    region_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.engine not in RECOM_ENGINES:
            raise ValueError(f"Unknown engine: {self.engine}.")
        if self.recom_variant not in RECOM_VARIANTS:
            raise ValueError(f"Unknown ReCom variant: {self.recom_variant}.")
        if self.engine == "rustrecom-tilted" and self.objective_file is None:
            raise ValueError("--objective-file is required for rustrecom-tilted.")
        if self.engine != "rustrecom-tilted" and self.objective_file is not None:
            raise ValueError("--objective-file is only valid for rustrecom-tilted.")
        if not self.graph_path.exists():
            raise ValueError(f"Graph path does not exist: {self.graph_path}")
        if self.objective_file is not None and not self.objective_file.exists():
            raise ValueError(f"Objective file does not exist: {self.objective_file}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.region_weights is not None and self.recom_variant not in (
            "cut-edges-mst",
            "district-pairs-mst",
        ):
            raise ValueError(
                "Region weights are only applicable for 'mst' ReCom variants. "
                f"Provided variant: {self.recom_variant}"
            )


def filename_token(
    _context: click.Context,
    parameter: click.Parameter,
    value: str,
) -> str:
    """Validates a user-supplied value before placing it in an output filename.

    Args:
        _context (click.Context): Click command context. It is unused by this validator.
        parameter (click.Parameter): Option being validated, used in any error message.
        value (str): Proposed filename component.

    Returns:
        str: The unchanged value when it contains only path-safe characters.

    Raises:
        click.BadParameter: If the value could create a path or ambiguous filename component.
    """
    if not fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise click.BadParameter(
            "use only letters, numbers, periods, underscores, and hyphens",
            param=parameter,
        )
    return value


def parse_region_weights(
    _context: click.Context,  # Click supplies context to every option callback.
    parameter: click.Parameter,
    value: str | None,
) -> dict[str, float] | None:
    """Parses region-aware ReCom surcharges from a command-line JSON object.

    A weight increases the cost of a spanning-tree edge that crosses the named region column.
    For example, ``{"COUNTY": 1.0}`` discourages county splits without prohibiting them.

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


def run_chain(settings: ChainSettings) -> tuple[int, int, Path]:
    """Runs one chain as a child process and records its console output.

    The function translates shared experiment settings into either the GerryChain example CLI or
    RustReCom CLI. Each chain writes to its own BENDL and log files, which prevents output from
    concurrent runs from being interleaved.

    Args:
        settings (ChainSettings): Complete settings for one independently seeded chain.

    Returns:
        tuple[int, int, Path]: Random seed, child-process exit code, and log path. An exit code of
            zero means the chain completed successfully.
    """
    engine_prefix = "PY" if settings.engine == "gerrychain" else "RUST"
    tolerance_label = f"{settings.population_tolerance:g}".replace(".", "p")
    filename_stem = (
        f"{engine_prefix}_{settings.output_prefix}"
        f"__STEPS_{settings.total_steps}"
        f"__RNGSEED_{settings.rng_seed}"
        f"__TOL_{tolerance_label}"
        f"__SEEDPLN__{settings.starting_plan}"
        f"__TAG_{settings.tag}"
        f"__DATE_{settings.run_date}"
    )
    output_path = settings.output_dir / f"{filename_stem}.bendl"
    log_path = settings.log_dir / f"{filename_stem}.log"
    if settings.engine == "gerrychain":
        command = [
            sys.executable,
            str(ROOT_DIR / "pipeline_scripts" / "chain_runners" / "example_cli.py"),
            "--graph-path",
            str(settings.graph_path),
            "--output-path",
            str(output_path),
            "--starting-plan",
            settings.starting_plan,
            "--pop-col",
            settings.pop_col,
            "--rng-seed",
            str(settings.rng_seed),
            "--population-tolerance",
            str(settings.population_tolerance),
            "--total-steps",
            str(settings.total_steps),
            "--recom-variant",
            settings.recom_variant,
        ]
    else:
        rustrecom_command = "chain" if settings.engine == "rustrecom-chain" else "tilted"
        command = [
            "rustrecom",
            rustrecom_command,
            "--graph-json",
            str(settings.graph_path),
            "--output-file",
            str(output_path),
            "--assignment-col",
            settings.starting_plan,
            "--pop-col",
            settings.pop_col,
            "--rng-seed",
            str(settings.rng_seed),
            "--tol",
            str(settings.population_tolerance),
            "--n-steps",
            str(settings.total_steps),
            "--variant",
            settings.recom_variant,
            "--writer",
            "bendl",
            "--overwrite-output",
        ]
        if settings.engine == "rustrecom-tilted" and settings.objective_file is not None:
            scores_path = output_path.with_name(f"{output_path.stem}_scores.csv")
            command.extend(
                [
                    "--objective",
                    str(settings.objective_file),
                    "--maximize",
                    str(settings.maximize).lower(),
                    "--scores-output-file",
                    str(scores_path),
                ]
            )

    if settings.region_weights is not None:
        command.extend(["--region-weights", json.dumps(settings.region_weights, sort_keys=True)])

    with log_path.open("wb") as log_file:
        log_file.write(
            f"Starting {settings.engine} chain with RNG seed {settings.rng_seed}.\n".encode()
        )
        log_file.flush()
        try:
            # Registration shares a lock with interruption cleanup, so cleanup cannot miss a
            # process that has started but is not yet tracked.
            with ACTIVE_PROCESSES_LOCK:
                if STOP_REQUESTED.is_set():
                    log_file.write(b"Cancelled before launch.\n")
                    return settings.rng_seed, 130, log_path
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
                ACTIVE_PROCESSES.add(process)
        except OSError as error:
            log_file.write(f"Could not start chain: {error}\n".encode())
            return settings.rng_seed, 127, log_path

        try:
            exit_code = process.wait()
            log_file.write(f"\nChild process exited with code {exit_code}.\n".encode())
            return settings.rng_seed, exit_code, log_path
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.discard(process)


def stop_active_processes() -> None:
    """Stops child chains after an interruption, then kills any that do not exit promptly."""
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_chains(
    settings_by_seed: Sequence[ChainSettings],
    max_workers: int = 1,
) -> None:
    """Runs independently seeded chains with a limit on simultaneous child processes.

    This is the main Python interface for batch runs. Each ``ChainSettings`` value describes one
    chain, and ``max_workers`` controls how many of those chains may run at the same time. Child
    output is kept in separate log files so messages from concurrent chains do not overlap.

    Args:
        settings_by_seed (Sequence[ChainSettings]): Settings for each independently seeded chain.
            Random seeds must be unique within the batch.
        max_workers (int): Maximum number of child processes to run simultaneously. CPU, memory,
            and I/O requirements scale with this value.

    Raises:
        ValueError: If the batch is empty, contains duplicate seeds, or has fewer than one worker.
        RuntimeError: If one or more child chains exit unsuccessfully.
        KeyboardInterrupt: If the user interrupts the batch. Active child chains are stopped
            before the exception is raised.
    """
    settings = list(settings_by_seed)
    if not settings:
        raise ValueError("At least one chain must be configured.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    rng_seeds = [chain.rng_seed for chain in settings]
    if len(set(rng_seeds)) != len(rng_seeds):
        raise ValueError("Each chain must use a unique RNG seed.")

    STOP_REQUESTED.clear()
    failures: list[tuple[int, int, Path]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(settings))) as executor:
        futures = []
        try:
            futures = [executor.submit(run_chain, chain) for chain in settings]
            pending = set(futures)
            completed = 0
            spinner = cycle("|/-\\")
            status_width = 0
            engine_names = ", ".join(sorted({chain.engine for chain in settings}))
            click.echo(f"Running {len(futures)} chain(s) with {engine_names}...", err=True)
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                if not done and sys.stderr.isatty():
                    status = (
                        f"{next(spinner)} Running chains ({completed}/{len(futures)} completed)"
                    )
                    status_width = max(status_width, len(status))
                    click.echo(f"\r{status:<{status_width}}", nl=False, err=True)
                    continue

                if sys.stderr.isatty() and status_width:
                    click.echo(f"\r{'':<{status_width}}\r", nl=False, err=True)
                for future in done:
                    seed, exit_code, log_path = future.result()
                    completed += 1
                    if exit_code:
                        failures.append((seed, exit_code, log_path))
                        click.echo(f"Seed {seed} failed; see {log_path}.", err=True)
                    else:
                        click.echo(f"Seed {seed} completed; log: {log_path}.", err=True)
        except KeyboardInterrupt:
            STOP_REQUESTED.set()
            for future in futures:
                future.cancel()
            stop_active_processes()
            raise

    if failures:
        failed_seeds = ", ".join(str(seed) for seed, _, _ in failures)
        raise RuntimeError(f"{len(failures)} chain(s) failed (seeds: {failed_seeds}).")


@click.command()
@click.option(
    "--engine",
    type=click.Choice(RECOM_ENGINES),
    default="gerrychain",
    show_default=True,
    help="Program used to generate each chain.",
)
@click.option(
    "--objective-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="RustReCom objective JSON; required for rustrecom-tilted.",
)
@click.option(
    "--maximize/--minimize",
    default=True,
    show_default=True,
    help="Direction for a tilted objective.",
)
@click.option(
    "--graph-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-prefix",
    required=True,
    callback=filename_token,
    help="Analysis name after the automatic PY_ or RUST_ prefix.",
)
@click.option(
    "--starting-plan",
    required=True,
    callback=filename_token,
    help="Starting-plan node attribute.",
)
@click.option("--pop-col", required=True, help="Population node attribute.")
@click.option("--rng-seed", type=int, multiple=True, required=True, help="Repeat for each chain.")
@click.option("--total-steps", type=click.IntRange(min=1), default=1_000, show_default=True)
@click.option(
    "--tag",
    required=True,
    callback=filename_token,
    help="Short name for the experiment, included in every output filename.",
)
@click.option(
    "--run-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=date.today().isoformat(),  # noqa: DTZ011 - use the experimenter's local date.
    show_default=True,
    help="Experiment date included in every output filename.",
)
@click.option(
    "--population-tolerance",
    type=click.FloatRange(min=0, max=1),
    default=0.01,
    show_default=True,
)
@click.option(
    "--recom-variant",
    type=click.Choice(RECOM_VARIANTS),
    default="district-pairs-mst",
    show_default=True,
)
@click.option(
    "--max-workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum chains to run at once.",
)
@click.option(
    "--region-weights",
    callback=parse_region_weights,
    help="Region-column surcharges as JSON, for example '{\"COUNTY\": 1.0}'. MST only.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=ROOT_DIR / "chain_outputs",
    show_default=True,
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=ROOT_DIR / "chain_logs",
    show_default=True,
)
def main(
    engine: str,
    graph_path: Path,
    output_prefix: str,
    starting_plan: str,
    pop_col: str,
    rng_seed: tuple[int, ...],
    total_steps: int,
    tag: str,
    run_date: datetime,
    population_tolerance: float,
    recom_variant: str,
    max_workers: int,
    region_weights: dict[str, float] | None,
    objective_file: Path | None,
    maximize: bool,
    output_dir: Path,
    log_dir: Path,
) -> None:
    """Runs the Python batch interface from command-line options.

    Args:
        engine (str): ``gerrychain``, ``rustrecom-chain``, or ``rustrecom-tilted``.
        graph_path (Path): Dual graph read by every chain.
        output_prefix (str): Analysis name placed after the automatic ``PY_`` or ``RUST_``
            filename prefix.
        starting_plan (str): Node attribute containing the initial district assignment.
        pop_col (str): Node attribute containing the population used by ReCom.
        rng_seed (tuple[int, ...]): Unique random seed for each independent chain.
        total_steps (int): Number of chain positions requested for every seed.
        tag (str): Experiment name included in each BENDL and log filename.
        run_date (datetime): Experiment date included in each output filename.
        population_tolerance (float): Maximum fractional deviation from ideal population.
        recom_variant (str): District-pair and spanning-tree sampling rule.
        max_workers (int): Maximum number of independently seeded child processes to run at once.
        region_weights (dict[str, float] | None): Optional region-column split surcharges.
        objective_file (Path | None): Objective JSON required for tilted RustReCom.
        maximize (bool): Whether tilted RustReCom searches for larger objective values.
        output_dir (Path): Destination for BENDL recordings and tilted score files.
        log_dir (Path): Destination for one child-process log per seed.

    Raises:
        click.ClickException: If settings are incompatible or one or more child chains fail.
        click.Abort: If the user interrupts the batch.
    """
    try:
        settings_by_seed = [
            ChainSettings(
                engine=cast(ReComEngine, engine),
                graph_path=graph_path,
                output_prefix=output_prefix,
                starting_plan=starting_plan,
                pop_col=pop_col,
                rng_seed=seed,
                total_steps=total_steps,
                population_tolerance=population_tolerance,
                recom_variant=cast(ReComVariant, recom_variant),
                run_date=run_date.date().isoformat(),
                output_dir=output_dir,
                log_dir=log_dir,
                tag=tag,
                objective_file=objective_file,
                maximize=maximize,
                region_weights=region_weights,
            )
            for seed in rng_seed
        ]
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    try:
        run_chains(settings_by_seed, max_workers=max_workers)
    except (ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    except KeyboardInterrupt as error:
        raise click.Abort() from error


if __name__ == "__main__":
    main()
