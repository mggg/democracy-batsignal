import json
import math
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
from itertools import cycle
from pathlib import Path
from re import fullmatch
from threading import Event, Lock
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parents[2]
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
        log_dir (Path): Directory for one child-process log per configured run.
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
        for name, value in (
            ("output_prefix", self.output_prefix),
            ("starting_plan", self.starting_plan),
            ("tag", self.tag),
            ("run_date", self.run_date),
        ):
            try:
                validate_filename_token(value)
            except ValueError as error:
                raise ValueError(f"Invalid {name}: {error}") from error
        if self.engine not in RECOM_ENGINES:
            raise ValueError(f"Unknown engine: {self.engine}.")
        if self.recom_variant not in RECOM_VARIANTS:
            raise ValueError(f"Unknown ReCom variant: {self.recom_variant}.")
        if (
            isinstance(self.total_steps, bool)
            or not isinstance(self.total_steps, int)
            or self.total_steps < 1
        ):
            raise ValueError("total_steps must be a positive integer.")
        if (
            isinstance(self.population_tolerance, bool)
            or not isinstance(self.population_tolerance, (int, float))
            or not math.isfinite(self.population_tolerance)
            or not 0 <= self.population_tolerance <= 1
        ):
            raise ValueError("population_tolerance must be a finite number from 0 through 1.")
        try:
            parsed_date = date.fromisoformat(self.run_date)
        except ValueError as error:
            raise ValueError("run_date must use YYYY-MM-DD format.") from error
        if parsed_date.isoformat() != self.run_date:
            raise ValueError("run_date must use YYYY-MM-DD format.")
        if self.engine == "rustrecom-tilted" and self.objective_file is None:
            raise ValueError("objective_file is required for rustrecom-tilted.")
        if self.engine != "rustrecom-tilted" and self.objective_file is not None:
            raise ValueError("objective_file is only valid for rustrecom-tilted.")
        if not self.graph_path.is_file():
            raise ValueError(f"graph_path must be an existing file: {self.graph_path}")
        if self.objective_file is not None and not self.objective_file.is_file():
            raise ValueError(f"objective_file must be an existing file: {self.objective_file}")
        for name, path in (("output_dir", self.output_dir), ("log_dir", self.log_dir)):
            if path.exists() and not path.is_dir():
                raise ValueError(f"{name} must be a directory: {path}")
            path.mkdir(parents=True, exist_ok=True)

        if self.region_weights is not None:
            if not isinstance(self.region_weights, dict):
                raise ValueError("region_weights must be a dictionary.")
            for column, weight in self.region_weights.items():
                if not isinstance(column, str) or not column:
                    raise ValueError("region_weights column names must be non-empty strings.")
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                ):
                    raise ValueError("region_weights values must be finite numbers.")

        if self.region_weights and self.recom_variant not in (
            "cut-edges-mst",
            "district-pairs-mst",
        ):
            raise ValueError(
                "Region weights are only applicable for 'mst' ReCom variants. "
                f"Provided variant: {self.recom_variant}"
            )


def validate_filename_token(value: str) -> None:
    """Validates one component of a generated output filename.

    Args:
        value (str): Proposed filename component.

    Raises:
        ValueError: If the value is not path-safe or contains the ``__`` field delimiter.
    """
    if not fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("use only letters, numbers, periods, underscores, and hyphens")
    if "__" in value:
        raise ValueError("must not contain the reserved '__' filename delimiter")


def run_chain(settings: ChainSettings) -> tuple[int, str, int, Path]:
    """Runs one chain as a child process and records its console output.

    The function translates shared experiment settings into either the GerryChain example CLI or
    RustReCom CLI. Each chain writes to its own BENDL and log files, which prevents output from
    concurrent runs from being interleaved.

    Args:
        settings (ChainSettings): Complete settings for one chain.

    Returns:
        tuple[int, str, int, Path]: Random seed, starting-plan column, child-process exit code, and
            log path. An exit code of zero means the chain completed successfully.
    """
    match settings.engine:
        case "gerrychain":
            engine_prefix = "PY"
        case "rustrecom-chain":
            engine_prefix = "RUST_CHAIN"
        case "rustrecom-tilted":
            engine_prefix = "RUST_TILTED"
        case _:
            raise ValueError(f"Unknown engine: {settings.engine}")

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
            str(ROOT_DIR / "pipeline_scripts" / "chain_runners" / "gerrychain_cli.py"),
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

    if settings.region_weights:
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
                    return settings.rng_seed, settings.starting_plan, 130, log_path
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
                ACTIVE_PROCESSES.add(process)
        except OSError as error:
            log_file.write(f"Could not start chain: {error}\n".encode())
            return settings.rng_seed, settings.starting_plan, 127, log_path

        try:
            exit_code = process.wait()
            log_file.write(f"\nChild process exited with code {exit_code}.\n".encode())
            return settings.rng_seed, settings.starting_plan, exit_code, log_path
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
    settings_by_run: Sequence[ChainSettings],
    max_workers: int = 1,
) -> None:
    """Runs configured chains with a limit on simultaneous child processes.

    This is the main Python interface for batch runs. Each ``ChainSettings`` value describes one
    chain, and ``max_workers`` controls how many of those chains may run at the same time. Child
    output is kept in separate log files so messages from concurrent chains do not overlap.

    Args:
        settings_by_run (Sequence[ChainSettings]): Settings for each chain. Every random-seed and
            starting-plan combination must be unique within the batch.
        max_workers (int): Maximum number of child processes to run simultaneously. CPU, memory,
            and I/O requirements scale with this value.

    Raises:
        ValueError: If the batch is empty, contains duplicate run identifiers, or ``max_workers``
            is not a positive integer.
        RuntimeError: If one or more child chains exit unsuccessfully.
        KeyboardInterrupt: If the user interrupts the batch. Active child chains are stopped
            before the exception is raised.
    """
    settings = list(settings_by_run)
    if not settings:
        raise ValueError("At least one chain must be configured.")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    run_ids = [(chain.rng_seed, chain.starting_plan) for chain in settings]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Each RNG-seed and starting-plan combination must be unique.")

    STOP_REQUESTED.clear()
    failures: list[tuple[int, str, int, Path]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(settings))) as executor:
        futures = []
        try:
            futures = [executor.submit(run_chain, chain) for chain in settings]
            pending = set(futures)
            completed = 0
            spinner = cycle("|/-\\")
            status_width = 0
            engine_names = ", ".join(sorted({chain.engine for chain in settings}))
            print(
                f"Running {len(futures)} chain(s) with {engine_names}...",
                file=sys.stderr,
                flush=True,
            )
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                if not done and sys.stderr.isatty():
                    status = (
                        f"{next(spinner)} Running chains ({completed}/{len(futures)} completed)"
                    )
                    status_width = max(status_width, len(status))
                    print(
                        f"\r{status:<{status_width}}",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                if sys.stderr.isatty() and status_width:
                    print(
                        f"\r{'':<{status_width}}\r",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                for future in done:
                    seed, starting_plan, exit_code, log_path = future.result()
                    completed += 1
                    if exit_code:
                        failures.append((seed, starting_plan, exit_code, log_path))
                        print(
                            f"Seed {seed}, plan {starting_plan!r} failed; see {log_path}.",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(
                            f"Seed {seed}, plan {starting_plan!r} completed; log: {log_path}.",
                            file=sys.stderr,
                            flush=True,
                        )
        except KeyboardInterrupt:
            STOP_REQUESTED.set()
            for future in futures:
                future.cancel()
            stop_active_processes()
            raise

    if failures:
        failed_runs = ", ".join(
            f"seed {seed} / plan {starting_plan}" for seed, starting_plan, _, _ in failures
        )
        raise RuntimeError(f"{len(failures)} chain(s) failed ({failed_runs}).")
