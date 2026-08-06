# Democracy Batsignal

Democracy Batsignal creates a ready-to-run redistricting project with:

- a [uv](https://docs.astral.sh/uv/)-managed Python environment;
- [GerryChain](https://gerrychain.readthedocs.io/en/latest/) and
  [GerryTools](https://gerrytools.readthedocs.io/en/latest/);
- [binary-ensemble](https://binary-ensemble.readthedocs.io/en/latest/) for BENDL recordings;
- optional [RustReCom](https://github.com/mggg/rustrecom) installation; and
- Pennsylvania examples covering chain generation, optimization, scoring, and figures.

You only need the installer for your platform. You do not need to clone this repository.

## Download and install

Run the installer from the directory where the new project folder should be created.

### macOS or Linux

```bash
curl -LO https://raw.githubusercontent.com/mggg/democracy-batsignal/main/democracy-batsignal.sh
bash democracy-batsignal.sh
```

You can also download [`democracy-batsignal.sh`](democracy-batsignal.sh) in a browser and run it
with `bash democracy-batsignal.sh`.

### Windows

In PowerShell:

```powershell
Invoke-WebRequest `
    "https://raw.githubusercontent.com/mggg/democracy-batsignal/main/democracy-batsignal.ps1" `
    -OutFile democracy-batsignal.ps1
powershell -ExecutionPolicy Bypass -File .\democracy-batsignal.ps1
```

You can also download [`democracy-batsignal.ps1`](democracy-batsignal.ps1) in a browser. With
PowerShell 7, `pwsh -File .\democracy-batsignal.ps1` also works.

The installer asks for:

1. a name for the new project directory;
2. whether to install RustReCom; and
3. a Python version from 3.11 through 3.14.

It offers to install missing prerequisites such as uv, Rust/Cargo, and the Windows build tools.
The chosen Python is installed and managed by uv, so it does not replace the system Python. The
installer then creates the project, resolves its Python dependencies, and downloads the pinned
Pennsylvania example geometry.

RustReCom supplies the fast ordinary-chain and objective-guided examples. The Python GerryChain
examples work without it. A newly installed `uv`, `cargo`, or `rustrecom` command becomes available
in new terminals after the relevant installation directory is added to `PATH`.

## Verify the installation

Change into the generated project and check the Python environment:

```bash
cd my_project
uv run python --version
uv run python -c "import gerrychain, gerrytools, binary_ensemble; print('Python tools ready')"
```

If you installed RustReCom, also run:

```bash
rustrecom --version
```

## Ten-minute chain quickstart

The generated project includes a Python-first
[quickstart](development/template_project/QUICKSTART.md) and one editable experiment file. Its
default settings run a 100-step Pennsylvania GerryChain example:

```bash
uv run run_chains.py
```

To use another graph, edit the `Experiment settings` block near the top of `run_chains.py`. It
contains the graph path, starting-plan and population columns, random seeds, step count, population
tolerance, engine, experiment tag, and maximum simultaneous chains.

Change `ENGINE` to `"gerrychain"` for the modifiable Python workflow, `"rustrecom-chain"` for
ordinary RustReCom chains, or `"rustrecom-tilted"` with an objective file for objective-guided
search. The same Python file and command work in Bash and PowerShell.

Successful chains produce self-contained `.bendl` recordings in `chain_outputs/` and one log per
seed in `chain_logs/`. A BENDL file contains the assignment stream, graph, and run metadata, so
downstream scoring does not need a separate graph path.

The shell and PowerShell files under `pipeline_scripts/chain_runners/` are direct RustReCom command
references. `run_chains.py` adds Python-based seed lists, concurrency, logs, and failure handling
around the same commands.

For an interactive introduction, open
`notebooks/gerrychain_cut_edges_walkthrough.ipynb` in the generated project. It loads
Gerrymandria, builds a GerryChain ReCom proposal, runs 10,000 steps, and displays the cut-edge
histogram. The generated README also maps each common experiment change to the file that owns it.

## From chains to figures

The supplied Pennsylvania evaluator and ensemble figures process direct RustReCom reference outputs
whose filenames begin with `VANILLA_PA`.

1. Create one or more ordinary RustReCom recordings.

   On macOS or Linux:

   ```bash
   bash pipeline_scripts/chain_runners/pa_example_script_vanilla.sh
   ```

   In PowerShell:

   ```powershell
   .\pipeline_scripts\chain_runners\pa_example_script_vanilla.ps1
   ```

2. Score those recordings:

   ```bash
   uv run pipeline_scripts/metrics/collect_data_vanilla_pa.py --max-workers 1
   ```

3. Generate the ensemble figures:

   ```bash
   uv run pipeline_scripts/figure_generators/cut_edges_histogram.py
   uv run pipeline_scripts/figure_generators/disprop_scatter.py
   uv run pipeline_scripts/figure_generators/reock_boxplot.py
   ```

4. Generate maps of the starting plan and example alternate plan:

   ```bash
   uv run pipeline_scripts/figure_generators/base_plan_figures.py
   ```

Statistics are written under `stats/<recording-name>/`, and figures are written under `figures/`.
The Python commands are the same in PowerShell. `--max-workers` controls how many independent BENDL
files are scored concurrently.

`run_chains.py` uses a shared cross-platform Python runner for GerryChain and both RustReCom modes.
It limits concurrent child processes, writes one log per seed, stops children when interrupted,
and exits nonzero if any seed fails. CPU, memory, and I/O requirements scale with `MAX_WORKERS`.

The runner displays one aggregate spinner; child output and progress indicators stay in per-seed
log files. A representative 10,000-step chain provides an estimate of runtime and output size for a
particular graph, constraint set, and objective.

`EXPERIMENT_TAG` names the experiment. Parallel-runner filenames use this form:

```text
<ENGINE>_<OUTPUT_PREFIX>__STEPS_<steps>__RNGSEED_<seed>__TOL_<tol>__SEEDPLN__<plan_name>__TAG_<tag>__DATE_<date>.bendl
```

The runner adds `PY_` for GerryChain or `RUST_` for either RustReCom mode. Logs use the same stem
with `.log`; tilted RustReCom score files add `_scores.csv`.

## What the tools do

- **uv** installs the selected Python, resolves dependencies from `pyproject.toml`, and runs Python
  commands inside the project environment. Prefix project commands with `uv run`.
- **[GerryChain](https://gerrychain.readthedocs.io/en/latest/)** provides the Python graph,
  partition, updater, and ReCom APIs. The included
  `pipeline_scripts/chain_runners/example_cli.py` shows how to record a GerryChain chain directly
  to BENDL.
- **RustReCom** is a fast command-line ReCom implementation. `rustrecom chain` samples an ordinary
  chain, `rustrecom tilted` probabilistically favors better objective scores, and
  `rustrecom short-bursts` repeatedly continues from the best plan found in each burst.
- **[binary-ensemble](https://binary-ensemble.readthedocs.io/en/latest/)** reads and writes
  BEN/BENDL assignment streams through its Python API.
- **[GerryTools](https://gerrytools.readthedocs.io/en/latest/)** scores recorded ensembles and
  creates the example plots.

The generated project's
[`README.md`](development/template_project/README.md) is the complete user guide. It explains the
required graph attributes, every important RustReCom option and variant, the supplied objectives,
BENDL inspection, GerryChain recording, scoring, plotting, adaptation to another state, and common
failure modes.

## Generated project layout

```text
my_project/
├── QUICKSTART.md               # ~10-minute guide for adapting the project
├── run_chains.py               # edit one settings block, then run this file
├── pyproject.toml
├── JSON_dualgraphs/            # example dual graphs
├── data/                       # geometry and alternate-plan data
├── notebooks/                  # interactive GerryChain walkthrough
├── pipeline_scripts/
│   ├── run_parallel_chains.py  # bounded cross-platform chain runner
│   ├── chain_runners/          # GerryChain and direct RustReCom references
│   ├── metrics/                # GerryTools evaluation
│   └── figure_generators/      # maps and ensemble plots
├── chain_outputs/              # BENDL recordings
├── chain_logs/                 # redirected child-process output
├── stats/                      # EnsembleEvalResult data (Parquet files)
└── figures/                    # generated PNG files
```

Projects created by the Bash installer receive direct `.sh` RustReCom references. Projects created
by the PowerShell installer receive equivalent `.ps1` references. The same Python entry point runs
on macOS, Linux, and Windows.

## Reproducibility

Chain runs use explicit RNG seeds. BENDL recordings contain the graph, assignment stream, and run
metadata. Reproduction also depends on the relevant software versions and other inputs remaining
fixed.

## Repository development

The only user-facing files at the repository root are this README and the two downloadable
installers. Maintainer sources and tests live under `development/`.

`democracy-batsignal.sh` and `democracy-batsignal.ps1` are generated files. Edit
`development/template_project/` or `development/installer_src/`, then regenerate them with:

```bash
python3 development/clean_notebooks.py
python3 development/generate_installers.py
python3 development/generate_installers.py --check
```

The notebook cleaner removes outputs, execution counts, cell IDs, and notebook and cell metadata.
Use `python3 development/clean_notebooks.py --check` in automated checks. The installer generator
also cleans notebook payloads in memory, so local execution state cannot enter either downloadable
installer even when the source notebook has not been cleaned first.

The generator includes only the platform-appropriate helper scripts, skips `uv.lock` and local
caches, and resolves dependencies using the Python version selected during installation.

Run the complete clean-container check with:

```bash
./development/test_generated_projects.sh
```

It installs both generated project variants independently and checks the GerryChain, ordinary
RustReCom, tilted RustReCom, short-bursts, notebook, scoring, and plotting workflows. It also runs
the template's lint and type checks and verifies the expected artifacts.
