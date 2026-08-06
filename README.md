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
installer then creates the project, resolves its Python dependencies, and downloads the checked
Pennsylvania example geometry.

RustReCom is optional. Install it if you want the fast Rust chain and optimization examples. The
Python GerryChain examples work without it. If a newly installed `uv`, `cargo`, or `rustrecom`
command is not immediately visible in another terminal, restart that terminal first.

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
rustrecom --help
```

## Run the Pennsylvania example

The quickest RustReCom example runs an ordinary ReCom chain from the `seed_plan` node attribute in
the included Pennsylvania dual graph. The shipped scripts default to two seeds and 100,000 steps,
so reduce the step count while testing.

On macOS or Linux, edit `n_steps` and `rng_seed` near the top of
`pipeline_scripts/pa_example_script_vanilla.sh`, then run:

```bash
bash pipeline_scripts/pa_example_script_vanilla.sh
```

On Windows, pass the values as PowerShell parameters:

```powershell
.\pipeline_scripts\pa_example_script_vanilla.ps1 -NSteps 100 -RngSeeds 42
```

The result is a self-contained `.bendl` recording in `chain_outputs/`. A BENDL file contains the
assignment stream, the graph, and run metadata, so downstream scoring does not need a separate
graph path.

To try objective-guided search, use the matching `pa_example_script_opt` script. It calls
`rustrecom tilted` with the supplied Gingles partial objective:

```bash
bash pipeline_scripts/pa_example_script_opt.sh
```

```powershell
.\pipeline_scripts\pa_example_script_opt.ps1 -NSteps 100 -RngSeeds 42
```

The optimization examples are demonstrations of search behavior, not legal conclusions about a
plan or district.

## From chains to figures

The included Pennsylvania analysis pipeline expects an ordinary RustReCom output whose filename
starts with `VANILLA_PA`.

1. Run `pa_example_script_vanilla` to create one or more recordings in `chain_outputs/`.
2. Score those recordings:

   ```bash
   uv run pipeline_scripts/metrics/collect_data_vanilla_pa.py
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
The Python commands are the same in PowerShell.

## What the tools do

- **uv** installs the selected Python, resolves dependencies from `pyproject.toml`, and runs Python
  commands inside the project environment. Prefix project commands with `uv run`.
- **[GerryChain](https://gerrychain.readthedocs.io/en/latest/)** provides the Python graph,
  partition, updater, and ReCom APIs. The included `example_cli.py` shows how to record a
  GerryChain chain directly to BENDL.
- **RustReCom** is a fast command-line ReCom implementation. `rustrecom chain` samples an ordinary
  chain, `rustrecom tilted` probabilistically favors better objective scores, and
  `rustrecom short-bursts` repeatedly continues from the best plan found in each burst.
- **[binary-ensemble](https://binary-ensemble.readthedocs.io/en/latest/)** reads and writes
  BEN/BENDL assignment streams. It is a Python package in version 2.0, not a separate `ben`
  command.
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
├── pyproject.toml
├── JSON_dualgraphs/          # example dual graphs
├── data/                     # geometry and alternate-plan data
├── pipeline_scripts/
│   ├── example_cli.py        # GerryChain RecordedChain example
│   ├── pa_example_script_*   # ordinary and tilted RustReCom examples
│   ├── rustrecom_objectives/ # reusable objective JSON files
│   ├── metrics/              # BENDL scoring pipeline
│   └── figure_generators/    # maps and ensemble plots
├── batch_example_python_cli_* # sequential and parallel GerryChain examples
├── chain_outputs/             # BENDL recordings
├── chain_logs/                # batch-run logs
├── stats/                     # GerryTools evaluation results
└── figures/                   # generated PNG files
```

Bash projects receive `.sh` helpers, and Windows projects receive equivalent `.ps1` helpers. The
Python files are identical on every platform.

## Reproducibility

Chain runs use explicit RNG seeds. Record the graph, starting-plan column, population column,
tolerance, seed, RustReCom variant, and objective settings with any reported result. A fixed seed
reproduces a run only when the relevant software versions and inputs also stay fixed.

## Repository development

The only user-facing files at the repository root are this README and the two downloadable
installers. Maintainer sources and tests live under `development/`.

`democracy-batsignal.sh` and `democracy-batsignal.ps1` are generated files. Edit
`development/template_project/` or `development/installer_src/`, then regenerate them with:

```bash
python3 development/generate_installers.py
python3 development/generate_installers.py --check
```

The generator includes only the platform-appropriate helper scripts, skips `uv.lock` and local
caches, and resolves dependencies using the Python version selected during installation.

Run the complete clean-container check with:

```bash
./development/test_generated_projects.sh
```

It installs both generated project variants independently, runs every Bash and PowerShell helper,
scores a Pennsylvania recording, generates every figure, and checks the expected artifacts.
