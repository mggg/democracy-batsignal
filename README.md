# Democracy Batsignal

Sets up a ready-to-run redistricting-analysis project in one step: a [uv](https://docs.astral.sh/uv/)-managed
Python environment with [GerryChain](https://github.com/mggg/GerryChain) and friends, the Rust chain runner
[RustReCom](https://github.com/mggg/rustrecom), the
[BEN](https://pypi.org/project/binary-ensemble/) ensemble-compression tools, and example scripts for every
stage of the pipeline: running chains, compressing ensembles, and computing scores.

## Quickstart

On macOS or Linux:

```bash
./democracy-batsignal.sh
```

On Windows (PowerShell 5.1 or later):

```powershell
powershell -ExecutionPolicy Bypass -File .\democracy-batsignal.ps1
```

The script asks for a project name, whether to install RustReCom, and a Python version. BEN and
GerryTools are installed in the uv-managed Python environment.
It offers to install anything that is missing (uv, Rust/cargo, and on Windows the MSVC build tools),
then creates the project folder next to wherever you ran it from.

Both installers are fully self-contained single files. You do not need to clone this repo; copying
just `democracy-batsignal.sh` or `democracy-batsignal.ps1` onto a machine is enough. The only
network access needed is for installing the tools themselves and downloading the PA example
geometry.

## What you get

```
my_project/
├── pyproject.toml            # uv-managed environment (gerrychain, binary-ensemble, ...)
├── JSON_dualgraphs/          # dual graphs: gerrymandria.json and pa_dualgraph.json examples
├── pipeline_scripts/
│   ├── example_cli.py        # click CLI that records a GerryChain ReCom chain as BENDL
│   ├── pa_example_script_*.*  # paired Bash and PowerShell RustReCom examples
│   └── metrics/              # per-metric scoring scripts that read a BENDL recording
├── batch_example_python_cli_simple.*    # run a few seeded chains one after another
├── batch_example_python_cli_parallel.*  # run many seeded chains with a concurrency cap
├── chain_outputs/            # self-contained BENDL chain recordings land here
├── chain_logs/               # per-run logs from the batch scripts
├── stats/                    # metric outputs
├── figures/, notebooks/, data/, dev_files/
└── .env                      # PYTHONHASHSEED=0 for reproducible python runs
```

Bash projects get `.sh` helper scripts, Windows projects get `.ps1` versions; the Python scripts are
identical on every platform.

## The pipeline

1. **Run a chain.** Either the Python route (`batch_example_python_cli_simple.*` drives
   `pipeline_scripts/example_cli.py`) or the much faster Rust route
   (`pipeline_scripts/pa_example_script_vanilla.sh` on Bash or the matching `.ps1` script on
   PowerShell drives `rustrecom chain`). Both write `.bendl`
   recordings into `chain_outputs/`; each recording bundles the assignments with its graph and
   run metadata.

   Ready-to-use tilted-run objectives are in `pipeline_scripts/rustrecom_objectives/`.

2. **Inspect or convert ensembles.** BENDL is the default working format. See `uv run ben --help` for
   tools to inspect, decode, look up, relabel, or convert recordings.

3. **Score the ensemble.** The scripts in `pipeline_scripts/metrics/` read a `.bendl` file and write
   per-plan scores to `stats/` (Polsby-Popper, Reock, county splits and cut edges, partisan bias,
   Dem seat counts). They are plain Python driven by constants at the top of each file; point them at
   your chain file and graph and run them with `uv run`.

Reproducibility notes: chain runs are seeded through `--rng-seed`, the metric scripts seed their
subsampling, and the batch scripts export `PYTHONHASHSEED=0` so GerryChain runs are repeatable.

## Repo development

`democracy-batsignal.sh` and `democracy-batsignal.ps1` are **generated files**; do not edit them
directly.
The sources are:

- `template_project/` — a runnable uv project and the source of every generated project.
  Edit its `pyproject.toml`, scripts, data, or directory layout as you would any other
  project. Keep both `.sh` and `.ps1` helper variants beside each other; the generator
  includes only the variant for each installer.
- `installer_src/skeleton.sh` and `installer_src/skeleton.ps1` — the installer logic, with a
  `# {{GENERATED_PAYLOADS}}` marker where the template files get embedded.

After changing any of those, regenerate the installers with:

```bash
python3 generate_installers.py          # rewrite democracy-batsignal.sh / .ps1
python3 generate_installers.py --check  # verify they are up to date (useful in CI)
```

The generator embeds UTF-8 project files, preserves directories represented by `.gitkeep`,
and skips `uv.lock` plus local environment/cache directories such as `.venv` and
`__pycache__`. Each installer resolves dependencies after applying the user's Python choice.

## Container smoke test

Run the generated installers and every example workflow in a disposable Ubuntu/PowerShell
container with:

```bash
./test_generated_projects.sh
```

The Docker build runs each installer in its own clean stage, including separate uv, Rust, and
RustReCom bootstraps, then caches those expensive setup layers. The container run reduces the
example chains to two steps, runs every Bash and PowerShell helper, evaluates a PA chain, generates
every figure, and checks the expected artifacts. Repeated runs reuse Docker's build cache until an
installer changes.
