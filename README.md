# Democracy Batsignal

Sets up a ready-to-run redistricting-analysis project in one step: a [uv](https://docs.astral.sh/uv/)-managed
Python environment with [GerryChain](https://github.com/mggg/GerryChain) and friends, the Rust chain runner
[FRCW](https://github.com/mggg/rustrecom), the [BEN](https://crates.io/crates/binary-ensemble) ensemble-compression
tools, and example scripts for every stage of the pipeline: running chains, compressing ensembles, and
computing scores.

## Quickstart

On macOS or Linux:

```bash
./template_maker.sh
```

On Windows (PowerShell 5.1 or later):

```powershell
powershell -ExecutionPolicy Bypass -File .\template_maker.ps1
```

The script asks for a project name, whether to install FRCW and/or BEN (both via cargo, along with
the [ben-process](https://github.com/peterrrock2/ben-process) metrics engine), and a Python version.
It offers to install anything that is missing (uv, Rust/cargo, and on Windows the MSVC build tools),
then creates the project folder next to wherever you ran it from.

Both installers are fully self-contained single files. You do not need to clone this repo; copying
just `template_maker.sh` or `template_maker.ps1` onto a machine is enough. The only network access
needed is for installing the tools themselves and downloading the MN example data.

## What you get

```
my_project/
├── pyproject.toml            # uv-managed environment (gerrychain, binary-ensemble, ...)
├── JSON_dualgraphs/          # dual graphs: gerrymandria.json and MN_precincts.geojson examples
├── pipeline_scripts/
│   ├── example_cli.py        # click CLI that runs a GerryChain ReCom chain -> JSONL or BEN
│   ├── rust_example_script.* # the same idea, but running a chain through FRCW
│   └── metrics/              # per-metric scoring scripts that read a BEN ensemble
├── batch_example_python_cli_simple.*    # run a few seeded chains one after another
├── batch_example_python_cli_parallel.*  # run many seeded chains with a concurrency cap
├── chain_outputs/            # chain results land here (plus jsonl_to_ben / ben_to_xben helpers)
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
   (`pipeline_scripts/rust_example_script.*` drives `frcw`). Both write either plain JSONL or
   compressed `.ben` ensembles into `chain_outputs/`.

2. **Compress / convert ensembles.** `chain_outputs/jsonl_to_ben.*` converts any JSONL ensembles
   sitting in `chain_outputs/` to BEN, and `ben_to_xben.*` recompresses BEN to the much smaller XBEN
   for archiving. See `ben --help` for the full toolset (decode, lookup, relabel, canonicalize, ...).

3. **Score the ensemble.** The scripts in `pipeline_scripts/metrics/` read a `.ben` file and write
   per-plan scores to `stats/` (Polsby-Popper, Reock, county splits and cut edges, partisan bias,
   Dem seat counts). They are plain Python driven by constants at the top of each file; point them at
   your chain file and graph and run them with `uv run`. For large ensembles the `ben-process` CLI
   computes most of these metrics natively in Rust (Parquet output); run `ben-process --help`.

Reproducibility notes: chain runs are seeded through `--rng-seed`, the metric scripts seed their
subsampling, and the batch scripts export `PYTHONHASHSEED=0` so GerryChain runs are repeatable.

## Repo development

`template_maker.sh` and `template_maker.ps1` are **generated files**; do not edit them directly.
The sources are:

- `template/` — the real project files (`common/` shared Python and data, `bash/` and
  `powershell/` platform wrappers), editable and testable as ordinary scripts.
- `installer_src/skeleton.sh` and `installer_src/skeleton.ps1` — the installer logic, with a
  `# {{GENERATED_PAYLOADS}}` marker where the template files get embedded.

After changing any of those, regenerate the installers with:

```bash
python3 generate_installers.py          # rewrite template_maker.sh / template_maker.ps1
python3 generate_installers.py --check  # verify they are up to date (useful in CI)
```

The `my_project/` folder is a checked-in example of what the generator produces.
