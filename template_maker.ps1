Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'


# =====================================
# ========  UTILITY FUNCTIONS  ========
# =====================================

function Write-Info($msg)
{ Write-Host "[*] $msg" -ForegroundColor Cyan 
}
function Write-OK($msg)
{ Write-Host "[OK] $msg" -ForegroundColor Green 
}
function Write-Warn($msg)
{ Write-Host "[!] $msg" -ForegroundColor Yellow 
}
function Write-Err($msg)
{ Write-Host "[X] $msg" -ForegroundColor Red 
}

function Test-Command
{
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Confirm-Realpath
{
    if (-not (Test-Command -Name 'Resolve-Path'))
    {
        Write-Err "Resolve-Path not available. Please update PowerShell."
        throw "Resolve-Path missing"
    }
}

function New-FileUtf8
{
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    $dir = Split-Path -Path $Path -Parent
    if ([string]::IsNullOrWhiteSpace($dir))
    { $dir = '.' 
    }

    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $Content | Out-File -FilePath $Path -Encoding UTF8 -Force
}

function Invoke-WithRetry
{
    param(
        [Parameter(Mandatory)] [scriptblock] $Action,
        [int] $MaxAttempts = 5,
        [int] $InitialDelaySeconds = 2
    )

    $attempt = 1
    $delay = $InitialDelaySeconds

    while ($true)
    {
        try
        {
            return & $Action
        } catch
        {
            if ($attempt -ge $MaxAttempts)
            {
                throw  # rethrow last error after max attempts
            }

            Write-Warn "Attempt $attempt failed: $($_.Exception.Message)"
            Write-Info "Retrying in $delay seconds..."
            Start-Sleep -Seconds $delay

            $attempt++
            # simple backoff (cap it a bit)
            $delay = [Math]::Min($delay * 2, 30)
        }
    }
}

# ====================================================
# ========  SOFTWARE CHECKERS AND INSTALLERS  ========
# ====================================================

function Confirm-Uv
{
    if (Test-Command -Name 'uv')
    { return 
    }
    $choice = Read-Host "uv not found. Install it now? (y/[n])"
    if ($choice -notin @('y','Y'))
    {
        Write-Err "uv is required to run this script. Exiting."
        exit 1
    }
    Write-Info "Installing uv..."
    try
    {
        # Recommended installer
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Common install path
        $uvBin = Join-Path $HOME ".local\bin"
        if (Test-Path $uvBin)
        { $env:Path = "$uvBin;$env:Path" 
        }
    } catch
    {
        Write-Err "uv installation failed. See https://docs.astral.sh/uv/getting-started/installation/"
        throw
    }
    if (-not (Test-Command -Name 'uv'))
    {
        Write-Err "uv still not found on PATH after install."
        throw "uv not found"
    }
    Write-OK "uv installed."
}

function Confirm-BuildTools
{
    param(
        [bool]$InstallIfMissing = $true,
        [bool]$RequireWinSDK    = $true
    )

    function Test-CppToolchain
    {
        $hasLink = [bool](Get-Command link.exe -ErrorAction SilentlyContinue)
        $hasCl   = [bool](Get-Command cl.exe   -ErrorAction SilentlyContinue)


        if (-not $RequireWinSDK)
        {
            $sdkOk = $true
        } else
        {
            $sdkOk = $false
        }

        if ($RequireWinSDK)
        {
            $candidates = @()

            # 1) Registry
            try
            {
                $roots = Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -ErrorAction Stop
                if ($roots -and $roots.PSObject.Properties.Name -contains 'KitsRoot10')
                {
                    $candidates += $roots.KitsRoot10
                }
            } catch
            { 
            }

            # 2) Env var
            if ($env:WindowsSdkDir)
            { $candidates += $env:WindowsSdkDir 
            }

            # 3) Common locations
            $candidates += @(
                'C:\Program Files (x86)\Windows Kits\10\',
                'C:\Program Files\Windows Kits\10\'
            )

            foreach ($root in $candidates | Where-Object { $_ -and (Test-Path $_) })
            {
                if (Test-Path (Join-Path $root 'Lib'))
                { $sdkOk = $true; break 
                }
            }
        }

        [pscustomobject]@{
            Link = $hasLink
            Cl   = $hasCl
            Sdk  = $sdkOk
        }
    }

    function Update-MsvcPath
    {
        $pf86 = ${env:ProgramFiles(x86)}
        if (-not $pf86)
        { return 
        }
        $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path $vswhere))
        { return 
        }

        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if (-not $vsPath)
        { return 
        }

        $toolRoot = Join-Path $vsPath 'VC\Tools\MSVC'
        if (-not (Test-Path $toolRoot))
        { return 
        }

        $latest = Get-ChildItem $toolRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $latest)
        { return 
        }

        $binCandidates = @(
            Join-Path $latest.FullName 'bin\Hostx64\x64'
            Join-Path $latest.FullName 'bin\Hostx86\x64'
            Join-Path $latest.FullName 'bin\Hostx64\x86'
            Join-Path $latest.FullName 'bin\Hostx86\x86'
        ) | Where-Object { Test-Path $_ }

        foreach ($bin in $binCandidates)
        {
            $escaped = [regex]::Escape($bin)
            if ($env:Path -notmatch "(^|;)$escaped(;|$)")
            {
                $env:Path = "$bin;$env:Path"
            }
        }
    }

    Write-Info "Checking MSVC toolchain (cl/link) and Windows SDK..."
    Update-MsvcPath
    $state = Test-CppToolchain
    if ($state.Link -and $state.Cl -and $state.Sdk)
    {
        Write-OK "MSVC & Windows SDK detected."
        if (Get-Command rustup -ErrorAction SilentlyContinue)
        {
            try
            { & rustup default stable-x86_64-pc-windows-msvc | Out-Null 
            } catch
            {
            }
        }
        return $true
    }

    if (-not $InstallIfMissing)
    {
        Write-Err "MSVC build tools or Windows SDK missing."
        throw "Build tools not present."
    }

    if (-not (Test-Command -Name 'winget'))
    {
        Write-Err "winget not found. Install Build Tools manually via Visual Studio Installer."
        throw "winget missing"
    }

    Write-Warn "Installing Visual Studio 2022 Build Tools (C++ workload + SDK)... (this can take a while)"
    $override = @(
        '--quiet','--wait','--norestart',
        '--add','Microsoft.VisualStudio.Workload.VCTools',
        '--includeRecommended'
    ) -join ' '

    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "$override"

    Update-MsvcPath
    $state = Test-CppToolchain
    if (-not ($state.Link -and $state.Cl -and $state.Sdk))
    {
        Write-Err "MSVC/SDK still not detected after install."
        Write-Info "Open 'Visual Studio Installer' -> Modify 'Build Tools' -> ensure 'C++ build tools' + a Windows 10/11 SDK are selected."
        throw "Build tools not detected"
    }

    Write-OK "MSVC build tools ready."
    if (Test-Command -Name 'rustup')
    {
        try
        {
            & rustup default stable-x86_64-pc-windows-msvc | Out-Null
            & rustup component add rustfmt clippy | Out-Null
        } catch
        {
        }
    }
    return $true
}

function Confirm-Cargo
{
    if (Test-Command -Name 'cargo')
    {
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin)
        { $env:Path = "$cargoBin;$env:Path" 
        }
        return
    }
    $choice = Read-Host "Cargo not found. Install Rust/Cargo via rustup now? (y/[n])"
    if ($choice -notin @('y','Y'))
    {
        Write-Err "Cargo is required for FRCW/BEN path. Exiting."
        exit 1
    }
    Write-Info "Installing Rust/Cargo (rustup)..."
    try
    {
        if (Test-Command -Name 'winget')
        {
            winget install Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
        } else
        {
            $tmp = Join-Path $env:TEMP "rustup-init.exe"
            Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $tmp
            & $tmp -y
        }
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin)
        { $env:Path = "$cargoBin;$env:Path" 
        }
    } catch
    {
        Write-Err "Rust/Cargo installation failed. Install from https://www.rust-lang.org/tools/install and re-run."
        throw
    }
    if (-not (Test-Command -Name 'cargo'))
    {
        Write-Err "cargo still not found on PATH."
        throw "cargo not found"
    }
    Write-OK "Rust and Cargo installed."
}

# =====================================
# ========  MAIN PWSH SCRIPTS  ========
# =====================================

function Write-BatchExampleSimple
{
    @'
param(
  [int[]]$RngSeeds = @(42,43,44),
  [int]$TotalSteps = 1000,
  [int[]]$RngSeeds2 = @(42),
  [int]$TotalSteps2 = 100000
)

$TOPDIR = (Resolve-Path $PSScriptRoot).Path
$env:PYTHONHASHSEED = '0'

$chainOut  = Join-Path $TOPDIR 'chain_outputs'
$chainLogs = Join-Path $TOPDIR 'chain_logs'
New-Item -ItemType Directory -Force -Path $chainOut,$chainLogs | Out-Null

foreach ($seed in $RngSeeds) {
  $outFile = Join-Path $chainOut  "gerrymandria_chain_${TotalSteps}_steps_seed$seed.jsonl"
  $logFile = Join-Path $chainLogs "log_simple_rng_seed_$seed.log"

  & uv run (Join-Path "$TOPDIR" (Join-Path "pipeline_scripts" "example_cli.py")) `
    --graph-path   (Join-Path "$TOPDIR" (Join-Path "JSON_dualgraphs" "gerrymandria.json")) `
    --output-path  "$outFile" `
    --starting-plan "district" `
    --pop-col       "TOTPOP" `
    --rng-seed      $seed `
    --population-tolerance 0.01 `
    --total-steps   $TotalSteps *> $logFile
}

foreach ($seed in $RngSeeds2) {
  $outFile = Join-Path (Join-Path $TOPDIR "chain_outputs") ("MN_chain_{0}_steps_seed{1}.jsonl.ben" -f $NSteps2, $seed)

  & uv run (Join-Path $TOPDIR (Join-Path "pipeline_scripts" "example_cli.py")) `
    --graph-path   (Join-Path $TOPDIR (Join-Path "JSON_dualgraphs" "MN_precincts.geojson")) `
    --output-path  $outFile `
    --starting-plan "CONGDIST" `
    --pop-col       "TOTPOP" `
    --rng-seed      $seed `
    --population-tolerance 0.05 `
    --total-steps   $TotalSteps2 `
    --writeas "ben"
}
'@
}

function Write-BatchExampleParallel
{
    @'
param(
    [int]$MaxJobs = [Environment]::ProcessorCount,
    [int[]]$RngSeeds = 1..50,
    [int]$TotalSteps = 1000
)

$TOPDIR = (Resolve-Path $PSScriptRoot).Path
$env:PYTHONHASHSEED = '0'

$chainOut  = Join-Path $TOPDIR 'chain_outputs'
$chainLogs = Join-Path $TOPDIR 'chain_logs'
New-Item -ItemType Directory -Force -Path $chainOut, $chainLogs | Out-Null

# Resolve uv once so jobs don't depend on profile PATH
$uvExe = (Get-Command uv -ErrorAction Stop).Source

$jobs = @()

foreach ($seed in $RngSeeds)
{

    # throttle
    while (($jobs | Where-Object State -eq 'Running').Count -ge $MaxJobs)
    {
        Start-Sleep -Milliseconds 200
        $done = $jobs | Where-Object State -in 'Completed','Failed','Stopped'
        if ($done)
        {
            Receive-Job -Job $done -Keep | Out-Null
            $jobs = $jobs | Where-Object State -in 'Running','NotStarted'
        }
    }

    $outFile = Join-Path $chainOut ("gerrymandria_chain_{0}_steps_seed{1}.jsonl" -f $TotalSteps, $seed)
    $logFile = Join-Path $chainLogs ("log_parallel_rng_seed_{0}.log" -f $seed)

    $job = Start-Job -Name "seed$seed" `
        -ArgumentList $TOPDIR, $TotalSteps, $seed, $outFile, $logFile, $uvExe `
        -ScriptBlock {
        param($topdir, $nsteps, $seed, $outFile, $logFile, $uvExe)

        Set-StrictMode -Version Latest
        $ErrorActionPreference = 'Stop'
        $env:PYTHONHASHSEED = '0'

        # Cross-platform paths
        $exampleCli = Join-Path $topdir (Join-Path 'pipeline_scripts' 'example_cli.py')
        $graphPath  = Join-Path $topdir (Join-Path 'JSON_dualgraphs' 'gerrymandria.json')

        # Build args as an array
        $arguments = @(
            'run', '--project', $topdir, $exampleCli,
            '--graph-path', $graphPath,
            '--output-path', $outFile,
            '--starting-plan', 'district',
            '--pop-col', 'TOTPOP',
            '--rng-seed', $seed,
            '--population-tolerance', '0.01',
            '--total-steps', $nsteps,
            '--writeas', 'jsonl'
        )

        try
        {
            & $uvExe @arguments *> $logFile
        } catch
        {
            $_ | Out-String | Add-Content $logFile
            throw
        }
    }

    $jobs += $job
}

Write-Progress -Activity "Running jobs" -Status "Waiting for completion..."
Wait-Job -Job $jobs
Receive-Job -Job $jobs -Keep | Out-Null
Write-Progress -Activity "Running jobs" -Completed

'@
}

function Write-RustShExample
{
    @'
param(
  [string]$PlanName = 'district',
  [int]$n_steps = 1000,
  [int]$seed = 42,
  [double]$tol = 0.01,
  [string]$pop_col = 'TOTPOP'
)

# Project root is the parent of this script's folder
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$json_dir   = Join-Path $ProjectRoot 'JSON_dualgraphs'
$output_dir = Join-Path $ProjectRoot 'chain_outputs'

# Ensure output dir exists
New-Item -ItemType Directory -Force -Path $output_dir | Out-Null

# Build JSON path (don't Resolve-Path until we know it exists)
$json_file = Join-Path $json_dir 'gerrymandria.json'

if (-not (Test-Path $json_file)) {
  Write-Error "Could not find graph JSON at: $json_file`nDid the bootstrap step download it?"
  exit 1
}

$final_output_file = Join-Path $output_dir ("gerrymandria_chain_{0}_steps.jsonl.ben" -f $n_steps)

& frcw `
  --assignment-col $PlanName `
  --graph-json $json_file `
  --n-steps $n_steps `
  --pop-col $pop_col `
  --rng-seed $seed `
  --tol $tol `
  --variant district-pairs-rmst `
  --writer ben `
  --batch-size 1 `
  --n-threads 1 `
  --output-file $final_output_file
'@
}

function Write-JsonlToBen
{
    @'
param([switch]$Recurse = $true)

$files = Get-ChildItem -File -Filter *.jsonl -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben -m encode $f.FullName -v -w
}
'@
}

function Write-BenToXben
{
    @'
param([switch]$Recurse = $true)

$files = Get-ChildItem -File -Filter *.ben -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben -m x-encode $f.FullName -v -w
}
'@
}

# =====================================
# ========  PYTHON CLI SCRIPT  ========
# =====================================

function Write-BasicCliGerrychain
{
    @'
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally
from gerrychain.accept import always_accept
from gerrychain.proposals.tree_proposals import recom
from functools import partial
import random
import jsonlines as jl
import click
import numpy as np
from pathlib import Path
from pyben import PyBenEncoder
import sys


@click.command()
@click.option("--graph-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(writable=True, dir_okay=False))
@click.option("--starting-plan", type=str)
@click.option("--pop-col", type=str)
@click.option("--rng-seed", type=int)
@click.option("--population-tolerance", type=float, default=0.01)
@click.option("--total-steps", type=int, default=10_000)
@click.option("--writeas", type=click.Choice(["jsonl", "ben"]), default="ben")
def main(
    graph_path,
    output_path,
    starting_plan,
    pop_col,
    rng_seed,
    population_tolerance,
    total_steps,
    writeas,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        if graph_path.endswith(".json"):
            graph = Graph.from_json(graph_path)
        else:
            graph = Graph.from_file(graph_path)
    except Exception as e:
        raise ValueError(f"Failed to load graph from {graph_path}: {e}")

    initial_partition = Partition(
        graph,
        assignment=starting_plan,
        updaters={"population": Tally(pop_col, alias="population")},
    )

    ideal_pop = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(
        recom,
        pop_col=pop_col,
        pop_target=ideal_pop,
        epsilon=population_tolerance,
        node_repeats=1,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[],
        initial_state=initial_partition,
        total_steps=total_steps,
        accept=always_accept,
    )

    graph_node_order = list(graph.nodes)

    # This will print to the standard error stream so that logging does not interfere with the
    # standard output.
    print(
        f"Writing output to '{Path(output_path).name}' in '{writeas.upper()}' format.",
        file=sys.stderr,
        flush=True,
    )
    match writeas:
        case "jsonl":
            with jl.open(output_path, "w") as writer:
                for i, partition in enumerate(chain.with_progress_bar()):
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    writer.write(
                        {
                            "assignment": ordered_assignment,
                            "sample": i + 1,
                        }
                    )

        case "ben":
            with PyBenEncoder(output_path, overwrite=True) as encoder:
                for partition in chain.with_progress_bar():
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    encoder.write(ordered_assignment)

        case _:
            raise ValueError(f"Unsupported writeas format: {writeas}")


if __name__ == "__main__":
    main()
'@
}


# =============================================
# ========  PYTHON PIPELINE FUNCTIONS  ========
# =============================================


function Write-PythonProcessPartisanBias
{
    @'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from pathlib import Path
import geopandas as gpd
import numpy as np
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, vote_arrays):
    assign = np.asarray(assignment_vector, dtype=np.int64)

    k = int(assign.max())  # number of districts

    out = {"sample": sample_idx, "pb_scores": {}}

    for dem_votes, rep_votes, name in vote_arrays:
        dem_tot = np.bincount(assign, weights=dem_votes, minlength=k)
        rep_tot = np.bincount(assign, weights=rep_votes, minlength=k)

        total = dem_tot + rep_tot
        dem_share = np.divide(
            dem_tot, total, out=np.zeros_like(dem_tot, dtype="float64"), where=total > 0
        )

        mean_share = dem_share.mean()
        pb = (dem_share > mean_share).sum() / k - 0.5
        out["pb_scores"][name] = float(pb)

    return out


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_partisan_bias_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    gdf = gpd.read_file(GRAPH_PATH)

    elections = ["PRES16", "SSEN16"]
    election_pairs = [(f"{name}D", f"{name}R") for name in elections]

    # grab vote columns as numpy arrays once
    vote_arrays = []
    for d_col, r_col in election_pairs:
        vote_arrays.append(
            (
                gdf[d_col].to_numpy(dtype="float64", copy=True),
                gdf[r_col].to_numpy(dtype="float64", copy=True),
                d_col[:-1],
            )
        )
    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing partisan bias (batch {batch_no+1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, vote_arrays) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
}


function Write-PythonProcessPolsby
{
    @'
import jsonlines as jl
from gerrychain import GeographicPartition, Graph
from gerrychain.metrics import polsby_popper
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_number, assignment_vector, graph):
    part = GeographicPartition(
        graph, assignment={i: val for i, val in enumerate(assignment_vector)}
    )
    return {"sample": sample_number, "scores": polsby_popper(part)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_polsby_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    graph = Graph.from_file(GRAPH_PATH)

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Polsby-Popper scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, graph) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
}


function Write-PythonProcessReock
{
    @'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import geopandas as gpd
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, geo_only):
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="pygeos support was removed in 1.0. geopandas.use_pygeos is a no-op",
        category=UserWarning,
    )

    from gerrytools.scoring import reock

    geo_new = geo_only.copy()
    geo_new["assignment"] = np.array(assignment_vector) - 1
    dissolved = geo_new.dissolve(by="assignment")
    return {"sample": sample_idx, "scores": reock().apply(dissolved)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_reock_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    gdf = gpd.read_file(GRAPH_PATH)
    geo_only = gdf[["geometry"]]

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Reock scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, geo_only) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
}

function Write-PythonProcessSplits
{
    @'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    u,
    v,
    same_county_mask,
    county_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    cut = assignment[u] != assignment[v]
    cut_edges = int(cut.sum())
    county_splits = len(set(county_names[cut & same_county_mask]))
    return {
        "sample": sample_idx,
        "scores": {
            "county_splits": county_splits,
            "cut_edges": cut_edges,
        },
    }


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_split_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    graph = Graph.from_file(GRAPH_PATH)
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]

    # Masks that depend only on the graph (not on assignments)
    # A "split" counts only when it's a cut edge AND both endpoints share the same county/place.
    same_county_mask = np.fromiter(
        (
            graph.nodes[int(a)]["COUNTYNAME"] == graph.nodes[int(b)]["COUNTYNAME"]
            for a, b in edges
        ),
        dtype=bool,
        count=len(edges),
    )

    # Don't need to record b since we are going to filter to when it has the same value as a
    county_names = np.fromiter(
        (graph.nodes[int(a)]["COUNTYNAME"] for a, _ in edges),
        dtype=object,
        count=len(edges),
    )

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all split scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, u, v, same_county_mask, county_names)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
}


function Write-PythonProcessTotalDemWins
{
    @'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import geopandas as gpd
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    dem_count_matrix,
    rep_count_matrix,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    race_totals = {name: 0 for name in race_names}

    for part in np.unique(assignment):
        mask = assignment == part
        dem_totals = dem_count_matrix[mask].sum(axis=0)
        rep_totals = rep_count_matrix[mask].sum(axis=0)
        dem_wins = dem_totals > rep_totals

        for i, race in enumerate(race_names):
            race_totals[race] += 1 if dem_wins[i] else 0

    return ({"sample": sample_idx, "scores": race_totals},)


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_dem_win_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    df = gpd.read_file(GRAPH_PATH)
    graph = Graph.from_geodataframe(df)
    df.drop(columns=["geometry"], inplace=True)

    race_names = [
        "PRES16",
        "SSEN16",
    ]

    dem_rep_pairs = [(f"{name}D", f"{name}R") for name in race_names]
    dem_count_matrix = df[[pair[0] for pair in dem_rep_pairs]].to_numpy()
    rep_count_matrix = df[[pair[1] for pair in dem_rep_pairs]].to_numpy()

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing total dem wins in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, dem_count_matrix, rep_count_matrix)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
'@
}

# ==============================================
# ========  MAIN INSTALLATION FUNCTION  ========
# ==============================================

function Main
{
    Confirm-Realpath
    Confirm-Uv

    $projectName = Read-Host "Enter the name of the new project to create"
    if ([string]::IsNullOrWhiteSpace($projectName))
    {
        $projectName = "my_project"
        Write-Warn "No project name provided. Using default: $projectName"
    }

    $useFrcw = Read-Host "Would you like to use FRCW in this project? (y/[n])"
    if ($useFrcw -match '^(y|Y)$')
    {
        Confirm-BuildTools
        Confirm-Cargo
        Write-Info "Installing FRCW from latest git commit..."
        & cargo install --git "https://github.com/mggg/frcw.rs" --branch "main"
        Write-OK "FRCW installed."
        Write-Info "Installing binary-ensemble..."
        & cargo install binary-ensemble
        Write-OK "binary-ensemble installed."
    } else
    {
        $ans = Read-Host "Would you like to use BEN in this project? (y/[n])"
        if ($ans -match '^(y|Y)$')
        {
            Confirm-Cargo
            Write-Info "Installing binary-ensemble..."
            & cargo install binary-ensemble
            Write-OK "binary-ensemble installed."
        }
    }

    $pythonVersion = Read-Host "What python version would you like (3.11, 3.12, 3.13)? (default: 3.11)"
    if ($pythonVersion -notmatch '^(3\.11|3\.12|3\.13)$')
    {
        Write-Warn "Invalid python version. Using default 3.11."
        $pythonVersion = '3.11'
    }

    Write-Info "Creating project: $projectName"
    New-Item -ItemType Directory -Force -Path $projectName | Out-Null
    Push-Location $projectName

    # Ensure uv Python and init
    & uv python install $pythonVersion
    & uv init --python $pythonVersion

    Write-OK "Project $projectName initialized with uv ($pythonVersion)."
    Write-Info "Adding standard packages to pyproject.toml..."

    # Remove default files uv created (if present)
    Remove-Item -Force -ErrorAction SilentlyContinue "README.md","main.py"

    # Add deps (include jsonlines used by example script)
    & uv add numpy pandas matplotlib seaborn "gerrychain[geo]" maup ipykernel `
        ipywidgets click gerrytools binary-ensemble joblib joblib-progress docker

    # Formatter that I like
    & uv add tool black

    # Create directories
    $dirs = @(
        "data","JSON_dualgraphs","notebooks","pipeline_scripts",
        "figures","stats","chain_outputs","chain_logs","dev_files"
    )
    $dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

    # .gitignore
    Add-Content -Path ".gitignore" -Value "dev_files"

    # .env (uv --env-file expects KEY=VALUE lines; no 'export')
    Add-Content -Path ".env" -Value "PYTHONHASHSEED=0"

    # Download JSON file
    Write-Info "Downloading gerrymandria.json..."
    # Make sure destination exists
    $uri = "https://raw.githubusercontent.com/mggg/GerryChain/refs/heads/main/docs/_static/gerrymandria.json"
    $destDir = "JSON_dualgraphs"
    $destFile = Join-Path $destDir "gerrymandria.json"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    Invoke-WithRetry -MaxAttempts 5 -Action {
        Invoke-WebRequest -Uri $uri -OutFile $destFile -UseBasicParsing
    }

    Write-Info "Downloading MN_precincts.geojson..."
    $uri = "https://github.com/mggg/GerryChain/raw/main/docs/_static/MN.zip"

    Invoke-WithRetry -MaxAttempts 5 -Action {

        # create a temp *zip* path (PS5 Expand-Archive checks extension)
        $tmpZip = Join-Path $env:TEMP ("MN_" + [guid]::NewGuid().ToString() + ".zip")

        try
        {
            Invoke-WebRequest -Uri $uri -OutFile $tmpZip -UseBasicParsing
            Expand-Archive -LiteralPath $tmpZip -DestinationPath $destDir -Force
        } finally
        {
            Remove-Item -LiteralPath $tmpZip -ErrorAction SilentlyContinue
        }
    }

    # Write helper files (.ps1 since we're on Windows)
    New-FileUtf8 -Path "pipeline_scripts\example_cli.py"          -Content (Write-BasicCliGerrychain)
    New-FileUtf8 -Path "pipeline_scripts\rust_example_script.ps1"  -Content (Write-RustShExample)
    New-FileUtf8 -Path "batch_example_python_cli_simple.ps1"       -Content (Write-BatchExampleSimple)
    New-FileUtf8 -Path "batch_example_python_cli_parallel.ps1"     -Content (Write-BatchExampleParallel)
    New-FileUtf8 -Path "chain_outputs\jsonl_to_ben.ps1"            -Content (Write-JsonlToBen)
    New-FileUtf8 -Path "chain_outputs\ben_to_xben.ps1"             -Content (Write-BenToXben)

    # Make the python processing scripts
    New-Item -ItemType Directory -Force -Path "pipeline_scripts\metrics" | Out-Null
    New-FileUtf8 -Path "pipeline_scripts\metrics\process_partisan_bias.py" -Content (Write-PythonProcessPartisanBias)
    New-FileUtf8 -Path "pipeline_scripts\metrics\process_polsby.py"        -Content (Write-PythonProcessPolsby)
    New-FileUtf8 -Path "pipeline_scripts\metrics\process_reock.py"         -Content (Write-PythonProcessReock)
    New-FileUtf8 -Path "pipeline_scripts\metrics\process_splits.py"        -Content (Write-PythonProcessSplits)
    New-FileUtf8 -Path "pipeline_scripts\metrics\process_total_dem_wins.py"  -Content (Write-PythonProcessTotalDemWins)


    Write-OK "Your project is ready!"
    Write-Warn "If 'uv' or 'cargo' commands are not recognized in *new* shells, log out/in or ensure these are on PATH:"
    Write-Host "  $HOME\.local\bin"
    Write-Host "  $HOME\.cargo\bin"
    Pop-Location
}


if ($MyInvocation.InvocationName -ne '.')
{
    Main
}
