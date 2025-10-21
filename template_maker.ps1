Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg){ Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-OK($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[X] $msg" -ForegroundColor Red }

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Realpath {
    if (-not (Test-Command -Name 'Resolve-Path')) {
        Write-Err "Resolve-Path not available. Please update PowerShell."
        throw "Resolve-Path missing"
    }
}

function Ensure-Uv {
    if (Test-Command -Name 'uv') { return }
    $choice = Read-Host "uv not found. Install it now? (y/[n])"
    if ($choice -notin @('y','Y')) {
        Write-Err "uv is required to run this script. Exiting."
        exit 1
    }
    Write-Info "Installing uv…"
    try {
        # Recommended installer
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Common install path
        $uvBin = Join-Path $HOME ".local\bin"
        if (Test-Path $uvBin) { $env:Path = "$uvBin;$env:Path" }
    } catch {
        Write-Err "uv installation failed. See https://docs.astral.sh/uv/getting-started/installation/"
        throw
    }
    if (-not (Test-Command -Name 'uv')) {
        Write-Err "uv still not found on PATH after install."
        throw "uv not found"
    }
    Write-OK "uv installed."
}

function Ensure-BuildTools {
    param(
        [switch]$InstallIfMissing = $true,
        [switch]$RequireWinSDK    = $true
    )

    function Test-CppToolchain {
        $hasLink = [bool](Get-Command link.exe -ErrorAction SilentlyContinue)
        $hasCl   = [bool](Get-Command cl.exe   -ErrorAction SilentlyContinue)

        $sdkOk = -not $RequireWinSDK ? $true : $false

        if ($RequireWinSDK) {
            $candidates = @()

            # 1) Registry
            try {
                $roots = Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -ErrorAction Stop
                if ($roots -and $roots.PSObject.Properties.Name -contains 'KitsRoot10') {
                    $candidates += $roots.KitsRoot10
                }
            } catch { }

            # 2) Env var
            if ($env:WindowsSdkDir) { $candidates += $env:WindowsSdkDir }

            # 3) Common locations
            $candidates += @(
                'C:\Program Files (x86)\Windows Kits\10\',
                'C:\Program Files\Windows Kits\10\'
            )

            foreach ($root in $candidates | Where-Object { $_ -and (Test-Path $_) }) {
                if (Test-Path (Join-Path $root 'Lib')) { $sdkOk = $true; break }
            }
        }

        [pscustomobject]@{
            Link = $hasLink
            Cl   = $hasCl
            Sdk  = $sdkOk
        }
    }

    function Refresh-MsvcPath {
        $pf86 = ${env:ProgramFiles(x86)}
        if (-not $pf86) { return }
        $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path $vswhere)) { return }

        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if (-not $vsPath) { return }

        $toolRoot = Join-Path $vsPath 'VC\Tools\MSVC'
        if (-not (Test-Path $toolRoot)) { return }

        $latest = Get-ChildItem $toolRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $latest) { return }

        $binCandidates = @(
            Join-Path $latest.FullName 'bin\Hostx64\x64'
            Join-Path $latest.FullName 'bin\Hostx86\x64'
            Join-Path $latest.FullName 'bin\Hostx64\x86'
            Join-Path $latest.FullName 'bin\Hostx86\x86'
        ) | Where-Object { Test-Path $_ }

        foreach ($bin in $binCandidates) {
            $escaped = [regex]::Escape($bin)
            if ($env:Path -notmatch "(^|;)$escaped(;|$)") {
                $env:Path = "$bin;$env:Path"
            }
        }
    }

    Write-Info "Checking MSVC toolchain (cl/link) and Windows SDK…"
    Refresh-MsvcPath
    $state = Test-CppToolchain
    if ($state.Link -and $state.Cl -and $state.Sdk) {
        Write-OK "MSVC & Windows SDK detected."
        if (Get-Command rustup -ErrorAction SilentlyContinue) {
            try { & rustup default stable-x86_64-pc-windows-msvc | Out-Null } catch {}
        }
        return $true
    }

    if (-not $InstallIfMissing) {
        Write-Err "MSVC build tools or Windows SDK missing."
        throw "Build tools not present."
    }

    if (-not (Test-Command -Name 'winget')) {
        Write-Err "winget not found. Install Build Tools manually via Visual Studio Installer."
        throw "winget missing"
    }

    Write-Warn "Installing Visual Studio 2022 Build Tools (C++ workload + SDK)… (this can take a while)"
    $override = @(
        '--quiet','--wait','--norestart',
        '--add','Microsoft.VisualStudio.Workload.VCTools',
        '--includeRecommended'
    ) -join ' '

    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --override "$override"

    Refresh-MsvcPath
    $state = Test-CppToolchain
    if (-not ($state.Link -and $state.Cl -and $state.Sdk)) {
        Write-Err "MSVC/SDK still not detected after install."
        Write-Info "Open 'Visual Studio Installer' → Modify 'Build Tools' → ensure 'C++ build tools' + a Windows 10/11 SDK are selected."
        throw "Build tools not detected"
    }

    Write-OK "MSVC build tools ready."
    if (Test-Command -Name 'rustup') {
        try {
            & rustup default stable-x86_64-pc-windows-msvc | Out-Null
            & rustup component add rustfmt clippy | Out-Null
        } catch {}
    }
    return $true
}

function Ensure-Cargo {
    if (Test-Command -Name 'cargo') {
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin) { $env:Path = "$cargoBin;$env:Path" }
        return
    }
    $choice = Read-Host "Cargo not found. Install Rust/Cargo via rustup now? (y/[n])"
    if ($choice -notin @('y','Y')) {
        Write-Err "Cargo is required for FRCW/BEN path. Exiting."
        exit 1
    }
    Write-Info "Installing Rust/Cargo (rustup)…"
    try {
        if (Test-Command -Name 'winget') {
            winget install Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
        } else {
            $tmp = Join-Path $env:TEMP "rustup-init.exe"
            Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $tmp
            & $tmp -y
        }
        $cargoBin = Join-Path $HOME ".cargo\bin"
        if (Test-Path $cargoBin) { $env:Path = "$cargoBin;$env:Path" }
    } catch {
        Write-Err "Rust/Cargo installation failed. Install from https://www.rust-lang.org/tools/install and re-run."
        throw
    }
    if (-not (Test-Command -Name 'cargo')) {
        Write-Err "cargo still not found on PATH."
        throw "cargo not found"
    }
    Write-OK "Rust and Cargo installed."
}

function New-FileUtf8 {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    $dir = Split-Path -Path $Path -Parent
    if ([string]::IsNullOrWhiteSpace($dir)) { $dir = '.' }

    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $Content | Out-File -FilePath $Path -Encoding UTF8 -Force
}


function Write-BasicCliGerrychain {
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

@click.command()
@click.option("--graph-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(writable=True, dir_okay=False))
@click.option("--starting-plan", type=str)
@click.option("--pop-col", type=str)
@click.option("--rng-seed", type=int)
@click.option("--population-tolerance", type=float, default=0.01)
@click.option("--total-steps", type=int, default=10_000)
def main(
    graph_path,
    output_path,
    starting_plan,
    pop_col,
    rng_seed,
    population_tolerance,
    total_steps,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    graph = Graph.from_json(graph_path)

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

    with jl.open(output_path, "w") as f:
        for i, part in enumerate(chain):
            f.write(
                {
                    "assignment": part.assignment.to_series()
                    .astype(int)
                    .sort_index()
                    .to_list(),
                    "sample": i + 1,
                }
            )

if __name__ == "__main__":
    main()
'@
}

function Write-BatchExampleSimple {
@'
param(
  [int[]]$RngSeeds = @(42,43,44),
  [int]$TotalSteps = 1000
)

$TOPDIR = (Resolve-Path $PSScriptRoot).Path
$env:PYTHONHASHSEED = '0'

$chainOut  = Join-Path $TOPDIR 'chain_outputs'
$chainLogs = Join-Path $TOPDIR 'chain_logs'
New-Item -ItemType Directory -Force -Path $chainOut,$chainLogs | Out-Null

foreach ($seed in $RngSeeds) {
  $outFile = Join-Path $chainOut  "gerrymandria_chain_${TotalSteps}_steps_seed$seed.jsonl"
  $logFile = Join-Path $chainLogs "log_simple_rng_seed_$seed.log"

  & uv run "$TOPDIR\pipeline_scripts\example_cli.py" `
    --graph-path   "$TOPDIR\JSON_dualgraphs\gerrymandria.json" `
    --output-path  "$outFile" `
    --starting-plan "district" `
    --pop-col       "TOTPOP" `
    --rng-seed      $seed `
    --population-tolerance 0.01 `
    --total-steps   $TotalSteps *> $logFile
}
'@
}

function Write-BatchExampleParallel {
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
New-Item -ItemType Directory -Force -Path $chainOut,$chainLogs | Out-Null

$jobs = @()

foreach ($seed in $RngSeeds) {
  while (($jobs | Where-Object State -eq 'Running').Count -ge $MaxJobs) {
    Start-Sleep -Milliseconds 250
    Receive-Job -Job $jobs -Keep | Out-Null
    $jobs = $jobs | Where-Object { $_.State -in 'Running','NotStarted' }
  }

  $outFile = Join-Path $chainOut  "gerrymandria_chain_${TotalSteps}_steps_seed$seed.jsonl"
  $logFile = Join-Path $chainLogs "log_parallel_rng_seed_$seed.log"

  $job = Start-Job -Name "seed$seed" -ArgumentList $TOPDIR,$TotalSteps,$seed,$outFile,$logFile -ScriptBlock {
    param($topdir,$nsteps,$seed,$outFile,$logFile)
    $env:PYTHONHASHSEED = '0'
    & uv run "$topdir\pipeline_scripts\example_cli.py" `
      --graph-path   "$topdir\JSON_dualgraphs\gerrymandria.json" `
      --output-path  "$outFile" `
      --starting-plan "district" `
      --pop-col       "TOTPOP" `
      --rng-seed      $seed `
      --population-tolerance 0.01 `
      --total-steps   $nsteps *> $logFile
  }

  $jobs += $job
}

Write-Progress -Activity "Running jobs" -Status "Waiting for completion…"
Wait-Job -Job $jobs
Receive-Job -Job $jobs -Keep | Out-Null
Write-Progress -Activity "Running jobs" -Completed
'@
}

function Write-RustShExample {
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

function Write-JsonlToBen {
@'
param([switch]$Recurse = $true)

$files = Get-ChildItem -File -Filter *.jsonl -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben -m encode $f.FullName -v -w
}
'@
}

function Write-BenToXben {
@'
param([switch]$Recurse = $true)

$files = Get-ChildItem -File -Filter *.ben -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben -m x-encode $f.FullName -v -w
}
'@
}

# ============================
#            MAIN
# ============================
Ensure-Realpath
Ensure-Uv

$projectName = Read-Host "Enter the name of the new project to create"
if ([string]::IsNullOrWhiteSpace($projectName)) {
    $projectName = "my_project"
    Write-Warn "No project name provided. Using default: $projectName"
}

$useFrcw = Read-Host "Would you like to use FRCW in this project? (y/[n])"
$useBen  = $false
if ($useFrcw -match '^(y|Y)$') {
    Ensure-BuildTools
    Ensure-Cargo
    Write-Info "Installing FRCW from latest git commit…"
    & cargo install --git "https://github.com/mggg/frcw.rs" --branch "main"
    Write-OK "FRCW installed."
    Write-Info "Installing binary-ensemble…"
    & cargo install binary-ensemble
    Write-OK "binary-ensemble installed."
} else {
    $ans = Read-Host "Would you like to use BEN in this project? (y/[n])"
    if ($ans -match '^(y|Y)$') {
        $useBen = $true
        Ensure-Cargo
        Write-Info "Installing binary-ensemble…"
        & cargo install binary-ensemble
        Write-OK "binary-ensemble installed."
    }
}

$pythonVersion = Read-Host "What python version would you like (3.11, 3.12, 3.13)? (default: 3.11)"
if ($pythonVersion -notmatch '^(3\.11|3\.12|3\.13)$') {
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
Write-Info "Adding standard packages to pyproject.toml…"

# Remove default files uv created (if present)
Remove-Item -Force -ErrorAction SilentlyContinue "README.md","main.py"

# Add deps (include jsonlines used by example script)
& uv add numpy pandas matplotlib seaborn "gerrychain[geo]" maup ipykernel ipywidgets click gerrytools jsonlines
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
Write-Info "Downloading gerrymandria.json…"
Invoke-WebRequest "https://raw.githubusercontent.com/mggg/GerryChain/refs/heads/main/docs/_static/gerrymandria.json" -OutFile "JSON_dualgraphs\gerrymandria.json"

# Write helper files (.ps1 since we're on Windows)
New-FileUtf8 -Path "pipeline_scripts\example_cli.py"          -Content (Write-BasicCliGerrychain)
New-FileUtf8 -Path "pipeline_scripts\rust_example_script.ps1"  -Content (Write-RustShExample)
New-FileUtf8 -Path "batch_example_python_cli_simple.ps1"       -Content (Write-BatchExampleSimple)
New-FileUtf8 -Path "batch_example_python_cli_parallel.ps1"     -Content (Write-BatchExampleParallel)
New-FileUtf8 -Path "chain_outputs\jsonl_to_ben.ps1"            -Content (Write-JsonlToBen)
New-FileUtf8 -Path "chain_outputs\ben_to_xben.ps1"             -Content (Write-BenToXben)

Write-OK "Your project is ready!"
Write-Warn "If 'uv' or 'cargo' commands are not recognized in *new* shells, log out/in or ensure these are on PATH:"
Write-Host "  $HOME\.local\bin"
Write-Host "  $HOME\.cargo\bin"
Pop-Location
