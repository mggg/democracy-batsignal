# This is a direct RustReCom CLI reference. Use run_chains.py to coordinate normal batches.
# `chain` samples ordinary ReCom plans from the assignment stored on each graph node.
# Input flags identify the adjacency-data graph and its assignment and population columns.
# Chain flags set the seed, number of steps, population tolerance, and ReCom proposal variant.
# Output flags record the graph, metadata, and assignment stream together in a BENDL file.

param(
    [int]$NSteps = 100000,
    [int[]]$RngSeeds = @(42, 43),
    [double]$Tolerance = 0.01,
    [string]$AssignmentColumn = 'seed_plan',
    [string]$PopulationColumn = 'total_pop_20'
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$GraphJson = Join-Path $ProjectRoot 'JSON_dualgraphs/pa_dualgraph.json'
$OutputDir = Join-Path $ProjectRoot 'chain_outputs'
$ToleranceLabel = $Tolerance.ToString(
    [System.Globalization.CultureInfo]::InvariantCulture
).Replace('.', 'p')

if (-not (Test-Path -LiteralPath $GraphJson -PathType Leaf))
{
    Write-Error "Could not find graph JSON at: $GraphJson"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($Seed in $RngSeeds)
{
    $OutputName = "VANILLA_PA__STEPS_${NSteps}__RNGSEED_${Seed}__TOL_${ToleranceLabel}.bendl"
    $OutputFile = Join-Path $OutputDir $OutputName

    Write-Host "Running rustrecom chain with seed: $Seed ..."

    & rustrecom chain `
        --assignment-col $AssignmentColumn `
        --graph-json $GraphJson `
        --n-steps $NSteps `
        --pop-col $PopulationColumn `
        --rng-seed $Seed `
        --tol $Tolerance `
        --variant district-pairs-mst `
        --writer bendl `
        --output-file $OutputFile `
        --overwrite-output `
        --show-progress

    if ($LASTEXITCODE -ne 0)
    {
        exit $LASTEXITCODE
    }
}
