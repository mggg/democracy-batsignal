# This is a direct RustReCom CLI reference. Use pipeline_scripts/run_chains.py for normal batches.
# If script execution is blocked, run: Set-ExecutionPolicy -Scope Process Bypass
# `tilted` runs ReCom while favoring proposals that improve the selected objective score.
# Input and chain flags have the same meaning as in the ordinary `chain` example.
# `--objective` loads the score definition, and `--maximize true` makes larger scores preferable.
# The BENDL file records plans; the companion CSV records objective values for analysis.

param(
    [int]$NSteps = 1000,
    [int[]]$RngSeeds = @(42, 43),
    [double]$Tolerance = 0.01,
    [string]$AssignmentColumn = 'seed_plan',
    [string]$PopulationColumn = 'total_pop_20'
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$GraphJson = Join-Path $ProjectRoot 'JSON_dualgraphs/pa_dualgraph.json'
$ObjectiveFile = Join-Path `
    $ProjectRoot 'pipeline_scripts/chain_runners/rustrecom_objectives/gingles_partial.json'
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
    $Prefix = "GINGLES_PARTIAL_PA__STEPS_${NSteps}__RNGSEED_${Seed}__TOL_${ToleranceLabel}"
    $BendlFile = Join-Path $OutputDir "${Prefix}.bendl"
    $ScoresFile = Join-Path $OutputDir "${Prefix}_scores.csv"

    Write-Host "Running rustrecom tilted with seed: $Seed ..."

    & rustrecom tilted `
        --assignment-col $AssignmentColumn `
        --graph-json $GraphJson `
        --n-steps $NSteps `
        --pop-col $PopulationColumn `
        --rng-seed $Seed `
        --tol $Tolerance `
        --objective $ObjectiveFile `
        --maximize true `
        --variant district-pairs-mst `
        --writer bendl `
        --output-file $BendlFile `
        --scores-output-file $ScoresFile `
        --overwrite-output `
        --show-progress

    if ($LASTEXITCODE -ne 0)
    {
        exit $LASTEXITCODE
    }
}
