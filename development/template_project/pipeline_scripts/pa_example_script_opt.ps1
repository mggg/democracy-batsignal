param(
    [int]$NSteps = 1000,
    [int[]]$RngSeeds = @(42, 43),
    [double]$Tolerance = 0.01,
    [string]$AssignmentColumn = 'seed_plan',
    [string]$PopulationColumn = 'total_pop_20'
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$GraphJson = Join-Path $ProjectRoot 'JSON_dualgraphs/pa_dualgraph.json'
$ObjectiveFile = Join-Path $ProjectRoot 'pipeline_scripts/rustrecom_objectives/gingles_partial.json'
$OutputDir = Join-Path $ProjectRoot 'chain_outputs'
$LogDir = Join-Path $ProjectRoot 'chain_logs'
$ToleranceLabel = $Tolerance.ToString([System.Globalization.CultureInfo]::InvariantCulture)

if (-not (Test-Path -LiteralPath $GraphJson -PathType Leaf))
{
    Write-Error "Could not find graph JSON at: $GraphJson"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir, $LogDir | Out-Null

foreach ($Seed in $RngSeeds)
{
    $Prefix = "GINGLES_PARTIAL_PA__STEPS_${NSteps}__RNGSEED_${Seed}__TOL_${ToleranceLabel}"
    $OutputFile = Join-Path $OutputDir "${Prefix}.bendl"

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
        --output-file $OutputFile `
        --overwrite-output `
        --show-progress

    if ($LASTEXITCODE -ne 0)
    {
        exit $LASTEXITCODE
    }
}
