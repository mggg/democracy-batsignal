param(
  [string]$AssignmentColumn = 'seed_plan',
  [int]$NSteps = 1000,
  [int]$RngSeed = 42,
  [double]$Tolerance = 0.01,
  [string]$PopulationColumn = 'total_pop_20'
)

# Project root is the parent of this script's folder
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$jsonDir = Join-Path $ProjectRoot 'JSON_dualgraphs'
$outputDir = Join-Path $ProjectRoot 'chain_outputs'

$graphJson = Join-Path $jsonDir 'pa_dualgraph.json'

if (-not (Test-Path $graphJson)) {
  Write-Error "Could not find graph JSON at: $graphJson"
  exit 1
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$outputName = "PA__STEPS_${NSteps}__RNGSEED_${RngSeed}__TOL_${Tolerance}.bendl"
$outputFile = Join-Path $outputDir $outputName

& rustrecom chain `
  --assignment-col $AssignmentColumn `
  --graph-json $graphJson `
  --n-steps $NSteps `
  --pop-col $PopulationColumn `
  --rng-seed $RngSeed `
  --tol $Tolerance `
  --variant district-pairs-mst `
  --writer bendl `
  --output-file $outputFile `
  --overwrite-output
