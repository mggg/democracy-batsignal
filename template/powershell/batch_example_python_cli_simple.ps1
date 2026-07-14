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
    --total-steps   $TotalSteps `
    --writeas "jsonl" *> $logFile
}

foreach ($seed in $RngSeeds2) {
  $outFile = Join-Path (Join-Path $TOPDIR "chain_outputs") ("MN_chain_{0}_steps_seed{1}.jsonl.ben" -f $TotalSteps2, $seed)

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
