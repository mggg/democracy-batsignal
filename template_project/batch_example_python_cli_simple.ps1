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
  $outFile = Join-Path $chainOut  "gerrymandria_chain_${TotalSteps}_steps_seed$seed.bendl"
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
  $outFile = Join-Path (Join-Path $TOPDIR "chain_outputs") ("PA_chain_{0}_steps_seed{1}.bendl" -f $TotalSteps2, $seed)

  & uv run (Join-Path $TOPDIR (Join-Path "pipeline_scripts" "example_cli.py")) `
    --graph-path   (Join-Path $TOPDIR (Join-Path "JSON_dualgraphs" "pa_dualgraph.json")) `
    --output-path  $outFile `
    --starting-plan "seed_plan" `
    --pop-col       "total_pop_20" `
    --rng-seed      $seed `
    --population-tolerance 0.01 `
    --total-steps   $TotalSteps2
}
