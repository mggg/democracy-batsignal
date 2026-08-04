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
