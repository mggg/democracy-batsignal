param([switch]$Recurse = $true)

# This script converts every JSONL file next to it to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

$files = Get-ChildItem -Path $PSScriptRoot -File -Filter *.jsonl -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  & ben encode $f.FullName -v -w
}
