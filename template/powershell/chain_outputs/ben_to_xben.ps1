param([switch]$Recurse = $true)

# This script converts every BEN file next to it to an XBEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

$files = Get-ChildItem -Path $PSScriptRoot -File -Filter *.ben -Recurse:$Recurse
foreach ($f in $files) {
  Write-Host "Processing $($f.FullName)"
  # -c -1 lets the XZ encoder use every available core
  & ben xencode $f.FullName -v -w -c -1
}
