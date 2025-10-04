#!/usr/bin/env bash

# This script converts a JSONL file to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

for f in $(find . -type f -name '*.ben'); do
    echo "Processing $f"
    ben -m x-encode $f -v -w
done
