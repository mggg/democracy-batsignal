#!/usr/bin/env bash

# This script converts a JSONL file to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

find . -type f -name '*.jsonl' -exec ben encode -v -w {} \;
