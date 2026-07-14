#!/usr/bin/env bash

# This script converts every JSONL file next to it to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

find "${SCRIPT_DIR}" -type f -name '*.jsonl' -exec ben encode -v -w {} \;
