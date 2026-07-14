#!/usr/bin/env bash

# This script converts every BEN file next to it to an XBEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

# -c -1 lets the XZ encoder use every available core
find "${SCRIPT_DIR}" -type f -name '*.ben' -exec ben xencode -v -w -c -1 {} \;
