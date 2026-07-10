#!/usr/bin/env bash

# This script converts a BEN file to an XBEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

# -c -1 lets the XZ encoder use every available core
find . -type f -name '*.ben' -exec ben xencode -v -w -c -1 {} \;
