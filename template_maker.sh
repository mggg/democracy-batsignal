#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# THIS FILE IS GENERATED from installer_src/skeleton.sh and template/.
# Edit those sources and run 'python3 generate_installers.py' instead of
# editing this script directly.
# ---------------------------------------------------------------------------

# ========================================
# ========  PRE-REQUISITE CHECKS  ========
# ========================================

function check_curl_installed() {
    if command -v curl &> /dev/null; then
        return 0
    fi

    echo "curl command not found. Please install it and re-run this script."

    case "$OSTYPE" in
        linux*)
            echo "You can usually install curl via your package manager."
            echo "For example, on Debian/Ubuntu: 'sudo apt-get install curl'"
            ;;
        darwin*)
            echo "curl normally ships with macOS; try 'xcode-select --install' or 'brew install curl'"
            ;;
        cygwin* | msys* | win32*)
            echo "You appear to be on Windows, consider using Git Bash or WSL which include curl."
            ;;
        *)
            echo "Please refer to your OS documentation for installing curl."
            ;;
    esac
    exit 1
}

function check_uv_installed() {
    if ! command -v uv &> /dev/null; then
        read -p "uv could not be found. Would you like to install it? (y/[n]): " choice
        if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
            # Use a disposable XDG config dir so the installer never touches user configs
            tmp_xdg="$(mktemp -d)"
            # On macOS/Linux/WSL/Git Bash this prevents the installer from writing ~/.config/fish/*
            (
                export XDG_CONFIG_HOME="$tmp_xdg"
                # Don't let a non-zero exit (e.g., shell integration step) kill our flow
                set +e
                curl -LsSf https://astral.sh/uv/install.sh | sh
                true
            )
            rm -rf "$tmp_xdg" 2> /dev/null || true

            # Ensure the common install location is on PATH and rehash
            export PATH="$HOME/.local/bin:$PATH"
            hash -r

            if ! command -v uv &> /dev/null; then
                echo "uv installation appears incomplete. Please install uv manually and re-run this script."
                echo "Docs: https://docs.astral.sh/uv/getting-started/installation/"
                exit 1
            fi
            echo "uv has been installed."
        else
            echo "uv is required to run this script. Exiting."
            exit 1
        fi
    fi
}

function check_cargo_installed() {
    if ! command -v cargo &> /dev/null; then
        read -p "Cargo could not be found. Would you like to install Rust and Cargo? (y/[n]): " choice
        if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
            case "$OSTYPE" in
                linux* | darwin*)
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
                    ;;
                *)
                    echo "Cannot install directly from script on this OS."
                    echo "Please install Rust and Cargo manually from https://www.rust-lang.org/tools/install and re-run this script."
                    ;;
            esac

            # Load cargo env if present (Unix), and ensure PATH for Windows shells
            if [[ -f "$HOME/.cargo/env" ]]; then
                source "$HOME/.cargo/env"
            fi
            export PATH="$HOME/.cargo/bin:$PATH"
            if command -v cygpath > /dev/null 2>&1; then
                win_cargo="$(cygpath -u "${USERPROFILE:-}")/.cargo/bin"
                export PATH="$win_cargo:$PATH"
            elif [[ -n "${USERPROFILE:-}" ]]; then
                export PATH="$USERPROFILE/.cargo/bin:$PATH"
            fi

            hash -r
            if ! command -v cargo &> /dev/null; then
                echo "Rust and Cargo installation failed. Please install them manually and re-run this script."
                exit 1
            fi
            echo "Rust and Cargo have been installed."
        else
            echo "Cargo is required to use FRCW or BEN. Exiting."
            exit 1
        fi
    fi
}

# ==================================================
# ========  EMBEDDED PROJECT FILES  ================
# ==================================================

# ====  GENERATED PAYLOADS (from template/) -- DO NOT EDIT BY HAND  ====
# ====  regenerate with: python3 generate_installers.py             ====

payload_files=(
    "JSON_dualgraphs/gerrymandria.json"
    "pipeline_scripts/example_cli.py"
    "pipeline_scripts/metrics/process_partisan_bias.py"
    "pipeline_scripts/metrics/process_polsby.py"
    "pipeline_scripts/metrics/process_reock.py"
    "pipeline_scripts/metrics/process_splits.py"
    "pipeline_scripts/metrics/process_total_dem_wins.py"
    "batch_example_python_cli_parallel.sh"
    "batch_example_python_cli_simple.sh"
    "chain_outputs/ben_to_xben.sh"
    "chain_outputs/jsonl_to_ben.sh"
    "pipeline_scripts/rust_example_script.sh"
)

function write_payload() {
    case "$1" in
    "JSON_dualgraphs/gerrymandria.json") cat << 'TEMPLATE_PAYLOAD_EOF'
{
    "directed": false,
    "multigraph": false,
    "graph": [],
    "nodes": [
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 0,
            "county": "1",
            "district": "1",
            "precinct": 0,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 0
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 1,
            "county": "1",
            "district": "1",
            "precinct": 1,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 1
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 2,
            "county": "1",
            "district": "1",
            "precinct": 2,
            "muni": "5",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 2
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 3,
            "county": "1",
            "district": "1",
            "precinct": 3,
            "muni": "5",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 3
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 4,
            "county": "3",
            "district": "1",
            "precinct": 4,
            "muni": "9",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 4
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 5,
            "county": "3",
            "district": "1",
            "precinct": 5,
            "muni": "9",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 5
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 6,
            "county": "3",
            "district": "1",
            "precinct": 6,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 6
        },
        {
            "TOTPOP": 1,
            "x": 0,
            "y": 7,
            "county": "3",
            "district": "1",
            "precinct": 7,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 7
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 0,
            "county": "1",
            "district": "2",
            "precinct": 8,
            "muni": "1",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "2",
            "id": 8
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 1,
            "county": "1",
            "district": "2",
            "precinct": 9,
            "muni": "1",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 9
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 2,
            "county": "1",
            "district": "2",
            "precinct": 10,
            "muni": "5",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 10
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 3,
            "county": "1",
            "district": "2",
            "precinct": 11,
            "muni": "5",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 11
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 4,
            "county": "3",
            "district": "2",
            "precinct": 12,
            "muni": "9",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 12
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 5,
            "county": "3",
            "district": "2",
            "precinct": 13,
            "muni": "9",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 13
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 6,
            "county": "3",
            "district": "2",
            "precinct": 14,
            "muni": "13",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 14
        },
        {
            "TOTPOP": 1,
            "x": 1,
            "y": 7,
            "county": "3",
            "district": "2",
            "precinct": 15,
            "muni": "13",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 15
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 0,
            "county": "1",
            "district": "3",
            "precinct": 16,
            "muni": "2",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 16
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 1,
            "county": "1",
            "district": "3",
            "precinct": 17,
            "muni": "2",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 17
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 2,
            "county": "1",
            "district": "3",
            "precinct": 18,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 18
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 3,
            "county": "1",
            "district": "3",
            "precinct": 19,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "2",
            "id": 19
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 4,
            "county": "3",
            "district": "3",
            "precinct": 20,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 20
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 5,
            "county": "3",
            "district": "3",
            "precinct": 21,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 21
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 6,
            "county": "3",
            "district": "3",
            "precinct": 22,
            "muni": "14",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 22
        },
        {
            "TOTPOP": 1,
            "x": 2,
            "y": 7,
            "county": "3",
            "district": "3",
            "precinct": 23,
            "muni": "14",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 23
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 0,
            "county": "1",
            "district": "4",
            "precinct": 24,
            "muni": "2",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "4",
            "id": 24
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 1,
            "county": "1",
            "district": "4",
            "precinct": 25,
            "muni": "2",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 25
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 2,
            "county": "1",
            "district": "4",
            "precinct": 26,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 26
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 3,
            "county": "1",
            "district": "4",
            "precinct": 27,
            "muni": "6",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 27
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 4,
            "county": "3",
            "district": "4",
            "precinct": 28,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 28
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 5,
            "county": "3",
            "district": "4",
            "precinct": 29,
            "muni": "10",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 29
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 6,
            "county": "3",
            "district": "4",
            "precinct": 30,
            "muni": "14",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 30
        },
        {
            "TOTPOP": 1,
            "x": 3,
            "y": 7,
            "county": "3",
            "district": "4",
            "precinct": 31,
            "muni": "14",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 31
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 0,
            "county": "2",
            "district": "5",
            "precinct": 32,
            "muni": "3",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 32
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 1,
            "county": "2",
            "district": "5",
            "precinct": 33,
            "muni": "3",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 33
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 2,
            "county": "2",
            "district": "5",
            "precinct": 34,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 34
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 3,
            "county": "2",
            "district": "5",
            "precinct": 35,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 35
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 4,
            "county": "4",
            "district": "5",
            "precinct": 36,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 36
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 5,
            "county": "4",
            "district": "5",
            "precinct": 37,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 37
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 6,
            "county": "4",
            "district": "5",
            "precinct": 38,
            "muni": "15",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 38
        },
        {
            "TOTPOP": 1,
            "x": 4,
            "y": 7,
            "county": "4",
            "district": "5",
            "precinct": 39,
            "muni": "15",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 39
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 0,
            "county": "2",
            "district": "6",
            "precinct": 40,
            "muni": "3",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 40
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 1,
            "county": "2",
            "district": "6",
            "precinct": 41,
            "muni": "3",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 41
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 2,
            "county": "2",
            "district": "6",
            "precinct": 42,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 42
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 3,
            "county": "2",
            "district": "6",
            "precinct": 43,
            "muni": "7",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "4",
            "id": 43
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 4,
            "county": "4",
            "district": "6",
            "precinct": 44,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 44
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 5,
            "county": "4",
            "district": "6",
            "precinct": 45,
            "muni": "11",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 45
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 6,
            "county": "4",
            "district": "6",
            "precinct": 46,
            "muni": "15",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 46
        },
        {
            "TOTPOP": 1,
            "x": 5,
            "y": 7,
            "county": "4",
            "district": "6",
            "precinct": 47,
            "muni": "15",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 47
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 0,
            "county": "2",
            "district": "7",
            "precinct": 48,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 48
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 1,
            "county": "2",
            "district": "7",
            "precinct": 49,
            "muni": "4",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 49
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 2,
            "county": "2",
            "district": "7",
            "precinct": 50,
            "muni": "8",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 50
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 3,
            "county": "2",
            "district": "7",
            "precinct": 51,
            "muni": "8",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 51
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 4,
            "county": "4",
            "district": "7",
            "precinct": 52,
            "muni": "12",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "3",
            "id": 52
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 5,
            "county": "4",
            "district": "7",
            "precinct": 53,
            "muni": "12",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 53
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 6,
            "county": "4",
            "district": "7",
            "precinct": 54,
            "muni": "16",
            "boundary_node": false,
            "boundary_perim": 0,
            "water_dist": "1",
            "id": 54
        },
        {
            "TOTPOP": 1,
            "x": 6,
            "y": 7,
            "county": "4",
            "district": "7",
            "precinct": 55,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 55
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 0,
            "county": "2",
            "district": "8",
            "precinct": 56,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 56
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 1,
            "county": "2",
            "district": "8",
            "precinct": 57,
            "muni": "4",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 57
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 2,
            "county": "2",
            "district": "8",
            "precinct": 58,
            "muni": "8",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 58
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 3,
            "county": "2",
            "district": "8",
            "precinct": 59,
            "muni": "8",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 59
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 4,
            "county": "4",
            "district": "8",
            "precinct": 60,
            "muni": "12",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "3",
            "id": 60
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 5,
            "county": "4",
            "district": "8",
            "precinct": 61,
            "muni": "12",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 61
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 6,
            "county": "4",
            "district": "8",
            "precinct": 62,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 62
        },
        {
            "TOTPOP": 1,
            "x": 7,
            "y": 7,
            "county": "4",
            "district": "8",
            "precinct": 63,
            "muni": "16",
            "boundary_node": true,
            "boundary_perim": 1,
            "water_dist": "1",
            "id": 63
        }
    ],
    "adjacency": [
        [
            {
                "id": 8
            },
            {
                "id": 1
            }
        ],
        [
            {
                "id": 0
            },
            {
                "id": 9
            },
            {
                "id": 2
            }
        ],
        [
            {
                "id": 1
            },
            {
                "id": 10
            },
            {
                "id": 3
            }
        ],
        [
            {
                "id": 2
            },
            {
                "id": 11
            },
            {
                "id": 4
            }
        ],
        [
            {
                "id": 3
            },
            {
                "id": 12
            },
            {
                "id": 5
            }
        ],
        [
            {
                "id": 4
            },
            {
                "id": 13
            },
            {
                "id": 6
            }
        ],
        [
            {
                "id": 5
            },
            {
                "id": 14
            },
            {
                "id": 7
            }
        ],
        [
            {
                "id": 6
            },
            {
                "id": 15
            }
        ],
        [
            {
                "id": 0
            },
            {
                "id": 16
            },
            {
                "id": 9
            }
        ],
        [
            {
                "id": 1
            },
            {
                "id": 8
            },
            {
                "id": 17
            },
            {
                "id": 10
            }
        ],
        [
            {
                "id": 2
            },
            {
                "id": 9
            },
            {
                "id": 18
            },
            {
                "id": 11
            }
        ],
        [
            {
                "id": 3
            },
            {
                "id": 10
            },
            {
                "id": 19
            },
            {
                "id": 12
            }
        ],
        [
            {
                "id": 4
            },
            {
                "id": 11
            },
            {
                "id": 20
            },
            {
                "id": 13
            }
        ],
        [
            {
                "id": 5
            },
            {
                "id": 12
            },
            {
                "id": 21
            },
            {
                "id": 14
            }
        ],
        [
            {
                "id": 6
            },
            {
                "id": 13
            },
            {
                "id": 22
            },
            {
                "id": 15
            }
        ],
        [
            {
                "id": 7
            },
            {
                "id": 14
            },
            {
                "id": 23
            }
        ],
        [
            {
                "id": 8
            },
            {
                "id": 24
            },
            {
                "id": 17
            }
        ],
        [
            {
                "id": 9
            },
            {
                "id": 16
            },
            {
                "id": 25
            },
            {
                "id": 18
            }
        ],
        [
            {
                "id": 10
            },
            {
                "id": 17
            },
            {
                "id": 26
            },
            {
                "id": 19
            }
        ],
        [
            {
                "id": 11
            },
            {
                "id": 18
            },
            {
                "id": 27
            },
            {
                "id": 20
            }
        ],
        [
            {
                "id": 12
            },
            {
                "id": 19
            },
            {
                "id": 28
            },
            {
                "id": 21
            }
        ],
        [
            {
                "id": 13
            },
            {
                "id": 20
            },
            {
                "id": 29
            },
            {
                "id": 22
            }
        ],
        [
            {
                "id": 14
            },
            {
                "id": 21
            },
            {
                "id": 30
            },
            {
                "id": 23
            }
        ],
        [
            {
                "id": 15
            },
            {
                "id": 22
            },
            {
                "id": 31
            }
        ],
        [
            {
                "id": 16
            },
            {
                "id": 32
            },
            {
                "id": 25
            }
        ],
        [
            {
                "id": 17
            },
            {
                "id": 24
            },
            {
                "id": 33
            },
            {
                "id": 26
            }
        ],
        [
            {
                "id": 18
            },
            {
                "id": 25
            },
            {
                "id": 34
            },
            {
                "id": 27
            }
        ],
        [
            {
                "id": 19
            },
            {
                "id": 26
            },
            {
                "id": 35
            },
            {
                "id": 28
            }
        ],
        [
            {
                "id": 20
            },
            {
                "id": 27
            },
            {
                "id": 36
            },
            {
                "id": 29
            }
        ],
        [
            {
                "id": 21
            },
            {
                "id": 28
            },
            {
                "id": 37
            },
            {
                "id": 30
            }
        ],
        [
            {
                "id": 22
            },
            {
                "id": 29
            },
            {
                "id": 38
            },
            {
                "id": 31
            }
        ],
        [
            {
                "id": 23
            },
            {
                "id": 30
            },
            {
                "id": 39
            }
        ],
        [
            {
                "id": 24
            },
            {
                "id": 40
            },
            {
                "id": 33
            }
        ],
        [
            {
                "id": 25
            },
            {
                "id": 32
            },
            {
                "id": 41
            },
            {
                "id": 34
            }
        ],
        [
            {
                "id": 26
            },
            {
                "id": 33
            },
            {
                "id": 42
            },
            {
                "id": 35
            }
        ],
        [
            {
                "id": 27
            },
            {
                "id": 34
            },
            {
                "id": 43
            },
            {
                "id": 36
            }
        ],
        [
            {
                "id": 28
            },
            {
                "id": 35
            },
            {
                "id": 44
            },
            {
                "id": 37
            }
        ],
        [
            {
                "id": 29
            },
            {
                "id": 36
            },
            {
                "id": 45
            },
            {
                "id": 38
            }
        ],
        [
            {
                "id": 30
            },
            {
                "id": 37
            },
            {
                "id": 46
            },
            {
                "id": 39
            }
        ],
        [
            {
                "id": 31
            },
            {
                "id": 38
            },
            {
                "id": 47
            }
        ],
        [
            {
                "id": 32
            },
            {
                "id": 48
            },
            {
                "id": 41
            }
        ],
        [
            {
                "id": 33
            },
            {
                "id": 40
            },
            {
                "id": 49
            },
            {
                "id": 42
            }
        ],
        [
            {
                "id": 34
            },
            {
                "id": 41
            },
            {
                "id": 50
            },
            {
                "id": 43
            }
        ],
        [
            {
                "id": 35
            },
            {
                "id": 42
            },
            {
                "id": 51
            },
            {
                "id": 44
            }
        ],
        [
            {
                "id": 36
            },
            {
                "id": 43
            },
            {
                "id": 52
            },
            {
                "id": 45
            }
        ],
        [
            {
                "id": 37
            },
            {
                "id": 44
            },
            {
                "id": 53
            },
            {
                "id": 46
            }
        ],
        [
            {
                "id": 38
            },
            {
                "id": 45
            },
            {
                "id": 54
            },
            {
                "id": 47
            }
        ],
        [
            {
                "id": 39
            },
            {
                "id": 46
            },
            {
                "id": 55
            }
        ],
        [
            {
                "id": 40
            },
            {
                "id": 56
            },
            {
                "id": 49
            }
        ],
        [
            {
                "id": 41
            },
            {
                "id": 48
            },
            {
                "id": 57
            },
            {
                "id": 50
            }
        ],
        [
            {
                "id": 42
            },
            {
                "id": 49
            },
            {
                "id": 58
            },
            {
                "id": 51
            }
        ],
        [
            {
                "id": 43
            },
            {
                "id": 50
            },
            {
                "id": 59
            },
            {
                "id": 52
            }
        ],
        [
            {
                "id": 44
            },
            {
                "id": 51
            },
            {
                "id": 60
            },
            {
                "id": 53
            }
        ],
        [
            {
                "id": 45
            },
            {
                "id": 52
            },
            {
                "id": 61
            },
            {
                "id": 54
            }
        ],
        [
            {
                "id": 46
            },
            {
                "id": 53
            },
            {
                "id": 62
            },
            {
                "id": 55
            }
        ],
        [
            {
                "id": 47
            },
            {
                "id": 54
            },
            {
                "id": 63
            }
        ],
        [
            {
                "id": 48
            },
            {
                "id": 57
            }
        ],
        [
            {
                "id": 49
            },
            {
                "id": 56
            },
            {
                "id": 58
            }
        ],
        [
            {
                "id": 50
            },
            {
                "id": 57
            },
            {
                "id": 59
            }
        ],
        [
            {
                "id": 51
            },
            {
                "id": 58
            },
            {
                "id": 60
            }
        ],
        [
            {
                "id": 52
            },
            {
                "id": 59
            },
            {
                "id": 61
            }
        ],
        [
            {
                "id": 53
            },
            {
                "id": 60
            },
            {
                "id": 62
            }
        ],
        [
            {
                "id": 54
            },
            {
                "id": 61
            },
            {
                "id": 63
            }
        ],
        [
            {
                "id": 55
            },
            {
                "id": 62
            }
        ]
    ]
}
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/example_cli.py") cat << 'TEMPLATE_PAYLOAD_EOF'
from gerrychain import Graph, Partition, MarkovChain
from gerrychain.updaters import Tally
from gerrychain.accept import always_accept
from gerrychain.proposals.tree_proposals import recom
from functools import partial
import random
import jsonlines as jl
import click
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenEncoder
import sys


@click.command()
@click.option("--graph-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-path", type=click.Path(writable=True, dir_okay=False))
@click.option("--starting-plan", type=str)
@click.option("--pop-col", type=str)
@click.option("--rng-seed", type=int)
@click.option("--population-tolerance", type=float, default=0.01)
@click.option("--total-steps", type=int, default=10_000)
@click.option("--writeas", type=click.Choice(["jsonl", "ben"]), default="ben")
def main(
    graph_path,
    output_path,
    starting_plan,
    pop_col,
    rng_seed,
    population_tolerance,
    total_steps,
    writeas,
):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    try:
        if graph_path.endswith(".json"):
            graph = Graph.from_json(graph_path)
        else:
            graph = Graph.from_file(graph_path)
    except Exception as e:
        raise ValueError(f"Failed to load graph from {graph_path}: {e}")

    initial_partition = Partition(
        graph,
        assignment=starting_plan,
        updaters={"population": Tally(pop_col, alias="population")},
    )

    ideal_pop = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(
        recom,
        pop_col=pop_col,
        pop_target=ideal_pop,
        epsilon=population_tolerance,
        node_repeats=1,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[],
        initial_state=initial_partition,
        total_steps=total_steps,
        accept=always_accept,
    )

    graph_node_order = list(graph.nodes)

    # This will print to the standard error stream so that logging does not interfere with the
    # standard output.
    print(
        f"Writing output to '{Path(output_path).name}' in '{writeas.upper()}' format.",
        file=sys.stderr,
        flush=True,
    )
    match writeas:
        case "jsonl":
            with jl.open(output_path, "w") as writer:
                for i, partition in enumerate(chain.with_progress_bar()):
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    writer.write(
                        {
                            "assignment": ordered_assignment,
                            "sample": i + 1,
                        }
                    )

        case "ben":
            with BenEncoder(output_path, overwrite=True) as encoder:
                for partition in chain.with_progress_bar():
                    assignment_series = partition.assignment.to_series()
                    ordered_assignment = (
                        assignment_series.loc[graph_node_order].astype(int).to_list()
                    )
                    encoder.write(ordered_assignment)

        case _:
            raise ValueError(f"Unsupported writeas format: {writeas}")


if __name__ == "__main__":
    main()
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/metrics/process_partisan_bias.py") cat << 'TEMPLATE_PAYLOAD_EOF'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from pathlib import Path
import geopandas as gpd
import numpy as np
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, vote_arrays):
    assign = np.asarray(assignment_vector, dtype=np.int64)

    k = int(assign.max())  # number of districts

    out = {"sample": sample_idx, "pb_scores": {}}

    for dem_votes, rep_votes, name in vote_arrays:
        # assignments are 1-indexed, so drop bin 0 to keep the phantom empty
        # district out of mean_share
        dem_tot = np.bincount(assign, weights=dem_votes, minlength=k + 1)[1:]
        rep_tot = np.bincount(assign, weights=rep_votes, minlength=k + 1)[1:]

        total = dem_tot + rep_tot
        dem_share = np.divide(
            dem_tot, total, out=np.zeros_like(dem_tot, dtype="float64"), where=total > 0
        )

        mean_share = dem_share.mean()
        pb = (dem_share > mean_share).sum() / k - 0.5
        out["pb_scores"][name] = float(pb)

    return out


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_partisan_bias_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    gdf = gpd.read_file(GRAPH_PATH)

    elections = ["PRES16", "SSEN16"]
    election_pairs = [(f"{name}D", f"{name}R") for name in elections]

    # grab vote columns as numpy arrays once
    vote_arrays = []
    for d_col, r_col in election_pairs:
        vote_arrays.append(
            (
                gdf[d_col].to_numpy(dtype="float64", copy=True),
                gdf[r_col].to_numpy(dtype="float64", copy=True),
                d_col[:-1],
            )
        )
    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing partisan bias (batch {batch_no+1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, vote_arrays) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/metrics/process_polsby.py") cat << 'TEMPLATE_PAYLOAD_EOF'
import jsonlines as jl
from gerrychain import GeographicPartition, Graph
from gerrychain.metrics import polsby_popper
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_number, assignment_vector, graph):
    part = GeographicPartition(
        graph, assignment={i: val for i, val in enumerate(assignment_vector)}
    )
    return {"sample": sample_number, "scores": polsby_popper(part)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_polsby_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    np.random.seed(42)  # seed so the subsample is reproducible
    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    graph = Graph.from_file(GRAPH_PATH)

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Polsby-Popper scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, graph) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/metrics/process_reock.py") cat << 'TEMPLATE_PAYLOAD_EOF'
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import geopandas as gpd
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, geo_only):
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="pygeos support was removed in 1.0. geopandas.use_pygeos is a no-op",
        category=UserWarning,
    )

    from gerrytools.scoring import reock

    geo_new = geo_only.copy()
    geo_new["assignment"] = np.array(assignment_vector) - 1
    dissolved = geo_new.dissolve(by="assignment")
    return {"sample": sample_idx, "scores": reock().apply(dissolved)}


if __name__ == "__main__":
    batch_size = 1000
    n_samples = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_reock_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

    np.random.seed(42)  # seed so the subsample is reproducible
    subsamples = sorted(
        map(
            int, np.random.choice(total_chain_length, size=n_samples, replace=False) + 1
        )
    )  # +1 for 1-based indexing

    gdf = gpd.read_file(GRAPH_PATH)
    geo_only = gdf[["geometry"]]

    all_scores = []
    n_batches = (len(subsamples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = subsamples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all Reock scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(delayed(compute_score)(idx, vec, geo_only) for idx, vec in pairs)

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/metrics/process_splits.py") cat << 'TEMPLATE_PAYLOAD_EOF'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    u,
    v,
    same_county_mask,
    county_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    cut = assignment[u] != assignment[v]
    cut_edges = int(cut.sum())
    county_splits = len(set(county_names[cut & same_county_mask]))
    return {
        "sample": sample_idx,
        "scores": {
            "county_splits": county_splits,
            "cut_edges": cut_edges,
        },
    }


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_split_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    graph = Graph.from_file(GRAPH_PATH)
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    u = edges[:, 0]
    v = edges[:, 1]

    # Masks that depend only on the graph (not on assignments)
    # A "split" counts only when it's a cut edge AND both endpoints share the same county/place.
    same_county_mask = np.fromiter(
        (
            graph.nodes[int(a)]["COUNTYNAME"] == graph.nodes[int(b)]["COUNTYNAME"]
            for a, b in edges
        ),
        dtype=bool,
        count=len(edges),
    )

    # Don't need to record b since we are going to filter to when it has the same value as a
    county_names = np.fromiter(
        (graph.nodes[int(a)]["COUNTYNAME"] for a, _ in edges),
        dtype=object,
        count=len(edges),
    )

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing all split scores in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(idx, vec, u, v, same_county_mask, county_names)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/metrics/process_total_dem_wins.py") cat << 'TEMPLATE_PAYLOAD_EOF'
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import geopandas as gpd
from pathlib import Path
from binary_ensemble.stream import BenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    dem_count_matrix,
    rep_count_matrix,
    race_names,
):
    assignment = np.asarray(assignment_vector, dtype=np.int32)
    race_totals = {name: 0 for name in race_names}

    for part in np.unique(assignment):
        mask = assignment == part
        dem_totals = dem_count_matrix[mask].sum(axis=0)
        rep_totals = rep_count_matrix[mask].sum(axis=0)
        dem_wins = dem_totals > rep_totals

        for i, race in enumerate(race_names):
            race_totals[race] += 1 if dem_wins[i] else 0

    return {"sample": sample_idx, "scores": race_totals}


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_dem_win_scores.jsonl"

    decoder = BenDecoder(CHAIN_FILE)
    n_samples = len(decoder)
    samples = list(range(1, n_samples + 1))

    df = gpd.read_file(GRAPH_PATH)
    graph = Graph.from_geodataframe(df)
    df.drop(columns=["geometry"], inplace=True)

    race_names = [
        "PRES16",
        "SSEN16",
    ]

    dem_rep_pairs = [(f"{name}D", f"{name}R") for name in race_names]
    dem_count_matrix = df[[pair[0] for pair in dem_rep_pairs]].to_numpy()
    rep_count_matrix = df[[pair[1] for pair in dem_rep_pairs]].to_numpy()

    all_scores = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_no in range(n_batches):
        current_batch = samples[batch_no * batch_size : (batch_no + 1) * batch_size]
        vectors = list(decoder.subsample_indices(current_batch))

        pairs = list(zip(current_batch, vectors))

        with joblib_progress(
            description=f"Computing total dem wins in parallel (batch {batch_no + 1}/{n_batches})",
            total=len(pairs),
        ):
            scores = Parallel(
                n_jobs=os.cpu_count() or 1,
            )(
                delayed(compute_score)(
                    idx, vec, dem_count_matrix, rep_count_matrix, race_names
                )
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
TEMPLATE_PAYLOAD_EOF
        ;;
    "batch_example_python_cli_parallel.sh") cat << 'TEMPLATE_PAYLOAD_EOF'
#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Change this as needed to get the top level directory of the repo
TOPDIR="${SCRIPT_DIR}"

mkdir -p "${TOPDIR}/chain_outputs" "${TOPDIR}/chain_logs"

export PYTHONHASHSEED=0
# source .env # <- This will also work

# ===================================================================
#   IGNORE THE FOLLOWING SECTION. IT JUST HELPS TO MANAGE RESOURCES
# ===================================================================
function count_cores() {
    if command -v nproc > /dev/null 2>&1; then
                                            nproc
    elif [[ "${OSTYPE:-}" == darwin* ]]; then
                                            sysctl -n hw.ncpu
    else echo 1; fi
}

_spinner_pid=""
function spinner_start() {
    [ -t 1 ] || return 0
    local msg="$*"
    command -v tput > /dev/null && tput civis || true
    (   
        local sp='-\|/' i=0
        while :; do
            printf "\r[%c] %s" "${sp:i++%4:1}" "$msg"
            sleep 0.1
        done
    ) &
      _spinner_pid=$!
}
function spinner_stop() {
    [ -n "${_spinner_pid:-}" ] || return 0
    kill "$_spinner_pid" 2> /dev/null || true
    wait "$_spinner_pid" 2> /dev/null || true
    _spinner_pid=""
    if [ -t 1 ] && command -v tput > /dev/null; then tput cnorm; fi
    printf "\r%*s\r" "$(tput cols 2> /dev/null || echo 80)" ""
}

declare -a pids=()

function prune_pids() {
    local live=() pid
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2> /dev/null && live+=("$pid")
    done
    pids=("${live[@]}")
}

function running_count() {
    prune_pids
    echo "${#pids[@]}"
}

function cleanup() {
    # stop spinner, forward INT/TERM to children, reap
    trap - INT TERM EXIT
    spinner_stop
    # kill whole process group to be extra sure:
    kill -- -$$ 2> /dev/null || true
    # also try direct PIDs we tracked
    ((${#pids[@]})) && kill -INT "${pids[@]}" 2> /dev/null || true
    wait 2> /dev/null || true
}
# Register cleanup function to be called on the EXIT signal
trap cleanup INT TERM EXIT
# ===============================================================
# ===============================================================

# Edit this to change the number of parallel jobs if you want
MAX_JOBS=$(count_cores)

rng_seeds=({1..50})
n_steps=1000

function start_job() {
    local seed=$1  # rng seed is the first positional argument
    local n_steps=$2 # number of steps is the second positional argument
    uv run "${TOPDIR}/pipeline_scripts/example_cli.py" \
        --graph-path "${TOPDIR}/JSON_dualgraphs/gerrymandria.json" \
        --output-path "${TOPDIR}/chain_outputs/gerrymandria_chain_${n_steps}_steps_seed${seed}.jsonl" \
        --starting-plan "district" \
        --pop-col "TOTPOP" \
        --rng-seed "$seed" \
        --population-tolerance 0.01 \
        --total-steps "$n_steps" \
        --writeas "jsonl" > "${TOPDIR}/chain_logs/log_parallel_rng_seed_$seed.log" 2>&1 &
    pids+=("$!")
}

# Launch with a simple concurrency gate
for seed in "${rng_seeds[@]}"; do
    # If we already have MAX_JOBS running, wait for one to finish
    while (($(running_count) >= MAX_JOBS)); do
        # show a spinner while we're blocked waiting
        spinner_start "Waiting for a free slot: $(jobs -pr | wc -l)/$MAX_JOBS running..."
        if wait -n 2> /dev/null; then
            :
        else
            # fallback: wait on the oldest tracked PID, then drop it
            if ((${#pids[@]})); then
                wait "${pids[0]}" 2> /dev/null || true
                pids=("${pids[@]:1}")
            else
                wait -p _ 2> /dev/null || true
            fi
        fi
        spinner_stop
        prune_pids
    done
    start_job "$seed" "$n_steps"
done

if (($(running_count) > 0)); then
    spinner_start "Finishing remaining jobs..."
    wait "${pids[@]}" 2> /dev/null || true
    spinner_stop
fi
TEMPLATE_PAYLOAD_EOF
        ;;
    "batch_example_python_cli_simple.sh") cat << 'TEMPLATE_PAYLOAD_EOF'
#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Change this as needed to get the top level directory of the repo
TOPDIR="${SCRIPT_DIR}"

mkdir -p "${TOPDIR}/chain_outputs" "${TOPDIR}/chain_logs"

export PYTHONHASHSEED=0
# source .env # <- This will also work

rng_seeds=(42 43 44)
n_steps=1000

for seed in "${rng_seeds[@]}"; do
    uv run "${TOPDIR}/pipeline_scripts/example_cli.py" \
        --graph-path "${TOPDIR}/JSON_dualgraphs/gerrymandria.json" \
        --output-path "${TOPDIR}/chain_outputs/gerrymandria_chain_${n_steps}_steps_seed${seed}.jsonl" \
        --starting-plan "district" \
        --pop-col "TOTPOP" \
        --rng-seed $seed \
        --population-tolerance 0.01 \
        --total-steps $n_steps \
        --writeas "jsonl" > "${TOPDIR}/chain_logs/log_simple_rng_seed_$seed.log" 2>&1
done


rng_seeds=(42)
n_steps=100000

for seed in "${rng_seeds[@]}"; do
    uv run "${TOPDIR}/pipeline_scripts/example_cli.py" \
        --graph-path "${TOPDIR}/JSON_dualgraphs/MN_precincts.geojson" \
        --output-path "${TOPDIR}/chain_outputs/MN_chain_${n_steps}_steps_seed${seed}.jsonl.ben" \
        --starting-plan "CONGDIST" \
        --pop-col "TOTPOP" \
        --rng-seed $seed \
        --population-tolerance 0.05 \
        --total-steps $n_steps \
        --writeas "ben"
done
TEMPLATE_PAYLOAD_EOF
        ;;
    "chain_outputs/ben_to_xben.sh") cat << 'TEMPLATE_PAYLOAD_EOF'
#!/usr/bin/env bash

# This script converts every BEN file next to it to an XBEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

# -c -1 lets the XZ encoder use every available core
find "${SCRIPT_DIR}" -type f -name '*.ben' -exec ben xencode -v -w -c -1 {} \;
TEMPLATE_PAYLOAD_EOF
        ;;
    "chain_outputs/jsonl_to_ben.sh") cat << 'TEMPLATE_PAYLOAD_EOF'
#!/usr/bin/env bash

# This script converts every JSONL file next to it to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

find "${SCRIPT_DIR}" -type f -name '*.jsonl' -exec ben encode -v -w {} \;
TEMPLATE_PAYLOAD_EOF
        ;;
    "pipeline_scripts/rust_example_script.sh") cat << 'TEMPLATE_PAYLOAD_EOF'
#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Project root is the parent of this script's folder
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)

plan_name="district"
n_steps=1000
seed=42
tol=0.01
pop_col="TOTPOP"

json_file="${PROJECT_ROOT}/JSON_dualgraphs/gerrymandria.json"
output_dir="${PROJECT_ROOT}/chain_outputs"

if [[ ! -f "$json_file" ]]; then
    echo "Could not find graph JSON at: $json_file" >&2
    exit 1
fi

mkdir -p "$output_dir"
final_output_file="${output_dir}/gerrymandria_chain_${n_steps}_steps.jsonl.ben"

frcw \
    --assignment-col $plan_name \
    --graph-json "$json_file" \
    --n-steps $n_steps \
    --pop-col $pop_col \
    --rng-seed $seed \
    --tol $tol \
    --variant district-pairs-rmst \
    --writer ben \
    --batch-size 1 \
    --n-threads 1 \
    --output-file "${final_output_file}"
TEMPLATE_PAYLOAD_EOF
        ;;
    esac
}

# Writes every embedded project file into the current (project) directory.
function write_payload_files() {
    local f
    for f in "${payload_files[@]}"; do
        mkdir -p "$(dirname "$f")"
        write_payload "$f" > "$f"
        if [[ "$f" == *.sh ]]; then
            chmod +x "$f"
        fi
    done
}

function download_with_retries() {
    local url="$1" dest="$2"
    curl -fLsS \
        --retry 5 \
        --retry-delay 2 \
        --retry-max-time 300 \
        --retry-all-errors \
        -o "$dest" \
        "$url"
}

# ==============================================
# ========  MAIN INSTALLATION FUNCTION  ========
# ==============================================

function main() {
    check_curl_installed
    check_uv_installed

    read -p "Enter the name of the new project to create: " project_name
    if [[ -z "$project_name" ]]; then
        project_name="my_project"
        echo "No project name provided. Using default name: $project_name"
    fi

    read -p "Would you like to use FRCW in this project? (y/[n]): " use_frcw
    if [[ "$use_frcw" == "y" || "$use_frcw" == "Y" ]]; then
        check_cargo_installed
        echo "Installing FRCW (rustrecom, branch 0.1.4)..."
        cargo install --git "https://github.com/mggg/rustrecom" --branch "0.1.4" --force
        echo "FRCW has been installed."

        echo "Installing binary-ensemble"
        cargo install binary-ensemble --force
        echo "binary-ensemble has been installed."

        echo "Installing ben-process (metrics engine)"
        cargo install --git "https://github.com/peterrrock2/ben-process" --force
        echo "ben-process has been installed."
    else
        read -p "Would you like to use BEN in this project? (y/[n]): " use_ben
        if [[ "$use_ben" == "y" || "$use_ben" == "Y" ]]; then
            check_cargo_installed
            echo "Installing binary-ensemble"
            cargo install binary-ensemble --force
            echo "binary-ensemble has been installed."

            echo "Installing ben-process (metrics engine)"
            cargo install --git "https://github.com/peterrrock2/ben-process" --force
            echo "ben-process has been installed."
        fi
    fi

    read -p "What python version would you like to use (3.11, 3.12, 3.13)? (default: 3.11): " python_version
    python_version="${python_version:-3.11}"  # if empty/unset, use 3.11
    case "$python_version" in
        3.11 | 3.12 | 3.13) ;;               # match = valid -> do nothing, then end this block
        *)                                     # anything else -> default
            echo "Invalid python version. Using default 3.11."
            python_version="3.11"
            ;;
    esac

    echo "Creating project: $project_name"
    mkdir -p "$project_name"
    cd "$project_name" || exit

    uv python install "$python_version"

    uv init --python "$python_version"

    echo "Project $project_name has been created and initialized with uv ($python_version)."
    echo "Adding standard packages to pyproject.toml..."

    # Get rid of some of the default files
    rm -f "README.md" "main.py"

    uv add numpy pandas matplotlib seaborn "gerrychain[geo]" maup ipykernel \
        ipywidgets click gerrytools "binary-ensemble>=1.0" jsonlines joblib \
        joblib-progress docker

    # A formatter that I like
    uv add --dev black

    mkdir -p "data"
    mkdir -p "JSON_dualgraphs"
    mkdir -p "notebooks"
    mkdir -p "pipeline_scripts"
    mkdir -p "figures"
    mkdir -p "stats"
    mkdir -p "chain_outputs"
    mkdir -p "chain_logs"
    mkdir -p "dev_files"

    echo "dev_files" >> .gitignore

    # NOTE: Needed to make python reproducible
    echo "export PYTHONHASHSEED=0" >> .env

    echo "Writing project files..."
    write_payload_files

    # Grab the MN example data. The zip is extracted with the project's Python so that
    # no unzip/bsdtar/tar is needed on the host.
    echo "Downloading MN example data..."
    mn_zip="$(mktemp)"
    if ! download_with_retries "https://github.com/mggg/GerryChain/raw/main/docs/_static/MN.zip" "$mn_zip"; then
        echo "Failed to download MN example data. Exiting."
        rm -f "$mn_zip"
        exit 1
    fi
    uv run python -m zipfile -e "$mn_zip" "JSON_dualgraphs"
    rm -f "$mn_zip"

    echo "Your project is ready! You may need to restart your shell for uv to work properly."
}

main "$@"
