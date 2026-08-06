#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# This installer is assembled from installer_src/skeleton.sh and template_project/.
# Run 'python3 generate_installers.py' after editing either source. Do not edit the
# generated template_maker.sh directly.
# ---------------------------------------------------------------------------

set -euo pipefail

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
            echo "Cargo is required to install the Rust command-line tools. Exiting."
            exit 1
        fi
    fi
}

# ==================================================
# ========  EMBEDDED PROJECT FILES  ================
# ==================================================

# {{GENERATED_PAYLOADS}}

# Writes every embedded project file into the current (project) directory.
function write_payload_files() {
    local d f
    for d in "${payload_directories[@]}"; do
        mkdir -p "$d"
    done
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

    read -p "Would you like to use RustReCom in this project? (y/[n]): " use_rustrecom
    if [[ "$use_rustrecom" == "y" || "$use_rustrecom" == "Y" ]]; then
        check_cargo_installed
        echo "Installing RustReCom (rustrecom, version 0.2.0)..."
        cargo install --git "https://github.com/mggg/rustrecom" --tag "v0.2.0" --locked --force
        echo "RustReCom has been installed."
    fi

    prompt="What python version would you like to use (3.11, 3.12, 3.13, 3.14)? (default: 3.11): "
    read -p "$prompt" python_version
    python_version="${python_version:-3.11}"  # if empty/unset, use 3.11
    case "$python_version" in
        3.11 | 3.12 | 3.13 | 3.14) ;;          # match = valid -> do nothing, then end this block
        *)                                     # anything else -> default
            echo "Invalid python version. Using default 3.11."
            python_version="3.11"
            ;;
    esac

    echo "Creating project: $project_name"
    if [[ -e "$project_name" ]]; then
        echo "A file or directory already exists at '$project_name'. Exiting."
        exit 1
    fi
    mkdir -- "$project_name"
    cd -- "$project_name"

    uv python install "$python_version"

    echo "Writing project files..."
    write_payload_files
    printf '%s\n' "$python_version" > .python-version
    sed "s/^requires-python = .*/requires-python = \">=$python_version\"/" \
        pyproject.toml > pyproject.toml.tmp
    mv pyproject.toml.tmp pyproject.toml

    echo "Installing the project environment with uv ($python_version)..."
    uv sync --python "$python_version"

    echo "Downloading PA geometry..."
    pa_url="https://raw.githubusercontent.com/mggg/democracy-batsignal"
    pa_url+="/13c1098d244df946263c9353a478ecf66ac8e484/template_project/data/pa_gdf.parquet"
    pa_sha256="06b3b927b09e3f049623869d0b15e20b1363a2eb4dad43461915382fc165446c"
    pa_tmp="$(mktemp)"
    if ! download_with_retries "$pa_url" "$pa_tmp"; then
        echo "Failed to download PA example data. Exiting."
        rm -f "$pa_tmp"
        exit 1
    fi
    actual_sha256="$(uv run python -c \
        'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
        "$pa_tmp")"
    if [[ "$actual_sha256" != "$pa_sha256" ]]; then
        echo "PA data checksum verification failed. Exiting."
        rm -f "$pa_tmp"
        exit 1
    fi
    mv "$pa_tmp" "data/pa_gdf.parquet"

    echo "Your project is ready! You may need to restart your shell for uv to work properly."
}

main "$@"
