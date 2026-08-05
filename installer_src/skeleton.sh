#!/usr/bin/env bash

# ---------------------------------------------------------------------------
# THIS FILE IS GENERATED from installer_src/skeleton.sh and template_project/.
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
            echo "Cargo is required to use RustReCom. Exiting."
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

    read -p "Would you like to use RustReCom in this project? (y/[n]): " use_frcw
    if [[ "$use_frcw" == "y" || "$use_frcw" == "Y" ]]; then
        check_cargo_installed
        echo "Installing RustReCom (rustrecom, version 0.2.0)..."
        cargo install --git "https://github.com/mggg/rustrecom" --tag "v0.2.0" --force
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
    mkdir -p "$project_name"
    cd "$project_name" || exit

    uv python install "$python_version"

    echo "Writing project files..."
    write_payload_files
    printf '%s\n' "$python_version" > .python-version
    sed "s/^requires-python = .*/requires-python = \">=$python_version\"/" \
        pyproject.toml > pyproject.toml.tmp
    mv pyproject.toml.tmp pyproject.toml

    echo "Installing the project environment with uv ($python_version)..."
    uv sync --python "$python_version"

    # Grab the PA example data. The zip is extracted with the project's Python so that
    # no unzip/bsdtar/tar is needed on the host.
    echo "Downloading PA example data..."
    pa_zip="$(mktemp)"
    pa_url="https://github.com/mggg-states/PA-shapefiles/raw/refs/heads/master/PA_2020_vtds.zip"
    if ! download_with_retries "$pa_url" "$pa_zip"; then
        echo "Failed to download PA example data. Exiting."
        rm -f "$pa_zip"
        exit 1
    fi
    uv run python -m zipfile -e "$pa_zip" "data"
    rm -f "$pa_zip"

    echo "Your project is ready! You may need to restart your shell for uv to work properly."
}

main "$@"
