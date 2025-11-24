#!/usr/bin/env bash

# ========================================
# ========  PRE-REQUISITE CHECKS  ========
# ========================================

function check_for_realpath() {
    if command -v realpath &> /dev/null; then
        return 0
    fi

    echo "realpath command not found. Please install it and re-run this script."

    case "$OSTYPE" in
        linux*)
            echo "You appear to be on Linux, you can usually install realpath via your package manager."
            echo "For example, on Debian/Ubuntu: 'sudo apt-get install coreutils'"
            ;;
        darwin*)
            echo "You appear to be on macOS, you can realpath via Homebrew: 'brew install coreutils'"
            ;;
        cygwin* | msys* | win32*)
            echo "You appear to be on Windows, consider using Git Bash or WSL which may include realpath."
            ;;
        *)
            echo "Please refer to your OS documentation for installing coreutils or equivalent."
            ;;
    esac
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

# =====================================
# ========  MAIN BASH SCRIPTS  ========
# =====================================

function batch_example_simple() {
    cat << "SH"
#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
# Change this as needed to get the top level directory of the repo
TOPDIR=$(realpath "${SCRIPT_DIR}")

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
        --writeas "jsonl" > "./chain_logs/log_simple_rng_seed_$seed.log" 2>&1
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
SH
}

function batch_example_parallel() {
    cat << "BASHSCRIPT"
#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
# Change this as needed to get the top level directory of the repo
TOPDIR=$(realpath "${SCRIPT_DIR}")

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
        --total-steps "$n_steps" > "./chain_logs/log_parallel_rng_seed_$seed.log" 2>&1 &
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
BASHSCRIPT
}

function rust_sh_example() {
    cat << "SH"
#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

plan_name="district"
n_steps=1000
seed=42
target_pop=8
tol=0.01
pop_col="TOTPOP"
json_dir="JSON_dualgraphs"
output_dir="chain_outputs"

json_file="$(realpath ./${json_dir}/gerrymandria.json)"
final_output_file="$(realpath ./${output_dir}/gerrymandria_chain_${n_steps}_steps.jsonl.ben)"

frcw \
    --assignment-col $plan_name \
    --graph-json $json_file \
    --n-steps $n_steps \
    --pop-col $pop_col \
    --rng-seed $seed \
    --tol $tol \
    --variant district-pairs-rmst \
    --writer ben \
    --batch-size 1 \
    --n-threads 1 \
    --output-file "${final_output_file}"
SH
}

function jsonl_to_ben_script() {
    cat << "SH"
#!/usr/bin/env bash

# This script converts a JSONL file to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

for f in $(find . -type f -name '*.jsonl'); do
    echo "Processing $f"
    ben -m encode $f -v -w
done
SH
}

function ben_to_xben_script() {
    cat << "SH"
#!/usr/bin/env bash

# This script converts a JSONL file to a BEN file using the BEN cli tool.
# Documentation at: https://crates.io/crates/binary-ensemble

for f in $(find . -type f -name '*.ben'); do
    echo "Processing $f"
    ben -m x-encode $f -v -w
done
SH
}

# =====================================
# ========  PYTHON CLI SCRIPT  ========
# =====================================

function basic_cli_gerrychain() {
    cat << "PY"
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
from pyben import PyBenEncoder
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
            with PyBenEncoder(output_path, overwrite=True) as encoder:
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
PY
}

# =============================================
# ========  PYTHON PIPELINE FUNCTIONS  ========
# =============================================

function python_process_partisan_bias() {
    cat << "PY"
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from pathlib import Path
import geopandas as gpd
import numpy as np
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(sample_idx, assignment_vector, vote_arrays):
    assign = np.asarray(assignment_vector, dtype=np.int64)

    k = int(assign.max())  # number of districts

    out = {"sample": sample_idx, "pb_scores": {}}

    for dem_votes, rep_votes, name in vote_arrays:
        dem_tot = np.bincount(assign, weights=dem_votes, minlength=k)
        rep_tot = np.bincount(assign, weights=rep_votes, minlength=k)

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

    decoder = PyBenDecoder(CHAIN_FILE)
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
PY
}

function python_process_polsby() {
    cat << "PY"
import jsonlines as jl
from gerrychain import GeographicPartition, Graph
from gerrychain.metrics import polsby_popper
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
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

    decoder = PyBenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

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
PY
}

function python_process_reock() {
    cat << "PY"
import jsonlines as jl
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import geopandas as gpd
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
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

    decoder = PyBenDecoder(CHAIN_FILE)
    total_chain_length = len(decoder)

    if n_samples > total_chain_length:
        print(
            "Requested more samples than available in chain; using full chain length."
        )
        n_samples = total_chain_length

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
PY
}

function python_process_splits() {
    cat << "PY"
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
from pathlib import Path
from pyben import PyBenDecoder
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

    decoder = PyBenDecoder(CHAIN_FILE)
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
PY
}

function python_process_total_dem_wins() {
    cat << "PY"
import jsonlines as jl
from gerrychain import Graph
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import numpy as np
import geopandas as gpd
from pathlib import Path
from pyben import PyBenDecoder
import os

script_dir = Path(__file__).parent
top_dir = script_dir.parents[1]


def compute_score(
    sample_idx,
    assignment_vector,
    dem_count_matrix,
    rep_count_matrix,
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

    return ({"sample": sample_idx, "scores": race_totals},)


if __name__ == "__main__":
    batch_size = 10_000

    CHAIN_FILE = f"{top_dir}/chain_outputs/MN_chain_100000_steps_seed42.jsonl.ben"
    GRAPH_PATH = f"{top_dir}/JSON_dualgraphs/MN_precincts.geojson"
    OUTPUT_PATH = f"{top_dir}/stats/MN_dem_win_scores.jsonl"

    decoder = PyBenDecoder(CHAIN_FILE)
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
                delayed(compute_score)(idx, vec, dem_count_matrix, rep_count_matrix)
                for idx, vec in pairs
            )

        all_scores.extend(scores)

    with jl.open(OUTPUT_PATH, "w") as writer:
        writer.write_all(all_scores)
PY
}

# ==============================================
# ========  MAIN INSTALLATION FUNCTION  ========
# ==============================================

function main() {
    check_for_realpath
    check_uv_installed
    read -p "Enter the name of the new project to create: " project_name
    if [[ -z "$project_name" ]]; then
        project_name="my_project"
        echo "No project name provided. Using default name: $project_name"
    fi

    read -p "Would you like to use FRCW in this project? (y/[n]): " use_frcw
    if [[ "$use_frcw" == "y" || "$use_frcw" == "Y" ]]; then
        check_cargo_installed
        echo "Installing FRCW from lattest git commit..."
        cargo install --git "https://github.com/mggg/frcw.rs" --branch "main"
        echo "FRCW has been installed."

        echo "Installing binary-ensemble"
        cargo install binary-ensemble
        echo "binary-ensemble has been installed."
    else
        read -p "Would you like to use BEN in this project? (y/[n]): " use_ben
        if [[ "$use_ben" == "y" || "$use_ben" == "Y" ]]; then
            check_cargo_installed
            echo "Installing binary-ensemble"
            cargo install binary-ensemble
            echo "binary-ensemble has been installed."
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

    echo "Project $project_name has been created and initialized with uv ($python_version)."
    echo "Adding standard packages to pyproject.toml..."

    # Get rid of some of the default files
    rm "README.md"
    rm "main.py"

    uv add numpy pandas matplotlib seaborn "gerrychain[geo]" maup ipykernel \
        ipywidgets click gerrytools binary-ensemble joblib joblib-progress docker

    # A formatter that I like
    uv add tool black

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

    # grab the JSON file
    curl -L -o "JSON_dualgraphs/gerrymandria.json" "https://raw.githubusercontent.com/mggg/GerryChain/refs/heads/main/docs/_static/gerrymandria.json"
    curl -L https://github.com/mggg/GerryChain/raw/main/docs/_static/MN.zip | bsdtar -xvf - -C "JSON_dualgraphs"

    # make the script files and mark bash files as executable
    basic_cli_gerrychain > pipeline_scripts/example_cli.py
    rust_sh_example > pipeline_scripts/rust_example_script.sh
    chmod +x pipeline_scripts/rust_example_script.sh
    batch_example_simple > batch_example_python_cli_simple.sh
    chmod +x batch_example_python_cli_simple.sh
    batch_example_parallel > batch_example_python_cli_parallel.sh
    chmod +x batch_example_python_cli_parallel.sh
    jsonl_to_ben_script > chain_outputs/jsonl_to_ben.sh
    chmod +x chain_outputs/jsonl_to_ben.sh
    ben_to_xben_script > chain_outputs/ben_to_xben.sh
    chmod +x chain_outputs/ben_to_xben.sh

    # make the python processing scripts
    metric_dir="pipeline_scripts/metrics"
    mkdir -p "$metric_dir"
    python_process_partisan_bias > "$metric_dir/process_partisan_bias.py"
    python_process_polsby > "$metric_dir/process_polsby.py"
    python_process_reock > "$metric_dir/process_reock.py"
    python_process_splits > "$metric_dir/process_splits.py"
    python_process_total_dem_wins > "$metric_dir/process_total_dem_wins.py"

    echo "Your project is ready! You may need to restart your shell for uv to work properly."
}

main "$@"
