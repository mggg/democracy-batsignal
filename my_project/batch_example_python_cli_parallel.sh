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
        spinner_start "Waiting for a free slot: $(jobs -pr | wc -l)/$MAX_JOBS running…"
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
    spinner_start "Finishing remaining jobs…"
    wait "${pids[@]}" 2> /dev/null || true
    spinner_stop
fi
