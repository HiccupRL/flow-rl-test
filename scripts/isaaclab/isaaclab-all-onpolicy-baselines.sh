#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALGOS=(${ALGOS:-dppo fpo fpopp genpo})
TASKS=(
    "Isaac-Humanoid-v0"
    "Isaac-Cartpole-v0"
    "Isaac-Ant-v0"
    "Isaac-Open-Drawer-Franka-v0"
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Lift-Cube-Franka-v0"
    "Isaac-Velocity-Flat-G1-v0"
    "Isaac-Velocity-Rough-G1-v0"
)
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
WANDB_PROJECT="${WANDB_PROJECT:-isaaclab-ppo-baseline}"
CURRENT_PID=""

stop_current() {
    local status="$1"

    trap - INT TERM
    if [[ -n "$CURRENT_PID" ]]; then
        echo "[$(date '+%F %T')] stopping current baseline suite ..."
        kill -TERM "$CURRENT_PID" 2>/dev/null || true
        wait "$CURRENT_PID" 2>/dev/null || true
    fi
    exit "$status"
}

trap 'stop_current 130' INT
trap 'stop_current 143' TERM

status=0
total_algos="${#ALGOS[@]}"
runs_per_algo=$((${#TASKS[@]} * ${#SEEDS[@]}))
total_runs=$((total_algos * runs_per_algo))
for index in "${!ALGOS[@]}"; do
    algo="${ALGOS[$index]}"
    current=$((index + 1))
    remaining_algos=$((total_algos - current))
    start_run=$((index * runs_per_algo + 1))
    end_run=$(((index + 1) * runs_per_algo))
    exp_name="isaaclab-${algo}-baseline"

    echo "[$(date '+%F %T')] progress | algo ${current}/${total_algos} | runs ${start_run}-${end_run}/${total_runs} | running=${algo} | remaining_algos=${remaining_algos}"
    echo "[$(date '+%F %T')] wandb | project=${WANDB_PROJECT} | group=${exp_name}"

    ALGO="$algo" EXP_NAME="$exp_name" WANDB_PROJECT="$WANDB_PROJECT" \
        SEEDS="${SEEDS[*]}" PROGRESS_OFFSET="$((index * runs_per_algo))" PROGRESS_TOTAL="$total_runs" \
        bash "${SCRIPT_DIR}/isaaclab-onpolicy-baseline.sh" &
    CURRENT_PID="$!"

    if ! wait "$CURRENT_PID"; then
        status=1
        echo "[$(date '+%F %T')] algo failed | ${current}/${total_algos} | runs ${start_run}-${end_run}/${total_runs} | algo=${algo} | remaining_algos=${remaining_algos}" >&2
    else
        echo "[$(date '+%F %T')] algo done | ${current}/${total_algos} | runs ${start_run}-${end_run}/${total_runs} | algo=${algo} | remaining_algos=${remaining_algos}"
    fi
    CURRENT_PID=""
done

exit "$status"
