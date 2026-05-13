#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALGOS=(${ALGOS:-ppo dppo fpo fpopp genpo})
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
total="${#ALGOS[@]}"
for index in "${!ALGOS[@]}"; do
    algo="${ALGOS[$index]}"
    current=$((index + 1))
    remaining=$((total - current))
    exp_name="isaaclab-${algo}-baseline"

    echo "[$(date '+%F %T')] progress | script ${current}/${total} | running=${algo} | remaining=${remaining}"
    echo "[$(date '+%F %T')] wandb | project=${WANDB_PROJECT} | group=${exp_name}"

    ALGO="$algo" EXP_NAME="$exp_name" WANDB_PROJECT="$WANDB_PROJECT" \
        bash "${SCRIPT_DIR}/isaaclab-onpolicy-baseline.sh" &
    CURRENT_PID="$!"

    if ! wait "$CURRENT_PID"; then
        status=1
        echo "[$(date '+%F %T')] script failed | ${current}/${total} | algo=${algo} | remaining=${remaining}" >&2
    else
        echo "[$(date '+%F %T')] script done | ${current}/${total} | algo=${algo} | remaining=${remaining}"
    fi
    CURRENT_PID=""
done

exit "$status"
