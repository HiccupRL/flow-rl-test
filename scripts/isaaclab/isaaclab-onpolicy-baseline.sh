#!/usr/bin/env bash

set -uo pipefail

ALGO="${ALGO:-}"
if [[ -z "$ALGO" ]]; then
    echo "ALGO is required, e.g. ALGO=ppo bash scripts/isaaclab/isaaclab-onpolicy-baseline.sh" >&2
    exit 2
fi

EXP_NAME="${EXP_NAME:-isaaclab-${ALGO}-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-log/parallel/${EXP_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-isaaclab-ppo-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRAIN_FRAMES="${TRAIN_FRAMES:-200_000_000}"
PROGRESS_OFFSET="${PROGRESS_OFFSET:-0}"
PROGRESS_TOTAL="${PROGRESS_TOTAL:-0}"

# Space-separated overrides are supported, e.g.:
#   ALGO=ppo GPUS="0 1 2 3" SEEDS="0 1 2" bash scripts/isaaclab/isaaclab-onpolicy-baseline.sh
GPUS=(${GPUS:-0 1 2 3})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
PIDS=()
STOPPING=0

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

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

kill_tree() {
    local parent="$1"
    local signal="$2"
    local child

    if command -v pgrep >/dev/null 2>&1; then
        while IFS= read -r child; do
            [[ -n "$child" ]] && kill_tree "$child" "$signal"
        done < <(pgrep -P "$parent" 2>/dev/null || true)
    fi

    kill "-${signal}" "$parent" 2>/dev/null || true
}

stop_all() {
    local status="$1"

    if (( STOPPING != 0 )); then
        exit "$status"
    fi
    STOPPING=1
    trap - INT TERM

    echo "[$(date '+%F %T')] stopping all ${ALGO} workers ..."
    for pid in "${PIDS[@]:-}"; do
        kill_tree "$pid" TERM
    done
    sleep 3
    for pid in "${PIDS[@]:-}"; do
        kill_tree "$pid" KILL
    done
    wait 2>/dev/null || true
    echo "[$(date '+%F %T')] stopped"
    exit "$status"
}

trap 'stop_all 130' INT
trap 'stop_all 143' TERM

run_one() {
    local task="$1"
    local seed="$2"
    local gpu="$3"
    local progress_idx="$4"
    local task_log_dir="${RUN_LOG_ROOT}/${task}"
    local run_log="${task_log_dir}/seed${seed}.log"

    mkdir -p "$task_log_dir"

    local cmd=(
        "$PYTHON_BIN" examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "seed=${seed}"
        "device=0"
        "algo=${ALGO}"
        "train_frames=${TRAIN_FRAMES}"
        "log.tag=${EXP_NAME}"
        "log.dir=${LOG_DIR}"
        "log.project=${WANDB_PROJECT}"
        "log.entity=${WANDB_ENTITY}"
    )

    local start_ts
    start_ts=$(date +%s)
    if (( PROGRESS_TOTAL > 0 )); then
        echo "[$(date '+%F %T')] run ${progress_idx}/${PROGRESS_TOTAL} start | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} frames=${TRAIN_FRAMES} log=${run_log}"
    else
        echo "[$(date '+%F %T')] start | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} frames=${TRAIN_FRAMES} log=${run_log}"
    fi
    if [[ "${DRY_RUN:-0}" != "0" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%s XLA_PYTHON_CLIENT_PREALLOCATE=%s ' \
            "$gpu" "$XLA_PYTHON_CLIENT_PREALLOCATE"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "${cmd[@]}"
    ) >"$run_log" 2>&1

    local status=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    if (( status != 0 )); then
        if (( PROGRESS_TOTAL > 0 )); then
            echo "[$(date '+%F %T')] run ${progress_idx}/${PROGRESS_TOTAL} failed | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} exit=${status} elapsed=${elapsed}s log=${run_log}" >&2
        else
            echo "[$(date '+%F %T')] failed | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} exit=${status} elapsed=${elapsed}s log=${run_log}" >&2
        fi
        echo "Last 40 log lines:" >&2
        tail -n 40 "$run_log" >&2
    else
        if (( PROGRESS_TOTAL > 0 )); then
            echo "[$(date '+%F %T')] run ${progress_idx}/${PROGRESS_TOTAL} done | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} elapsed=${elapsed}s log=${run_log}"
        else
            echo "[$(date '+%F %T')] done | algo=${ALGO} gpu=${gpu} task=${task} seed=${seed} elapsed=${elapsed}s log=${run_log}"
        fi
    fi
    return "$status"
}

worker() {
    local worker_idx="$1"
    local gpu="$2"
    local num_workers="$3"
    local job_idx=0
    local status=0

    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if (( job_idx % num_workers == worker_idx )); then
                run_one "$task" "$seed" "$gpu" "$((PROGRESS_OFFSET + job_idx + 1))" || status=1
            fi
            ((job_idx++))
        done
    done

    return "$status"
}

main() {
    local num_gpus="${#GPUS[@]}"
    local status=0

    if (( num_gpus == 0 )); then
        echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
        exit 1
    fi

    mkdir -p "$RUN_LOG_ROOT"
    echo "Experiment: ${EXP_NAME}"
    echo "Algorithm: ${ALGO}"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "WandB project: ${WANDB_PROJECT}"
    echo "WandB entity: ${WANDB_ENTITY:-<default>}"
    echo "GPUs: ${GPUS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Run logs: ${RUN_LOG_ROOT}"

    for worker_idx in "${!GPUS[@]}"; do
        worker "$worker_idx" "${GPUS[$worker_idx]}" "$num_gpus" &
        PIDS+=("$!")
    done

    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    exit "$status"
}

main "$@"
