#!/usr/bin/env bash

set -uo pipefail

EXP_NAME="${EXP_NAME:-isaaclab-ppo-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-log/parallel/${EXP_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-${EXP_NAME}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Space-separated overrides are supported, e.g.:
#   GPUS="0 1 2 3" SEEDS="0 1 2" bash scripts/isaaclab/isaaclab-ppo-baseline.sh
GPUS=(${GPUS:-0 1 2 3})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})

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

run_one() {
    local task="$1"
    local seed="$2"
    local gpu="$3"
    local task_log_dir="${RUN_LOG_ROOT}/${task}"
    local run_log="${task_log_dir}/seed${seed}.log"

    mkdir -p "$task_log_dir"

    local cmd=(
        "$PYTHON_BIN" examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "seed=${seed}"
        "device=0"
        "algo=ppo"
        "log.tag=${EXP_NAME}"
        "log.dir=${LOG_DIR}"
        "log.project=${WANDB_PROJECT}"
        "log.entity=${WANDB_ENTITY}"
    )

    echo "[$(date '+%F %T')] GPU ${gpu} | ${task} | seed ${seed}"
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
    ) 2>&1 | tee "$run_log"

    local status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        echo "Failed: ${task} seed ${seed} on GPU ${gpu}; see ${run_log}" >&2
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
                run_one "$task" "$seed" "$gpu" || status=1
            fi
            ((job_idx++))
        done
    done

    return "$status"
}

main() {
    local num_gpus="${#GPUS[@]}"
    local status=0
    local pids=()

    if (( num_gpus == 0 )); then
        echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
        exit 1
    fi

    mkdir -p "$RUN_LOG_ROOT"
    echo "Experiment: ${EXP_NAME}"
    echo "WandB project: ${WANDB_PROJECT}"
    echo "WandB entity: ${WANDB_ENTITY:-<default>}"
    echo "GPUs: ${GPUS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Run logs: ${RUN_LOG_ROOT}"

    for worker_idx in "${!GPUS[@]}"; do
        worker "$worker_idx" "${GPUS[$worker_idx]}" "$num_gpus" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    exit "$status"
}

main "$@"
