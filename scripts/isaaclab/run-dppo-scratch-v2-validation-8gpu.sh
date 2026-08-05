#!/usr/bin/env bash
set -euo pipefail

# Focused validation for the from-scratch DPPO adaptation:
#   2 tasks * 4 seeds = 8 runs, one run per GPU.
#
# Lift Cube keeps disable_bootstrap=true from its task YAML.  This script does
# not override task-specific bootstrap handling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

read -r -a GPUS <<<"${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a TASKS <<<"${TASKS:-Isaac-Lift-Cube-Franka-v0 Isaac-Open-Drawer-Franka-v0}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2 3}"

TRAIN_FRAMES="${TRAIN_FRAMES:-50000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
LOG_TAG="${LOG_TAG:-isaaclab-dppo-scratch-v2-validation-50m}"
LOG_GROUP="${LOG_GROUP:-${LOG_TAG}}"
LOG_ROOT="${LOG_ROOT:-run_logs/${LOG_TAG}}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-dppo-scratch-v2}"
WANDB_MODE="${WANDB_MODE:-online}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

if (( ${#GPUS[@]} != ${#TASKS[@]} * ${#SEEDS[@]} )); then
    echo "Need one GPU per task/seed run: gpus=${#GPUS[@]} runs=$((${#TASKS[@]} * ${#SEEDS[@]}))." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}"
declare -a PIDS=()

stop_all() {
    local pid
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    exit 130
}
trap stop_all INT TERM

index=0
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu="${GPUS[${index}]}"
        run_name="${task}-dppo-scratch-v2-seed${seed}-50m"
        log_file="${LOG_ROOT}/${run_name}.gpu${gpu}.log"
        command=(
            "${PYTHON_BIN}" examples/online/main_isaaclab_onpolicy.py
            "task=${task}"
            "algo=dppo_scratch"
            "seed=${seed}"
            "device=0"
            "train_frames=${TRAIN_FRAMES}"
            "eval_frames=${EVAL_FRAMES}"
            "log.tag=${LOG_TAG}"
            "log.group=${LOG_GROUP}"
            "log.name=${run_name}"
            "log.tags=[${task},dppo_scratch,isaaclab,identity-score-init,critic-warmup,4seed]"
            "log.save_ckpt=true"
            "log.resume=true"
            "log.wandb=true"
            "log.wandb_mode=${WANDB_MODE}"
            "log.project=${WANDB_PROJECT_NAME}"
            "log.entity=${WANDB_ENTITY}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
            printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "${gpu}" "${REPO_ROOT}"
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            echo "[$(date '+%F %T')] start gpu=${gpu} task=${task} seed=${seed}"
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                export EGL_VISIBLE_DEVICES="${gpu}"
                export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
                export WANDB_MODE="${WANDB_MODE}"
                export OMNI_KIT_ACCEPT_EULA=YES
                export FLOWRL_ISAACLAB_CLOSE_APP=0
                export XLA_PYTHON_CLIENT_PREALLOCATE=false
                export PYTHONUNBUFFERED=1
                "${command[@]}"
            ) >"${log_file}" 2>&1 &
            PIDS+=("$!")
        fi
        index=$((index + 1))
    done
done

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        failed=$((failed + 1))
    fi
done

echo "Validation finished: runs=${#PIDS[@]} failed=${failed} logs=${LOG_ROOT}"
(( failed == 0 ))
