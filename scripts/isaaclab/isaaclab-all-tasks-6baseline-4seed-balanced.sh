#!/usr/bin/env bash
set -uo pipefail

# Run the six online on-policy baselines with four seeds on every IsaacLab task
# listed in the baseline report:
#   12 tasks * 6 algorithms * 4 seeds = 288 expected runs.
#
# All task/algo/seed combinations are placed in one global queue. GenPO uses at
# most four concurrent slots by default, leaving the other GPUs available for
# the remaining algorithms. There is no task-level scheduling barrier.
#
# Usage:
#   bash scripts/isaaclab/isaaclab-all-tasks-6baseline-4seed-balanced.sh
#
# Inspect every command without launching jobs or writing metadata:
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-all-tasks-6baseline-4seed-balanced.sh
#
# Run only selected tasks:
#   TASKS="Isaac-Ant-v0 Isaac-Humanoid-v0" bash \
#     scripts/isaaclab/isaaclab-all-tasks-6baseline-4seed-balanced.sh
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GLOBAL_QUEUE_LAUNCHER="${SCRIPT_DIR}/isaaclab-lift-cube-6baseline-4seed-balanced.sh"
cd "${REPO_ROOT}"

PYTHON_BIN_REQUESTED="${PYTHON_BIN:-}"
PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3}"
read -r -a ALGOS <<<"${ALGOS:-ppo dppo fpo fpopp genpo policyflow}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2 3}"
read -r -a TASKS <<<"${TASKS:-Isaac-Repose-Cube-Shadow-Direct-v0 Isaac-Velocity-Rough-H1-v0 Isaac-Humanoid-v0 Isaac-Velocity-Rough-Unitree-Go2-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Ant-v0 Isaac-Quadcopter-Direct-v0 Isaac-Cartpole-v0 Isaac-Lift-Cube-Franka-v0}"

TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
MASTER_SWEEP_ID="${MASTER_SWEEP_ID:-isaaclab-all-tasks-smoothness-200m-4seed-v1}"
MASTER_LOG_ROOT="${MASTER_LOG_ROOT:-run_logs/${MASTER_SWEEP_ID}}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-smoothness-alltasks-4seed-v1}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-${MASTER_SWEEP_ID}}"
LOG_GROUP="${LOG_GROUP:-${MASTER_SWEEP_ID}}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT:-isaaclab-onpolicy-smoothness}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
GENPO_EXTRA_ARGS="${GENPO_EXTRA_ARGS:-algo.batch_size=4096 algo.num_minibatches=6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
RESUME_LOCAL="${RESUME_LOCAL:-1}"
SAVE_CKPT="${SAVE_CKPT:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
ALGO_MAX_CONCURRENT="${ALGO_MAX_CONCURRENT:-genpo=4}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-40000}"
DRY_RUN="${DRY_RUN:-0}"
REUSE_EXISTING_LIFT="${REUSE_EXISTING_LIFT:-0}"
READY_FILE="${READY_FILE:-}"

CURRENT_PID=""

resolve_python() {
    local candidate
    if [[ -n "${PYTHON_BIN_REQUESTED}" ]]; then
        candidate="${PYTHON_BIN_REQUESTED}"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        candidate="${CONDA_PREFIX}/bin/python"
    else
        candidate="python3"
    fi

    if [[ "${candidate}" != */* ]]; then
        candidate="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
        echo "Python executable not found: ${PYTHON_BIN_REQUESTED:-python3}" >&2
        return 1
    fi

    PYTHON_BIN="$("${candidate}" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
        echo "Could not resolve Python executable from: ${candidate}" >&2
        return 1
    fi
}

preflight_python() {
    local summary
    if ! summary="$(PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" -c '
import sys
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"Python >=3.11 is required, got {sys.version.split()[0]} at {sys.executable}. "
        "Create and activate a Python 3.11 environment, then rerun with "
        "PYTHON_BIN=$CONDA_PREFIX/bin/python."
    )
import hydra
import jax
import jaxlib
import examples.online.main_isaaclab_onpolicy
print(f"{sys.executable} | jax={jax.__version__} jaxlib={jaxlib.__version__}")
' 2>&1)"; then
        echo "Python preflight failed for: ${PYTHON_BIN}" >&2
        printf '%s\n' "${summary}" >&2
        echo "Activate the intended environment in this same shell, or launch with PYTHON_BIN=/absolute/path/to/python." >&2
        return 1
    fi
    echo "Python preflight: ${summary}"
}

preflight_graphics() {
    local summary
    if ! command -v vulkaninfo >/dev/null 2>&1; then
        echo "Warning: vulkaninfo is unavailable; skipping the Isaac Sim graphics preflight." >&2
        return 0
    fi
    summary="$(vulkaninfo --summary 2>&1 || true)"
    if ! grep -Eiq 'vendorID[[:space:]]*= 0x10de|deviceName[[:space:]]*= NVIDIA' <<<"${summary}"; then
        echo "Isaac Sim graphics preflight failed: Vulkan cannot see an NVIDIA GPU." >&2
        echo "CUDA/nvidia-smi may still work because CUDA and Vulkan load different driver libraries." >&2
        printf '%s\n' "${summary}" >&2
        return 1
    fi
    echo "Isaac Sim graphics preflight: NVIDIA Vulkan GPU available"
}

is_bool() {
    [[ "$1" == "0" || "$1" == "1" ]]
}

validate_inputs() {
    local task algo seed gpu flag value
    local -A seen_tasks=()
    local -A seen_algos=()
    local -A seen_seeds=()
    local -A seen_gpus=()

    if [[ ! -x "${GLOBAL_QUEUE_LAUNCHER}" ]]; then
        echo "Global queue launcher is missing or not executable: ${GLOBAL_QUEUE_LAUNCHER}" >&2
        return 1
    fi
    if (( ${#GPUS[@]} == 0 )); then
        echo "At least one GPU is required." >&2
        return 1
    fi
    if (( ${#TASKS[@]} == 0 || ${#ALGOS[@]} == 0 || ${#SEEDS[@]} == 0 )); then
        echo "TASKS, ALGOS, and SEEDS must all be non-empty." >&2
        return 1
    fi

    for gpu in "${GPUS[@]}"; do
        if ! [[ "${gpu}" =~ ^[0-9]+$ ]] || [[ -v "seen_gpus[${gpu}]" ]]; then
            echo "GPU ids must be unique non-negative integers; got: ${GPUS[*]}" >&2
            return 1
        fi
        seen_gpus[${gpu}]=1
    done
    for seed in "${SEEDS[@]}"; do
        if ! [[ "${seed}" =~ ^[0-9]+$ ]] || [[ -v "seen_seeds[${seed}]" ]]; then
            echo "Seeds must be unique non-negative integers; got: ${SEEDS[*]}" >&2
            return 1
        fi
        seen_seeds[${seed}]=1
    done
    for algo in "${ALGOS[@]}"; do
        if [[ -v "seen_algos[${algo}]" ]]; then
            echo "Duplicate algorithm: ${algo}" >&2
            return 1
        fi
        seen_algos[${algo}]=1
        if [[ ! -f "examples/online/config/isaaclab_onpolicy/algo/${algo}.yaml" ]]; then
            echo "Missing algorithm config: examples/online/config/isaaclab_onpolicy/algo/${algo}.yaml" >&2
            return 1
        fi
    done
    for task in "${TASKS[@]}"; do
        if [[ -v "seen_tasks[${task}]" ]]; then
            echo "Duplicate task: ${task}" >&2
            return 1
        fi
        seen_tasks[${task}]=1
        if [[ ! -f "examples/online/config/isaaclab_onpolicy/task/${task}.yaml" ]]; then
            echo "Missing task config: examples/online/config/isaaclab_onpolicy/task/${task}.yaml" >&2
            return 1
        fi
    done
    for flag in WANDB_ENABLED RESUME_LOCAL SAVE_CKPT AUTO_RESUME CHECK_GPU_MEM DRY_RUN REUSE_EXISTING_LIFT; do
        value="${!flag}"
        if ! is_bool "${value}"; then
            echo "${flag} must be 0 or 1, got: ${value}" >&2
            return 1
        fi
    done
    if ! [[ "${TRAIN_FRAMES}" =~ ^[0-9]+$ && "${EVAL_FRAMES}" =~ ^[0-9]+$ ]]; then
        echo "TRAIN_FRAMES and EVAL_FRAMES must be non-negative integers." >&2
        return 1
    fi
    if ! [[ "${INTERRUPT_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "INTERRUPT_GRACE_SECONDS must be a non-negative integer." >&2
        return 1
    fi
    if [[ "${REUSE_EXISTING_LIFT}" != "0" ]]; then
        echo "REUSE_EXISTING_LIFT=1 is incompatible with the global queue; copy those .done markers into MASTER_LOG_ROOT first." >&2
        return 1
    fi
}

stop_current() {
    local status="$1"
    trap - INT TERM
    if [[ -n "${CURRENT_PID}" ]]; then
        echo "[$(date '+%F %T')] stopping current task suite pid=${CURRENT_PID}" >&2
        kill -TERM "${CURRENT_PID}" 2>/dev/null || true
        wait "${CURRENT_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}

trap 'stop_current 130' INT
trap 'stop_current 143' TERM

print_plan() {
    local total_expected
    total_expected=$(( ${#TASKS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))

    echo "Repo: ${REPO_ROOT}"
    echo "Tasks (${#TASKS[@]}): ${TASKS[*]}"
    echo "Algorithms (${#ALGOS[@]}): ${ALGOS[*]}"
    echo "Seeds (${#SEEDS[@]}): ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Expected matrix represented by this launch: ${total_expected}"
    echo "Schedule: one global cross-task queue; no task barrier"
    echo "Algorithm concurrency limits: ${ALGO_MAX_CONCURRENT}"
    echo "Train/eval frames: ${TRAIN_FRAMES}/${EVAL_FRAMES}"
    echo "Checkpoint/resume: save=${SAVE_CKPT} auto_resume=${AUTO_RESUME}"
    echo "Interrupt grace: ${INTERRUPT_GRACE_SECONDS}s"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} mode=${WANDB_MODE}"
    echo "W&B group for new runs: ${LOG_GROUP}"
    echo "Master logs: ${MASTER_LOG_ROOT}"
}

run_global_suite() {
    TASKS="${TASKS[*]}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    PYTHON_PREFLIGHT_DONE=1 \
    GRAPHICS_PREFLIGHT_DONE=1 \
    GPUS="${GPUS[*]}" \
    ALGOS="${ALGOS[*]}" \
    SEEDS="${SEEDS[*]}" \
    TRAIN_FRAMES="${TRAIN_FRAMES}" \
    EVAL_FRAMES="${EVAL_FRAMES}" \
    SWEEP_ID="${MASTER_SWEEP_ID}" \
    RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX}" \
    LOG_DIR="${LOG_DIR}" \
    LOG_TAG="${LOG_TAG}" \
    LOG_GROUP="${LOG_GROUP}" \
    LOG_ROOT="${MASTER_LOG_ROOT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_ENABLED="${WANDB_ENABLED}" \
    GENPO_EXTRA_ARGS="${GENPO_EXTRA_ARGS}" \
    EXTRA_ARGS="${EXTRA_ARGS}" \
    JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS}" \
    RESUME_LOCAL="${RESUME_LOCAL}" \
    SAVE_CKPT="${SAVE_CKPT}" \
    AUTO_RESUME="${AUTO_RESUME}" \
    INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS}" \
    ALGO_MAX_CONCURRENT="${ALGO_MAX_CONCURRENT}" \
    CHECK_GPU_MEM="${CHECK_GPU_MEM}" \
    MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB}" \
    READY_FILE="${READY_FILE}" \
    DRY_RUN="${DRY_RUN}" \
        bash "${GLOBAL_QUEUE_LAUNCHER}"
}

main() {
    resolve_python || return 1
    validate_inputs || return 1
    print_plan

    if [[ "${DRY_RUN}" != "1" ]]; then
        preflight_python || return 1
        preflight_graphics || return 1
    fi
    run_global_suite &
    CURRENT_PID="$!"
    wait "${CURRENT_PID}"
    status=$?
    CURRENT_PID=""
    return "${status}"
}

main "$@"
