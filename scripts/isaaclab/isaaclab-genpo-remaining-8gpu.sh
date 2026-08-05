#!/usr/bin/env bash
set -euo pipefail

# Resume only the unfinished GenPO runs from the 12-task action-smoothness
# sweep. The original training artifact path and run names are retained so
# interrupted jobs restore their local checkpoints and W&B run ids.
#
# The scheduler metadata uses a new directory because the original sweep was
# configured with all six algorithms and a GenPO concurrency limit of four.
# Completed GenPO markers are mirrored into the new directory before launch,
# so the global queue schedules only failed, interrupted, or never-started
# runs.
#
# Usage (foreground; keep the shell alive):
#   bash scripts/isaaclab/isaaclab-genpo-remaining-8gpu.sh
#
# Inspect the 29 expected remaining commands without launching GPU jobs:
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-genpo-remaining-8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUEUE_WRAPPER="${SCRIPT_DIR}/isaaclab-all-tasks-6baseline-4seed-balanced.sh"

SOURCE_SWEEP_ID="isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1"
SOURCE_LOG_ROOT="${REPO_ROOT}/run_logs/${SOURCE_SWEEP_ID}"
RESUME_SWEEP_ID="isaaclab-genpo-remaining-200m-4seed-8gpu-resume-v1"
DEFAULT_RESUME_LOG_ROOT="${REPO_ROOT}/run_logs/${RESUME_SWEEP_ID}"
ORIGINAL_RUN_NAME_SUFFIX="smoothness-alltasks-4seed-8gpu-clean-v1"

TASKS=(
    Isaac-Repose-Cube-Shadow-Direct-v0
    Isaac-Velocity-Rough-H1-v0
    Isaac-Humanoid-v0
    Isaac-Velocity-Rough-Unitree-Go2-v0
    Isaac-Velocity-Rough-G1-v0
    Isaac-Velocity-Flat-G1-v0
    Isaac-Velocity-Flat-Anymal-D-v0
    Isaac-Open-Drawer-Franka-v0
    Isaac-Ant-v0
    Isaac-Quadcopter-Direct-v0
    Isaac-Cartpole-v0
    Isaac-Lift-Cube-Franka-v0
)
SEEDS=(0 1 2 3)

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
    exit 2
fi
if [[ ! -x "${QUEUE_WRAPPER}" ]]; then
    echo "Queue wrapper is missing or not executable: ${QUEUE_WRAPPER}" >&2
    exit 1
fi
if [[ ! -d "${SOURCE_LOG_ROOT}" ]]; then
    echo "Original sweep log root is missing: ${SOURCE_LOG_ROOT}" >&2
    exit 1
fi

DRY_RUN_ROOT=""
cleanup() {
    if [[ -n "${DRY_RUN_ROOT}" && -d "${DRY_RUN_ROOT}" ]]; then
        rm -rf -- "${DRY_RUN_ROOT}"
    fi
}
trap cleanup EXIT

if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_RUN_ROOT="$(mktemp -d /tmp/flowrl-genpo-resume-dry-run.XXXXXX)"
    RESUME_LOG_ROOT="${DRY_RUN_ROOT}/run_logs/${RESUME_SWEEP_ID}"
else
    RESUME_LOG_ROOT="${RESUME_LOG_ROOT:-${DEFAULT_RESUME_LOG_ROOT}}"
fi

mirror_completed_markers() {
    local task seed run_name source_marker target_marker mirrored=0 existing=0

    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_name="${task}-genpo-seed${seed}-${ORIGINAL_RUN_NAME_SUFFIX}"
            source_marker="${SOURCE_LOG_ROOT}/${task}/${run_name}.done"
            target_marker="${RESUME_LOG_ROOT}/${task}/${run_name}.done"
            [[ -f "${source_marker}" ]] || continue
            if [[ -f "${target_marker}" ]]; then
                existing=$((existing + 1))
                continue
            fi
            mkdir -p "$(dirname "${target_marker}")"
            cp -p -- "${source_marker}" "${target_marker}"
            mirrored=$((mirrored + 1))
        done
    done

    local completed_total
    completed_total="$(find "${RESUME_LOG_ROOT}" -type f -name '*-genpo-seed*.done' | wc -l)"
    echo "GenPO completion markers: source/new=${mirrored}, already/new=${existing}, total/new=${completed_total}/48"
    echo "GenPO runs to schedule: $((48 - completed_total))"
}

mkdir -p "${RESUME_LOG_ROOT}"
mirror_completed_markers

cd "${REPO_ROOT}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.11}"
export GPUS="0 1 2 3 4 5 6 7"
export ALGOS="genpo"
export SEEDS="${SEEDS[*]}"
export TASKS="${TASKS[*]}"

export TRAIN_FRAMES=200000000
export EVAL_FRAMES=5000000
export RESUME_LOCAL=1
export SAVE_CKPT=1
export AUTO_RESUME=1
export INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
export ALGO_MAX_CONCURRENT="genpo=8"
export GENPO_EXTRA_ARGS="algo.batch_size=4096 algo.num_minibatches=6"
export EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
export CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
export MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-40000}"
export REUSE_EXISTING_LIFT=0
export DRY_RUN

export WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-onpolicy-smoothness}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"

export MASTER_SWEEP_ID="${RESUME_SWEEP_ID}"
export MASTER_LOG_ROOT="${RESUME_LOG_ROOT}"
export RUN_NAME_SUFFIX="${ORIGINAL_RUN_NAME_SUFFIX}"
export LOG_TAG="${SOURCE_SWEEP_ID}"
export LOG_GROUP="${SOURCE_SWEEP_ID}"
export READY_FILE="${RESUME_LOG_ROOT}/launcher.ready"

if [[ "${DRY_RUN}" == "1" ]]; then
    bash "${QUEUE_WRAPPER}"
    exit $?
fi

LAUNCHER_LOG="${RESUME_LOG_ROOT}/launcher.log"
{
    echo
    echo "[$(date --iso-8601=seconds)] GenPO 8-GPU resume requested"
    echo "Original status root: ${SOURCE_LOG_ROOT}"
    echo "Resume status root: ${RESUME_LOG_ROOT}"
    echo "Training artifacts: logs/genpo/${SOURCE_SWEEP_ID}"
} >>"${LAUNCHER_LOG}"

echo "Starting the remaining GenPO runs on GPUs 0-7 in the foreground."
echo "Keep this shell alive; Ctrl-C requests checkpointed shutdown."
echo "Launcher log: ${LAUNCHER_LOG}"

exec > >(tee -a "${LAUNCHER_LOG}") 2>&1
exec bash "${QUEUE_WRAPPER}"
