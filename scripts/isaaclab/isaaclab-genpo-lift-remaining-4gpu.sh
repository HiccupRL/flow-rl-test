#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the four unfinished Isaac-Lift-Cube-Franka-v0 GenPO seeds on four
# GPUs. This keeps the original run names, W&B runs, and checkpoint tree while
# using an independent scheduler-status root, so it can run concurrently with
# the Flat-G1 single-GPU resume script on another machine.
#
# Usage:
#   bash scripts/isaaclab/isaaclab-genpo-lift-remaining-4gpu.sh
#
# Optional GPU remapping (exactly four ids are required):
#   GPUS="2 3 4 5" bash scripts/isaaclab/isaaclab-genpo-lift-remaining-4gpu.sh
#
# Inspect the four commands without launching:
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-genpo-lift-remaining-4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUEUE_WRAPPER="${SCRIPT_DIR}/isaaclab-all-tasks-6baseline-4seed-balanced.sh"

SOURCE_SWEEP_ID="isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1"
RESUME_SWEEP_ID="isaaclab-genpo-lift-remaining-4gpu-resume-v1"
RUN_NAME_SUFFIX="smoothness-alltasks-4seed-8gpu-clean-v1"
TASK="Isaac-Lift-Cube-Franka-v0"
GPU_LIST="${GPUS:-0 1 2 3}"
read -r -a GPU_IDS <<<"${GPU_LIST}"

if (( ${#GPU_IDS[@]} != 4 )); then
    echo "This launcher requires exactly four GPU ids; got: ${GPU_LIST}" >&2
    exit 2
fi
if [[ ! -x "${QUEUE_WRAPPER}" ]]; then
    echo "Queue wrapper is missing or not executable: ${QUEUE_WRAPPER}" >&2
    exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
    exit 2
fi

LOG_DIR="${LOG_DIR:-logs}"
if [[ "${LOG_DIR}" == /* ]]; then
    ARTIFACT_ROOT="${LOG_DIR}"
else
    ARTIFACT_ROOT="${REPO_ROOT}/${LOG_DIR}"
fi

require_checkpoint() {
    local seed="$1"
    local run_name="${TASK}-genpo-seed${seed}-${RUN_NAME_SUFFIX}"
    local checkpoint_root="${ARTIFACT_ROOT}/genpo/${SOURCE_SWEEP_ID}/${TASK}/${run_name}/ckpt"
    local latest

    if [[ ! -d "${checkpoint_root}" ]]; then
        echo "Missing checkpoint directory for seed ${seed}: ${checkpoint_root}" >&2
        return 1
    fi
    latest="$(
        find "${checkpoint_root}" -mindepth 2 -maxdepth 2 -type f -name _SUCCESS -printf '%h\n' |
            sed 's#.*/##' |
            sort -n |
            tail -1
    )"
    if [[ -z "${latest}" ]]; then
        echo "No complete checkpoint found for seed ${seed}: ${checkpoint_root}" >&2
        return 1
    fi
    echo "Lift-Cube seed ${seed}: resume checkpoint ${latest}/200000000"
}

for seed in 0 1 2 3; do
    require_checkpoint "${seed}"
done

RESUME_LOG_ROOT="${RESUME_LOG_ROOT:-${REPO_ROOT}/run_logs/${RESUME_SWEEP_ID}}"

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.11}"
export GPUS="${GPU_LIST}"
export ALGOS="genpo"
export SEEDS="0 1 2 3"
export TASKS="${TASK}"
export TRAIN_FRAMES=200000000
export EVAL_FRAMES=5000000
export RESUME_LOCAL=1
export SAVE_CKPT=1
export AUTO_RESUME=1
export INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
export ALGO_MAX_CONCURRENT="genpo=4"
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

export LOG_DIR
export MASTER_SWEEP_ID="${RESUME_SWEEP_ID}"
export MASTER_LOG_ROOT="${RESUME_LOG_ROOT}"
export RUN_NAME_SUFFIX
export LOG_TAG="${SOURCE_SWEEP_ID}"
export LOG_GROUP="${SOURCE_SWEEP_ID}"
export READY_FILE="${RESUME_LOG_ROOT}/launcher.ready"

if [[ "${DRY_RUN}" == "1" ]]; then
    exec bash "${QUEUE_WRAPPER}"
fi

mkdir -p "${RESUME_LOG_ROOT}"
LAUNCHER_LOG="${RESUME_LOG_ROOT}/launcher.log"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] resuming Lift-Cube GenPO seeds 0-3 on GPUs ${GPU_LIST}"
echo "Training artifacts: ${ARTIFACT_ROOT}/genpo/${SOURCE_SWEEP_ID}"
echo "Scheduler status: ${RESUME_LOG_ROOT}"
echo "Keep this shell alive; Ctrl-C requests checkpointed shutdown."

exec bash "${QUEUE_WRAPPER}"
