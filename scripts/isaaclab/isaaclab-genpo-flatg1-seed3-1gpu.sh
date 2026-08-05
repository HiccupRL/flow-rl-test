#!/usr/bin/env bash
set -Eeuo pipefail

# Resume Isaac-Velocity-Flat-G1-v0 GenPO seed 3 on one GPU. This keeps the
# original run name, W&B run, and checkpoint tree while using a scheduler-status
# root independent from the four-GPU Lift-Cube launcher.
#
# Usage:
#   bash scripts/isaaclab/isaaclab-genpo-flatg1-seed3-1gpu.sh
#
# Optional GPU remapping (exactly one id is required):
#   GPUS="2" bash scripts/isaaclab/isaaclab-genpo-flatg1-seed3-1gpu.sh
#
# Inspect the command without launching:
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-genpo-flatg1-seed3-1gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUEUE_WRAPPER="${SCRIPT_DIR}/isaaclab-all-tasks-6baseline-4seed-balanced.sh"

SOURCE_SWEEP_ID="isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1"
RESUME_SWEEP_ID="isaaclab-genpo-flatg1-seed3-1gpu-resume-v1"
RUN_NAME_SUFFIX="smoothness-alltasks-4seed-8gpu-clean-v1"
TASK="Isaac-Velocity-Flat-G1-v0"
SEED=3
GPU_LIST="${GPUS:-0}"
read -r -a GPU_IDS <<<"${GPU_LIST}"

if (( ${#GPU_IDS[@]} != 1 )); then
    echo "This launcher requires exactly one GPU id; got: ${GPU_LIST}" >&2
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

RUN_NAME="${TASK}-genpo-seed${SEED}-${RUN_NAME_SUFFIX}"
CHECKPOINT_ROOT="${ARTIFACT_ROOT}/genpo/${SOURCE_SWEEP_ID}/${TASK}/${RUN_NAME}/ckpt"
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "Missing checkpoint directory: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi
LATEST_CHECKPOINT="$(
    find "${CHECKPOINT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name _SUCCESS -printf '%h\n' |
        sed 's#.*/##' |
        sort -n |
        tail -1
)"
if [[ -z "${LATEST_CHECKPOINT}" ]]; then
    echo "No complete checkpoint found: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi
echo "Flat-G1 seed 3: resume checkpoint ${LATEST_CHECKPOINT}/200000000"

RESUME_LOG_ROOT="${RESUME_LOG_ROOT:-${REPO_ROOT}/run_logs/${RESUME_SWEEP_ID}}"

cd "${REPO_ROOT}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.11}"
export GPUS="${GPU_LIST}"
export ALGOS="genpo"
export SEEDS="${SEED}"
export TASKS="${TASK}"
export TRAIN_FRAMES=200000000
export EVAL_FRAMES=5000000
export RESUME_LOCAL=1
export SAVE_CKPT=1
export AUTO_RESUME=1
export INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
export ALGO_MAX_CONCURRENT="genpo=1"
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

echo "[$(date --iso-8601=seconds)] resuming Flat-G1 GenPO seed 3 on GPU ${GPU_LIST}"
echo "Training artifacts: ${ARTIFACT_ROOT}/genpo/${SOURCE_SWEEP_ID}"
echo "Scheduler status: ${RESUME_LOG_ROOT}"
echo "Keep this shell alive; Ctrl-C requests checkpointed shutdown."

exec bash "${QUEUE_WRAPPER}"
