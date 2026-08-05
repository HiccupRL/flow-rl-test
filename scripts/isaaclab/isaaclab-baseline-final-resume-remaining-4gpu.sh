#!/usr/bin/env bash
set -euo pipefail

# Resume unfinished final IsaacLab baseline runs.
#
# This is a thin, reproducible wrapper around the complete-matrix launcher:
# it re-queries W&B, schedules only non-finished runs, and writes a fresh set
# of local logs/markers so the interrupted 2026-06-04 launch stays intact.
#
# Usage:
#   bash scripts/isaaclab/isaaclab-baseline-final-resume-remaining-4gpu.sh
#
# Useful overrides:
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-baseline-final-resume-remaining-4gpu.sh
#   GPUS="0 1 2 3" JOB_TIMEOUT_SECONDS=72000 bash scripts/isaaclab/isaaclab-baseline-final-resume-remaining-4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export GPUS="${GPUS:-0 1 2 3}"
export WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-baseline-final}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_STATES="${WANDB_RUN_STATES:-missing crashed failed killed running}"
export LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_baseline_final_remaining_12task_4gpu_10seed_200m}"
export LOG_TAG="${LOG_TAG:-isaaclab-baseline-final-200m}"
export TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
export GENPO_EXTRA_ARGS="${GENPO_EXTRA_ARGS:-algo.batch_size=4096 algo.num_minibatches=6}"

# Previous failures were mostly W&B SSL/init timeout and long post-training
# upload stalls. These env vars are understood by the installed W&B SDK.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_FILE_PUSHER_TIMEOUT="${WANDB_FILE_PUSHER_TIMEOUT:-300}"

# Keep enough headroom for slower IsaacLab tasks, but still fail wedged uploads.
export JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-72000}"

# Reuse the validated final protocol and W&B-resume implementation.
exec bash "${SCRIPT_DIR}/isaaclab-baseline-final-complete-12task-4gpu.sh"
