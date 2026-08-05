#!/usr/bin/env bash
set -euo pipefail

# Reproducible clean launch for the complete action-smoothness matrix:
#   12 tasks * 6 algorithms * 4 seeds = 288 runs on 8 GPUs.
#
# This wrapper intentionally fixes the Python executable and experiment matrix.
# RESUME_LOCAL stays enabled so the same command can safely continue an
# interrupted clean sweep from its local .done markers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHON_BIN=/usr/local/bin/python3.11
export GPUS="0 1 2 3 4 5 6 7"
export ALGOS="ppo dppo fpo fpopp genpo policyflow"
export SEEDS="0 1 2 3"
export TASKS="Isaac-Repose-Cube-Shadow-Direct-v0 Isaac-Velocity-Rough-H1-v0 Isaac-Humanoid-v0 Isaac-Velocity-Rough-Unitree-Go2-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Ant-v0 Isaac-Quadcopter-Direct-v0 Isaac-Cartpole-v0 Isaac-Lift-Cube-Franka-v0"

export TRAIN_FRAMES=200000000
export EVAL_FRAMES=5000000
export RESUME_LOCAL=1
export SAVE_CKPT=1
export AUTO_RESUME=1
export INTERRUPT_GRACE_SECONDS=180
export ALGO_MAX_CONCURRENT="${ALGO_MAX_CONCURRENT:-genpo=4}"
export REUSE_EXISTING_LIFT=0
export GENPO_EXTRA_ARGS="algo.batch_size=4096 algo.num_minibatches=6"
export EXTRA_ARGS=""
export JOB_TIMEOUT_SECONDS=0
export CHECK_GPU_MEM=1
export MIN_FREE_MEM_MB=40000

export WANDB_ENTITY=hiccupnudt
export WANDB_PROJECT_NAME=isaaclab-onpolicy-smoothness
export WANDB_MODE=online
export WANDB_ENABLED=1

export MASTER_SWEEP_ID=isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1
export MASTER_LOG_ROOT=run_logs/isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1
export RUN_NAME_SUFFIX=smoothness-alltasks-4seed-8gpu-clean-v1
export LOG_TAG=isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1
export LOG_GROUP=isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1

exec bash "${SCRIPT_DIR}/isaaclab-all-tasks-6baseline-4seed-balanced.sh"
