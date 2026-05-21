#!/usr/bin/env bash
set -euo pipefail

# Rerun the GENPO jobs that failed in isaaclab_genpo_baseline_4gpu_10seed.
#
# Failure set:
#   Isaac-Velocity-Flat-G1-v0  seeds 0..9
#   Isaac-Velocity-Rough-G1-v0 seeds 0..9
#
# Usage:
#   WANDB_ENTITY=hiccupnudt GPUS="0 1 2 3" \
#     bash scripts/isaaclab/isaaclab-genpo-rerun-failed-g1-4gpu-10seed.sh
#
# The failed logs showed a single XLA allocation around 39.0-39.6 GiB, so this
# wrapper checks free VRAM before launching. Override with ALLOW_LOW_MEM=1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-0 1 2 3}"
TASKS="${TASKS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

export GPUS TASKS SEEDS
export WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-ppo-baseline}"
export WANDB_MODE="${WANDB_MODE:-online}"
export LOG_GROUP="${LOG_GROUP:-isaaclab-genpo-baseline-rerun-g1}"
export LOG_TAG="${LOG_TAG:-genpo-baseline-10seed-rerun-g1}"
export LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_genpo_failed_g1_rerun_4gpu_10seed}"
export TRAIN_FRAMES="${TRAIN_FRAMES:-100000000}"
export RESUME_WANDB="${RESUME_WANDB:-1}"
export ALLOW_WANDB_RESUME_FALLBACK="${ALLOW_WANDB_RESUME_FALLBACK:-0}"
export EXTRA_ARGS="${EXTRA_ARGS:-algo.batch_size=4096 algo.num_minibatches=6}"

CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
ALLOW_LOW_MEM="${ALLOW_LOW_MEM:-0}"

check_gpu_memory() {
    if [[ "${CHECK_GPU_MEM}" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; skipping free-memory check." >&2
        return 0
    fi

    local low_mem=0
    local gpu
    for gpu in ${GPUS}; do
        local line
        line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        if [[ -z "${line}" ]]; then
            echo "Could not query GPU ${gpu} memory with nvidia-smi." >&2
            low_mem=1
            continue
        fi

        local free_mb total_mb
        IFS=',' read -r free_mb total_mb <<<"${line}"
        free_mb="${free_mb//[[:space:]]/}"
        total_mb="${total_mb//[[:space:]]/}"

        echo "GPU ${gpu}: free=${free_mb} MiB total=${total_mb} MiB"
        if (( free_mb < MIN_FREE_MEM_MB )); then
            echo "GPU ${gpu} has less than MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB} MiB free." >&2
            low_mem=1
        fi
    done

    if (( low_mem != 0 && ALLOW_LOW_MEM != 1 )); then
        echo "Aborting before launch. Set ALLOW_LOW_MEM=1 to run anyway." >&2
        exit 2
    fi
}

check_gpu_memory
exec bash "${SCRIPT_DIR}/isaaclab-genpo-baseline-4gpu-10seed.sh"
