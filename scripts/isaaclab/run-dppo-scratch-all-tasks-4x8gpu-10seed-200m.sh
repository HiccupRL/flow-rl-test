#!/usr/bin/env bash
set -uo pipefail

# Seed-major DPPO-from-scratch IsaacLab sweep:
#   12 tasks * 1 best config * 10 seeds = 120 runs
#   4 shared-filesystem shards * 8 available GPUs
#   200M environment frames per run
#
# Jobs are queued seed-first: all seed-0 jobs precede all seed-1 jobs, and so
# on.  There is no completion barrier between seeds, so every GPU immediately
# refills from the next queued job.
#
# Launch once on each eight-GPU machine:
#   SHARD_ID=0 bash scripts/isaaclab/run-dppo-scratch-all-tasks-4x8gpu-10seed-200m.sh
#   SHARD_ID=1 bash scripts/isaaclab/run-dppo-scratch-all-tasks-4x8gpu-10seed-200m.sh
#   SHARD_ID=2 bash scripts/isaaclab/run-dppo-scratch-all-tasks-4x8gpu-10seed-200m.sh
#   SHARD_ID=3 bash scripts/isaaclab/run-dppo-scratch-all-tasks-4x8gpu-10seed-200m.sh
#
# The repository and SHARED_ROOT must resolve to the same shared filesystem on
# all four machines.  Use DRY_RUN=1 to inspect a shard without writing files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "This scheduler requires Bash >= 4.3." >&2
    exit 2
fi

PYTHON_BIN_REQUESTED="${PYTHON_BIN:-}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a CONFIGS <<<"${CONFIGS:-k10_e10}"
read -r -a TASKS <<<"${TASKS:-Isaac-Repose-Cube-Shadow-Direct-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Lift-Cube-Franka-v0 Isaac-Humanoid-v0 Isaac-Velocity-Rough-H1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Rough-Unitree-Go2-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Ant-v0 Isaac-Quadcopter-Direct-v0 Isaac-Cartpole-v0}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_ID="${SHARD_ID:-0}"
TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
SWEEP_ID="${SWEEP_ID:-isaaclab-dppo-scratch-alltasks-k10e10-10seed-4x8gpu-200m-v5}"
LOG_ROOT="${LOG_ROOT:-run_logs/${SWEEP_ID}/shard-${SHARD_ID}}"
SHARED_ROOT="${SHARED_ROOT:-run_logs/${SWEEP_ID}/coordination}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-${SWEEP_ID}}"
LOG_GROUP="${LOG_GROUP:-${SWEEP_ID}}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-dppo-scratch-alltasks-k10e10-10seed-200m-v5}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-dppo-scratch-200m}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
SAVE_CKPT="${SAVE_CKPT:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_LOCAL="${RESUME_LOCAL:-1}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
CHECK_GRAPHICS="${CHECK_GRAPHICS:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-40000}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# The public rollout contract remains identical to IsaacLab PPO/FPO.  The
# formal config uses the standard 10-step denoising chain and 10 update epochs.
declare -A CONFIG_ARGS=(
    [k10_e10]="algo.num_envs=1024 algo.rollout_length=24 algo.batch_size=6144 algo.diffusion.steps=10 algo.num_minibatches=4 algo.num_epochs=10"
)

declare -A CONFIG_DESCRIPTION=(
    [k10_e10]="K=10, 10 epochs; standard formal dppo_scratch configuration"
)

declare -A CONFIG_WEIGHT=(
    [k10_e10]=0
)

declare -A TASK_WEIGHT=(
    [Isaac-Repose-Cube-Shadow-Direct-v0]=1200
    [Isaac-Open-Drawer-Franka-v0]=1100
    [Isaac-Lift-Cube-Franka-v0]=1000
    [Isaac-Humanoid-v0]=900
    [Isaac-Velocity-Rough-H1-v0]=800
    [Isaac-Velocity-Rough-G1-v0]=780
    [Isaac-Velocity-Rough-Unitree-Go2-v0]=760
    [Isaac-Velocity-Flat-G1-v0]=700
    [Isaac-Velocity-Flat-Anymal-D-v0]=680
    [Isaac-Ant-v0]=500
    [Isaac-Quadcopter-Direct-v0]=400
    [Isaac-Cartpole-v0]=300
)

# Balanced static task placement.  Every shard receives three tasks and three
# runs per seed.
declare -A TASK_SHARD=(
    [Isaac-Repose-Cube-Shadow-Direct-v0]=0
    [Isaac-Ant-v0]=0
    [Isaac-Cartpole-v0]=0
    [Isaac-Open-Drawer-Franka-v0]=1
    [Isaac-Velocity-Rough-G1-v0]=1
    [Isaac-Quadcopter-Direct-v0]=1
    [Isaac-Lift-Cube-Franka-v0]=2
    [Isaac-Velocity-Rough-H1-v0]=2
    [Isaac-Velocity-Flat-Anymal-D-v0]=2
    [Isaac-Humanoid-v0]=3
    [Isaac-Velocity-Rough-Unitree-Go2-v0]=3
    [Isaac-Velocity-Flat-G1-v0]=3
)

declare -a JOBS=()
declare -a JOB_STATE=()
declare -a FREE_GPU=()
declare -a CMD=()
declare -A ACTIVE_GPU=()
declare -A ACTIVE_JOB=()

FAILED_RUNS_TOTAL=0
SKIPPED_DONE_TOTAL=0
STOPPING=0
TOTAL_GLOBAL=$(( ${#TASKS[@]} * ${#CONFIGS[@]} * ${#SEEDS[@]} ))
RUNS_PER_SEED=$(( ${#TASKS[@]} * ${#CONFIGS[@]} ))
RUNS_PER_SHARD=0
MAX_CONCURRENT_JOBS=0
COMMON_CONFIG_TEXT=""
SHARD_CONFIG_TEXT=""
SOURCE_FINGERPRINT=""

sanitize() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

is_bool() {
    [[ "$1" == "0" || "$1" == "1" ]]
}

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
    [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]
}

compute_source_fingerprint() {
    local source_file
    local -a source_files=(
        flowrl/agent/online/dppo.py
        flowrl/agent/online/dppo_scratch.py
        flowrl/config/online/algo/dppo.py
        flowrl/config/online/algo/dppo_scratch.py
        flowrl/flow/ddpm.py
        examples/online/main_isaaclab_onpolicy.py
        examples/online/config/isaaclab_onpolicy/algo/dppo_scratch.yaml
        scripts/isaaclab/run-dppo-scratch-all-tasks-4x8gpu-10seed-200m.sh
    )
    for source_file in "${source_files[@]}"; do
        [[ -f "${source_file}" ]] || {
            echo "Missing source fingerprint input: ${source_file}" >&2
            return 1
        }
    done
    SOURCE_FINGERPRINT="$(
        sha256sum "${source_files[@]}" |
            sha256sum |
            awk '{print $1}'
    )"
    [[ "${SOURCE_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ]]
}

preflight_python() {
    local summary
    if ! summary="$(PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" -c '
import sys
if sys.version_info < (3, 11):
    raise RuntimeError(f"Python >=3.11 is required, got {sys.version}")
import jax
import jaxlib
from examples.online.main_isaaclab_onpolicy import SUPPORTED_AGENTS
assert "dppo_scratch" in SUPPORTED_AGENTS
print(f"{sys.executable} | jax={jax.__version__} jaxlib={jaxlib.__version__}")
' 2>&1)"; then
        echo "Python preflight failed for ${PYTHON_BIN}" >&2
        printf '%s\n' "${summary}" >&2
        return 1
    fi
    echo "Python preflight: ${summary}"
}

preflight_graphics() {
    local summary
    [[ "${CHECK_GRAPHICS}" == "1" ]] || return 0
    if ! command -v vulkaninfo >/dev/null 2>&1; then
        echo "Warning: vulkaninfo unavailable; skipping graphics preflight." >&2
        return 0
    fi
    summary="$(vulkaninfo --summary 2>&1 || true)"
    if ! grep -Eiq 'vendorID[[:space:]]*= 0x10de|deviceName[[:space:]]*= NVIDIA' <<<"${summary}"; then
        echo "Isaac Sim graphics preflight failed: Vulkan cannot see an NVIDIA GPU." >&2
        return 1
    fi
    echo "Isaac Sim graphics preflight: NVIDIA Vulkan GPU available"
}

check_gpu_memory() {
    local gpu_index gpu line free_mb total_mb
    [[ "${CHECK_GPU_MEM}" == "1" ]] || return 0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the GPU memory preflight." >&2
        return 1
    fi
    for ((gpu_index = 0; gpu_index < MAX_CONCURRENT_JOBS; gpu_index++)); do
        gpu="${GPUS[${gpu_index}]}"
        line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free,memory.total \
            --format=csv,noheader,nounits 2>/dev/null || true)"
        if [[ -z "${line}" ]]; then
            echo "Could not query GPU ${gpu}." >&2
            return 1
        fi
        IFS=',' read -r free_mb total_mb <<<"${line}"
        free_mb="${free_mb//[[:space:]]/}"
        total_mb="${total_mb//[[:space:]]/}"
        echo "GPU ${gpu}: free=${free_mb} MiB total=${total_mb} MiB"
        if (( free_mb < MIN_FREE_MEM_MB )); then
            echo "GPU ${gpu} has less than ${MIN_FREE_MEM_MB} MiB free." >&2
            return 1
        fi
    done
}

validate_inputs() {
    local gpu task seed config flag value assigned_shard shard_count seed_index
    local -A seen_gpu=()
    local -A seen_task=()
    local -A seen_seed=()
    local -A seen_config=()
    local -a shard_task_count=(0 0 0 0)

    if (( NUM_SHARDS != 4 )); then
        echo "This four-machine scheduler requires NUM_SHARDS=4." >&2
        return 1
    fi
    if ! [[ "${SHARD_ID}" =~ ^[0-3]$ ]]; then
        echo "SHARD_ID must be one of 0, 1, 2, 3; got ${SHARD_ID}." >&2
        return 1
    fi
    if (( ${#GPUS[@]} != 8 )); then
        echo "Each shard requires exactly 8 GPUs; got: ${GPUS[*]}" >&2
        return 1
    fi
    if (( ${#TASKS[@]} != 12 )); then
        echo "Expected all 12 IsaacLab tasks; got ${#TASKS[@]}." >&2
        return 1
    fi
    if (( ${#CONFIGS[@]} != 1 )) || [[ "${CONFIGS[0]}" != "k10_e10" ]]; then
        echo "This formal sweep is locked to the standard config: CONFIGS='k10_e10'." >&2
        return 1
    fi
    if (( ${#SEEDS[@]} != 10 )); then
        echo "Expected ten seeds (0--9); got ${#SEEDS[@]}." >&2
        return 1
    fi
    RUNS_PER_SHARD=$(( TOTAL_GLOBAL / NUM_SHARDS ))
    MAX_CONCURRENT_JOBS="${RUNS_PER_SHARD}"
    if (( MAX_CONCURRENT_JOBS > ${#GPUS[@]} )); then
        MAX_CONCURRENT_JOBS="${#GPUS[@]}"
    fi

    for gpu in "${GPUS[@]}"; do
        if ! [[ "${gpu}" =~ ^[0-9]+$ ]] || [[ -v "seen_gpu[${gpu}]" ]]; then
            echo "GPU ids must be unique non-negative integers: ${GPUS[*]}" >&2
            return 1
        fi
        seen_gpu[${gpu}]=1
    done
    for seed in "${SEEDS[@]}"; do
        if ! [[ "${seed}" =~ ^[0-9]+$ ]] || [[ -v "seen_seed[${seed}]" ]]; then
            echo "Seeds must be unique non-negative integers: ${SEEDS[*]}" >&2
            return 1
        fi
        seen_seed[${seed}]=1
    done
    for seed in {0..9}; do
        if [[ ! -v "seen_seed[${seed}]" ]]; then
            echo "SEEDS must contain every seed from 0 through 9." >&2
            return 1
        fi
    done
    for seed_index in "${!SEEDS[@]}"; do
        if (( SEEDS[seed_index] != seed_index )); then
            echo "SEEDS must be ordered exactly as: 0 1 2 3 4 5 6 7 8 9." >&2
            return 1
        fi
    done
    for task in "${TASKS[@]}"; do
        if [[ -v "seen_task[${task}]" ]]; then
            echo "Duplicate task: ${task}" >&2
            return 1
        fi
        seen_task[${task}]=1
        if [[ ! -f "examples/online/config/isaaclab_onpolicy/task/${task}.yaml" ]]; then
            echo "Missing task config: ${task}" >&2
            return 1
        fi
        if [[ ! -v "TASK_WEIGHT[${task}]" || ! -v "TASK_SHARD[${task}]" ]]; then
            echo "Missing scheduling metadata for task: ${task}" >&2
            return 1
        fi
        assigned_shard="${TASK_SHARD[${task}]}"
        if ! [[ "${assigned_shard}" =~ ^[0-3]$ ]]; then
            echo "Invalid shard assignment for ${task}: ${assigned_shard}." >&2
            return 1
        fi
        shard_task_count[${assigned_shard}]=$((shard_task_count[${assigned_shard}] + 1))
    done
    for shard_count in "${shard_task_count[@]}"; do
        if (( shard_count != 3 )); then
            echo "Each shard must own exactly three tasks; counts=${shard_task_count[*]}." >&2
            return 1
        fi
    done
    for config in "${CONFIGS[@]}"; do
        if [[ -v "seen_config[${config}]" ]]; then
            echo "Duplicate config: ${config}" >&2
            return 1
        fi
        seen_config[${config}]=1
        if [[ ! -v "CONFIG_ARGS[${config}]" ||
              ! -v "CONFIG_DESCRIPTION[${config}]" ||
              ! -v "CONFIG_WEIGHT[${config}]" ]]; then
            echo "Unknown or incomplete config: ${config}" >&2
            return 1
        fi
    done
    for flag in WANDB_ENABLED SAVE_CKPT AUTO_RESUME RESUME_LOCAL \
                CHECK_GPU_MEM CHECK_GRAPHICS DRY_RUN; do
        value="${!flag}"
        if ! is_bool "${value}"; then
            echo "${flag} must be 0 or 1, got ${value}." >&2
            return 1
        fi
    done
    if ! [[ "${TRAIN_FRAMES}" =~ ^[1-9][0-9]*$ &&
            "${EVAL_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_FRAMES and EVAL_FRAMES must be positive integers." >&2
        return 1
    fi
    if ! [[ "${MIN_FREE_MEM_MB}" =~ ^[0-9]+$ &&
            "${JOB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ &&
            "${INTERRUPT_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "Invalid memory, timeout, or grace setting." >&2
        return 1
    fi
    if (( JOB_TIMEOUT_SECONDS > 0 )) && ! command -v timeout >/dev/null 2>&1; then
        echo "timeout is required when JOB_TIMEOUT_SECONDS is nonzero." >&2
        return 1
    fi
}

run_name_for() {
    local task="$1"
    local config="$2"
    local seed="$3"
    printf '%s-dppo-scratch-%s-seed%s-%s\n' \
        "${task}" "${config}" "${seed}" "${RUN_NAME_SUFFIX}"
}

task_log_root() {
    printf '%s/%s\n' "${LOG_ROOT}" "$(sanitize "$1")"
}

done_file_for() {
    local task="$1"
    local run_name="$2"
    printf '%s/%s.done\n' \
        "$(task_log_root "${task}")" "$(sanitize "${run_name}")"
}

has_resumable_checkpoint() {
    local task="$1"
    local run_name="$2"
    local marker
    local checkpoint_root="${LOG_DIR}/dppo_scratch/${LOG_TAG}/${task}/${run_name}/ckpt"
    for marker in "${checkpoint_root}"/[0-9]*/_SUCCESS; do
        [[ -f "${marker}" ]] && return 0
    done
    return 1
}

build_config_texts() {
    local config
    COMMON_CONFIG_TEXT="$(
        printf 'sweep_id=%s\n' "${SWEEP_ID}"
        printf 'num_shards=%s\n' "${NUM_SHARDS}"
        printf 'tasks=%s\n' "${TASKS[*]}"
        printf 'configs=%s\n' "${CONFIGS[*]}"
        printf 'seeds=%s\n' "${SEEDS[*]}"
        printf 'train_frames=%s\n' "${TRAIN_FRAMES}"
        printf 'eval_frames=%s\n' "${EVAL_FRAMES}"
        printf 'queue_order=seed_priority_without_completion_barrier\n'
        printf 'log_dir=%s\n' "${LOG_DIR}"
        printf 'log_tag=%s\n' "${LOG_TAG}"
        printf 'run_name_suffix=%s\n' "${RUN_NAME_SUFFIX}"
        printf 'source_fingerprint=%s\n' "${SOURCE_FINGERPRINT}"
        printf 'extra_args=%s\n' "${EXTRA_ARGS}"
        for config in "${CONFIGS[@]}"; do
            printf 'config_%s=%s\n' "${config}" "${CONFIG_ARGS[${config}]}"
        done
    )"
    SHARD_CONFIG_TEXT="$(
        printf '%s\n' "${COMMON_CONFIG_TEXT}"
        printf 'shard_id=%s\n' "${SHARD_ID}"
        printf 'gpus=%s\n' "${GPUS[*]}"
        printf 'log_root=%s\n' "${LOG_ROOT}"
        printf 'shared_root=%s\n' "${SHARED_ROOT}"
    )"
}

validate_local_config() {
    local config_file="${LOG_ROOT}/sweep-config.txt"
    if [[ -f "${config_file}" ]] &&
       ! cmp -s <(printf '%s\n' "${SHARD_CONFIG_TEXT}") "${config_file}"; then
        echo "Sweep configuration differs from ${config_file}." >&2
        diff -u "${config_file}" <(printf '%s\n' "${SHARD_CONFIG_TEXT}") >&2 || true
        echo "Use a new SWEEP_ID for a different experiment." >&2
        return 1
    fi
}

initialize_shared_manifest() {
    local manifest="${SHARED_ROOT}/manifest.txt"
    local lock_dir="${SHARED_ROOT}/.manifest.lock"
    local deadline=$((SECONDS + 120))
    mkdir -p "${SHARED_ROOT}"

    while [[ ! -f "${manifest}" ]]; do
        if mkdir "${lock_dir}" 2>/dev/null; then
            printf '%s\n' "${COMMON_CONFIG_TEXT}" >"${manifest}.tmp-${SHARD_ID}-$$"
            mv "${manifest}.tmp-${SHARD_ID}-$$" "${manifest}"
            rmdir "${lock_dir}" 2>/dev/null || true
            break
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for shared manifest lock: ${lock_dir}" >&2
            return 1
        fi
        sleep 2
    done

    if ! cmp -s <(printf '%s\n' "${COMMON_CONFIG_TEXT}") "${manifest}"; then
        echo "Shared sweep configuration differs from ${manifest}." >&2
        diff -u "${manifest}" <(printf '%s\n' "${COMMON_CONFIG_TEXT}") >&2 || true
        echo "All four machines must use identical source and settings." >&2
        return 1
    fi
}

prepare_all_jobs() {
    local task config run_name done_file weight assigned_shard task_weight config_weight
    local -a records=()
    local -a seed_jobs=()
    local seed

    JOBS=()
    SKIPPED_DONE_TOTAL=0
    for seed in "${SEEDS[@]}"; do
        records=()
        seed_jobs=()
        for task in "${TASKS[@]}"; do
            assigned_shard="${TASK_SHARD[${task}]}"
            (( assigned_shard == SHARD_ID )) || continue
            for config in "${CONFIGS[@]}"; do
                run_name="$(run_name_for "${task}" "${config}" "${seed}")"
                done_file="$(done_file_for "${task}" "${run_name}")"
                if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
                    SKIPPED_DONE_TOTAL=$((SKIPPED_DONE_TOTAL + 1))
                    continue
                fi
                task_weight="${TASK_WEIGHT[${task}]}"
                config_weight="${CONFIG_WEIGHT[${config}]}"
                weight=$(( task_weight + config_weight ))
                if has_resumable_checkpoint "${task}" "${run_name}"; then
                    weight=$((weight + 10000))
                fi
                records+=("${weight}"$'\t'"${task}"$'\t'"${config}"$'\t'"${seed}")
            done
        done

        if (( ${#records[@]} > 0 )); then
            mapfile -t seed_jobs < <(
                printf '%s\n' "${records[@]}" |
                    LC_ALL=C sort -s -t $'\t' -k1,1nr
            )
            JOBS+=("${seed_jobs[@]}")
        fi
    done

    if (( ${#JOBS[@]} + SKIPPED_DONE_TOTAL != RUNS_PER_SHARD )); then
        echo "Shard ${SHARD_ID} has unexpected run coverage: " \
             "pending=${#JOBS[@]} skipped=${SKIPPED_DONE_TOTAL} " \
             "expected=${RUNS_PER_SHARD}." >&2
        return 1
    fi
}

build_command() {
    local task="$1"
    local config="$2"
    local seed="$3"
    local run_name="$4"
    local -a config_args=()
    local -a extra_args=()

    read -r -a config_args <<<"${CONFIG_ARGS[${config}]}"
    if [[ -n "${EXTRA_ARGS//[[:space:]]/}" ]]; then
        read -r -a extra_args <<<"${EXTRA_ARGS}"
    fi

    CMD=(
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
        "log.tags=[${task},dppo_scratch,isaaclab,${config},seed${seed},10seed,best-config,4x8gpu,200m,identity-score-init,critic-warmup]"
        "log.wandb_mode=${WANDB_MODE}"
        "log.save_ckpt=${SAVE_CKPT}"
        "log.resume=${AUTO_RESUME}"
        "${config_args[@]}"
        "${extra_args[@]}"
    )
    if [[ "${WANDB_ENABLED}" == "1" ]]; then
        CMD+=(
            "log.wandb=true"
            "log.project=${WANDB_PROJECT_NAME}"
            "log.entity=${WANDB_ENTITY}"
        )
    else
        CMD+=("log.wandb=false")
    fi
}

print_plan() {
    local config task assigned_shard
    echo "DPPO scratch IsaacLab seed-major 4x8-GPU sweep"
    echo "Global matrix: ${TOTAL_GLOBAL} runs"
    echo "Per seed: ${RUNS_PER_SEED} runs; seed-priority queue without completion barrier"
    echo "Shard: $((SHARD_ID + 1))/${NUM_SHARDS} (SHARD_ID=${SHARD_ID})"
    echo "Shard tasks:"
    for task in "${TASKS[@]}"; do
        assigned_shard="${TASK_SHARD[${task}]}"
        if (( assigned_shard == SHARD_ID )); then
            echo "  ${task}"
        fi
    done
    echo "Seeds: ${SEEDS[*]}"
    echo "Available GPUs: ${GPUS[*]}"
    echo "Active GPUs on this shard: ${GPUS[*]:0:${MAX_CONCURRENT_JOBS}}"
    echo "Train/eval frames: ${TRAIN_FRAMES}/${EVAL_FRAMES}"
    echo "Algorithm: dppo_scratch"
    echo "Task bootstrap: inherited from each task YAML"
    echo "Checkpoint/resume: save=${SAVE_CKPT} auto_resume=${AUTO_RESUME}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} mode=${WANDB_MODE}"
    echo "Local logs: ${LOG_ROOT}"
    echo "Shared coordination: ${SHARED_ROOT}"
    echo "Source fingerprint: ${SOURCE_FINGERPRINT}"
    echo "Configs:"
    for config in "${CONFIGS[@]}"; do
        echo "  ${config}: ${CONFIG_DESCRIPTION[${config}]}"
        echo "    ${CONFIG_ARGS[${config}]}"
    done
}

print_dry_run() {
    local index weight task config job_seed run_name
    for index in "${!JOBS[@]}"; do
        IFS=$'\t' read -r weight task config job_seed <<<"${JOBS[${index}]}"
        run_name="$(run_name_for "${task}" "${config}" "${job_seed}")"
        build_command "${task}" "${config}" "${job_seed}" "${run_name}"
        printf 'seed_wave=%q queue_rank=%q weight=%q gpu=assigned_at_runtime ' \
            "${job_seed}" "$((index + 1))" "${weight}"
        printf 'CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=%q ' "${PYTHONPATH_VALUE}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    done
}

validate_finished_log() {
    local log_file="$1"
    grep -Fq "training: 100%" "${log_file}" &&
        grep -Fq "train_frames: ${TRAIN_FRAMES}" "${log_file}" &&
        grep -Fq "name: dppo_scratch" "${log_file}"
}

run_one() {
    local task="$1"
    local config="$2"
    local seed="$3"
    local gpu="$4"
    local rank="$5"
    local run_name task_root safe_name log_file done_file failed_file
    local start_ts status elapsed archived_log marker_tmp

    run_name="$(run_name_for "${task}" "${config}" "${seed}")"
    task_root="$(task_log_root "${task}")"
    safe_name="$(sanitize "${run_name}")"
    log_file="${task_root}/${run_name}.gpu${gpu}.log"
    done_file="${task_root}/${safe_name}.done"
    failed_file="${task_root}/${safe_name}.failed"
    mkdir -p "${task_root}"

    if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
        echo "[$(date '+%F %T')] skip done | seed=${seed} rank=${rank} gpu=${gpu} task=${task} config=${config}"
        return 0
    fi
    if [[ -f "${log_file}" ]]; then
        archived_log="${log_file}.attempt-$(date '+%Y%m%d-%H%M%S')"
        mv "${log_file}" "${archived_log}"
    fi

    build_command "${task}" "${config}" "${seed}" "${run_name}"
    echo "[$(date '+%F %T')] start | seed=${seed} rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} config=${config}"
    start_ts="$(date +%s)"
    if (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EGL_VISIBLE_DEVICES="${gpu}"
        export PYTHONPATH="${PYTHONPATH_VALUE}"
        export WANDB_MODE="${WANDB_MODE}"
        export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
        export WANDB_ENTITY="${WANDB_ENTITY}"
        export OMNI_KIT_ACCEPT_EULA=YES
        export FLOWRL_ISAACLAB_CLOSE_APP=0
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONUNBUFFERED=1
        if (( JOB_TIMEOUT_SECONDS > 0 )); then
            timeout --preserve-status --signal=TERM --kill-after=120s \
                "${JOB_TIMEOUT_SECONDS}" "${CMD[@]}"
        else
            "${CMD[@]}"
        fi
    ) >"${log_file}" 2>&1; then
        status=0
    else
        status=$?
    fi
    elapsed=$(( $(date +%s) - start_ts ))

    if (( status == 0 )) && ! validate_finished_log "${log_file}"; then
        status=90
    fi
    if (( status == 0 )); then
        marker_tmp="${done_file}.tmp.$$"
        {
            echo "finished_at=$(date --iso-8601=seconds)"
            echo "gpu=${gpu}"
            echo "task=${task}"
            echo "config=${config}"
            echo "seed=${seed}"
            echo "elapsed_seconds=${elapsed}"
            echo "source_fingerprint=${SOURCE_FINGERPRINT}"
        } >"${marker_tmp}"
        mv "${marker_tmp}" "${done_file}"
        if [[ -f "${failed_file}" ]]; then
            mv "${failed_file}" "${failed_file}.recovered-$(date '+%Y%m%d-%H%M%S')"
        fi
        echo "[$(date '+%F %T')] done | seed=${seed} rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} config=${config} elapsed=${elapsed}s"
        return 0
    fi

    {
        echo "time=$(date --iso-8601=seconds)"
        echo "gpu=${gpu}"
        echo "task=${task}"
        echo "config=${config}"
        echo "seed=${seed}"
        echo "exit_code=${status}"
        echo "elapsed_seconds=${elapsed}"
        echo "log_file=${log_file}"
        echo "source_fingerprint=${SOURCE_FINGERPRINT}"
    } >"${failed_file}"
    echo "[$(date '+%F %T')] failed | seed=${seed} rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} config=${config} exit=${status}" >&2
    tail -n 40 "${log_file}" >&2 || true
    return "${status}"
}

kill_tree() {
    local parent="$1"
    local signal="$2"
    local child children=""
    if [[ -r "/proc/${parent}/task/${parent}/children" ]]; then
        read -r children <"/proc/${parent}/task/${parent}/children" || true
    fi
    for child in ${children}; do
        kill_tree "${child}" "${signal}"
    done
    kill "-${signal}" "${parent}" 2>/dev/null || true
}

SIGNAL_TARGETS=0
signal_python_descendants() {
    local parent="$1"
    local signal="$2"
    local child children="" comm=""
    if [[ -r "/proc/${parent}/task/${parent}/children" ]]; then
        read -r children <"/proc/${parent}/task/${parent}/children" || true
    fi
    for child in ${children}; do
        if [[ -r "/proc/${child}/comm" ]]; then
            read -r comm <"/proc/${child}/comm" || true
        fi
        if [[ "${comm}" == python || "${comm}" == python3 || "${comm}" == python3.* ]]; then
            kill "-${signal}" "${child}" 2>/dev/null || true
            SIGNAL_TARGETS=$((SIGNAL_TARGETS + 1))
        else
            signal_python_descendants "${child}" "${signal}"
        fi
    done
}

stop_all() {
    local status="$1"
    local pid deadline any_alive
    local -a active_pids=("${!ACTIVE_GPU[@]}")
    if (( STOPPING != 0 )); then
        for pid in "${active_pids[@]}"; do
            kill_tree "${pid}" KILL
        done
        exit "${status}"
    fi
    STOPPING=1
    trap - INT TERM
    echo "[$(date '+%F %T')] requesting checkpointed shutdown ..." >&2
    for pid in "${active_pids[@]}"; do
        SIGNAL_TARGETS=0
        signal_python_descendants "${pid}" TERM
        if (( SIGNAL_TARGETS == 0 )); then
            kill_tree "${pid}" TERM
        fi
    done
    deadline=$((SECONDS + INTERRUPT_GRACE_SECONDS))
    while (( SECONDS < deadline )); do
        any_alive=0
        for pid in "${active_pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && any_alive=1
        done
        (( any_alive == 0 )) && break
        sleep 1
    done
    for pid in "${active_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill_tree "${pid}" KILL
        fi
        wait "${pid}" 2>/dev/null || true
    done
    exit "${status}"
}

trap 'stop_all 130' INT
trap 'stop_all 143' TERM

launch_job() {
    local gpu_index="$1"
    local job_index="$2"
    local weight task config seed gpu pid
    IFS=$'\t' read -r weight task config seed <<<"${JOBS[${job_index}]}"
    gpu="${GPUS[${gpu_index}]}"
    JOB_STATE[${job_index}]=1
    FREE_GPU[${gpu_index}]=0
    run_one "${task}" "${config}" "${seed}" "${gpu}" "$((job_index + 1))" &
    pid=$!
    ACTIVE_GPU[${pid}]="${gpu_index}"
    ACTIVE_JOB[${pid}]="${job_index}"
}

find_next_job() {
    local index
    for index in "${!JOBS[@]}"; do
        if (( ${JOB_STATE[${index}]:-0} == 0 )); then
            printf '%s\n' "${index}"
            return 0
        fi
    done
    return 1
}

fill_free_gpus() {
    local gpu_index job_index
    for gpu_index in "${!GPUS[@]}"; do
        (( ${FREE_GPU[${gpu_index}]:-0} == 1 )) || continue
        if job_index="$(find_next_job)"; then
            launch_job "${gpu_index}" "${job_index}"
        fi
    done
}

run_scheduler() {
    local index pid state finished_pid finished_gpu finished_job rc
    JOB_STATE=()
    FREE_GPU=()
    ACTIVE_GPU=()
    ACTIVE_JOB=()
    FAILED_RUNS_TOTAL=0

    for index in "${!JOBS[@]}"; do
        JOB_STATE[${index}]=0
    done
    for index in "${!GPUS[@]}"; do
        FREE_GPU[${index}]=1
    done
    fill_free_gpus

    while (( ${#ACTIVE_GPU[@]} > 0 )); do
        finished_pid=""
        rc=0
        while [[ -z "${finished_pid}" ]]; do
            for pid in "${!ACTIVE_GPU[@]}"; do
                state=""
                if [[ -r "/proc/${pid}/stat" ]]; then
                    read -r _ _ state _ <"/proc/${pid}/stat" || true
                fi
                if [[ -z "${state}" || "${state}" == "Z" ]]; then
                    finished_pid="${pid}"
                    if wait "${pid}"; then
                        rc=0
                    else
                        rc=$?
                    fi
                    break
                fi
            done
            [[ -n "${finished_pid}" ]] || sleep 2
        done

        finished_gpu="${ACTIVE_GPU[${finished_pid}]}"
        finished_job="${ACTIVE_JOB[${finished_pid}]}"
        unset "ACTIVE_GPU[${finished_pid}]" "ACTIVE_JOB[${finished_pid}]"
        FREE_GPU[${finished_gpu}]=1
        JOB_STATE[${finished_job}]=2
        if (( rc != 0 )); then
            FAILED_RUNS_TOTAL=$((FAILED_RUNS_TOTAL + 1))
        fi
        fill_free_gpus
    done
}

archive_own_failed_marker() {
    local failed_marker="${SHARED_ROOT}/shard-${SHARD_ID}.failed"
    if [[ -f "${failed_marker}" ]]; then
        mv "${failed_marker}" \
            "${failed_marker}.retry-$(date '+%Y%m%d-%H%M%S')"
    fi
}

write_shard_status_marker() {
    local status="$1"
    local marker="${SHARED_ROOT}/shard-${SHARD_ID}.${status}"
    local temporary="${marker}.tmp-$$"
    mkdir -p "${SHARED_ROOT}"
    {
        echo "time=$(date --iso-8601=seconds)"
        echo "shard_id=${SHARD_ID}"
        echo "status=${status}"
        echo "scheduled_runs=${#JOBS[@]}"
        echo "skipped_done_runs=${SKIPPED_DONE_TOTAL}"
        echo "failed_runs=${FAILED_RUNS_TOTAL}"
        echo "source_fingerprint=${SOURCE_FINGERPRINT}"
    } >"${temporary}"
    mv "${temporary}" "${marker}"
}

main() {
    validate_inputs || exit 2
    compute_source_fingerprint || exit 2
    build_config_texts
    prepare_all_jobs || exit 2

    if [[ "${DRY_RUN}" == "1" ]]; then
        PYTHON_BIN="${PYTHON_BIN_REQUESTED:-python3}"
        print_plan
        print_dry_run
        exit 0
    fi

    resolve_python || exit 2
    preflight_python || exit 2
    preflight_graphics || exit 2
    check_gpu_memory || exit 2
    validate_local_config || exit 2
    initialize_shared_manifest || exit 2
    mkdir -p "${LOG_ROOT}"
    printf '%s\n' "${SHARD_CONFIG_TEXT}" >"${LOG_ROOT}/sweep-config.txt"

    print_plan
    echo "[$(date '+%F %T')] shard ${SHARD_ID}: pending=${#JOBS[@]} skipped=${SKIPPED_DONE_TOTAL}"
    archive_own_failed_marker
    if (( ${#JOBS[@]} > 0 )); then
        run_scheduler
    else
        FAILED_RUNS_TOTAL=0
    fi
    if (( FAILED_RUNS_TOTAL > 0 )); then
        write_shard_status_marker failed
        echo "Shard ${SHARD_ID} failed ${FAILED_RUNS_TOTAL} run(s); all other queued runs were still attempted." >&2
        exit 1
    fi
    write_shard_status_marker done
    echo "Sweep finished: ${TOTAL_GLOBAL} runs represented, shard=${SHARD_ID}/${NUM_SHARDS}."
}

main "$@"
