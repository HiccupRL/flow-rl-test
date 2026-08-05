#!/usr/bin/env bash
set -euo pipefail

# IsaacLab action-smoothness comparison with a global cross-task queue.
#
# Matrix:
#   N tasks * 6 algorithms * 4 seeds.
#
# Scheduling:
#   One IsaacLab process per configured GPU. All task/algo/seed combinations
#   share one queue. Per-algorithm concurrency limits prevent slow algorithms
#   from monopolizing every GPU while dynamic refill avoids task barriers.
#
# Usage:
#   bash scripts/isaaclab/isaaclab-lift-cube-6baseline-4seed-balanced.sh
#
# Dry-run (no files or GPU processes are created):
#   DRY_RUN=1 bash scripts/isaaclab/isaaclab-lift-cube-6baseline-4seed-balanced.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "This scheduler requires Bash >= 4.3." >&2
    exit 2
fi

PYTHON_BIN_REQUESTED="${PYTHON_BIN:-}"
read -r -a TASKS <<<"${TASKS:-${TASK:-Isaac-Lift-Cube-Franka-v0}}"
read -r -a ALGOS <<<"${ALGOS:-ppo dppo fpo fpopp genpo policyflow}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2 3}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3}"

TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
SWEEP_ID="${SWEEP_ID:-isaaclab-lift-cube-smoothness-200m-4seed-v1}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-smoothness-4seed-v1}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-${SWEEP_ID}}"
LOG_GROUP="${LOG_GROUP:-${SWEEP_ID}}"
LOG_ROOT="${LOG_ROOT:-run_logs/${SWEEP_ID}}"
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
PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_PREFLIGHT_DONE="${PYTHON_PREFLIGHT_DONE:-0}"
GRAPHICS_PREFLIGHT_DONE="${GRAPHICS_PREFLIGHT_DONE:-0}"
READY_FILE="${READY_FILE:-}"

# Relative Lift-Cube scheduling weights. Only their ordering matters; the
# dispatcher adapts to actual seed/runtime variation after launch.
declare -A ALGO_WEIGHTS=(
    [genpo]=600
    [fpopp]=450
    [dppo]=400
    [policyflow]=350
    [fpo]=300
    [ppo]=200
)

declare -a JOBS=()
declare -a CMD=()
declare -A ACTIVE_GPU=()
declare -A ACTIVE_JOB=()
declare -A ACTIVE_ALGO=()
declare -A ACTIVE_BY_ALGO=()
declare -A ALGO_LIMIT=()
declare -a JOB_STATE=()
declare -a FREE_GPU=()
FAILED_RUNS=0
STOPPING=0
SKIPPED_DONE=0
SWEEP_CONFIG_TEXT=""
TOTAL_CONFIGURED=$(( ${#TASKS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))
CHECKPOINT_PRIORITY_BOOST=100000

mark_ready() {
    local state="${1:-running}"
    local ready_tmp
    [[ -n "${READY_FILE}" ]] || return 0
    mkdir -p "$(dirname "${READY_FILE}")"
    ready_tmp="${READY_FILE}.tmp.$$"
    {
        echo "state=${state}"
        echo "ready_at=$(date --iso-8601=seconds)"
        echo "scheduler_pid=$$"
        echo "active_jobs=${#ACTIVE_GPU[@]}"
        echo "pending_runs=${#JOBS[@]}"
        echo "log_root=${LOG_ROOT}"
    } >"${ready_tmp}"
    mv -f "${ready_tmp}" "${READY_FILE}"
    echo "[$(date '+%F %T')] LAUNCHER_READY state=${state} active_jobs=${#ACTIVE_GPU[@]} file=${READY_FILE}"
}

sanitize() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
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

run_name_for() {
    local task="$1"
    local algo="$2"
    local seed="$3"
    printf '%s-%s-seed%s-%s\n' "${task}" "${algo}" "${seed}" "${RUN_NAME_SUFFIX}"
}

job_log_root() {
    local task="$1"
    if (( ${#TASKS[@]} == 1 )); then
        printf '%s\n' "${LOG_ROOT}"
    else
        printf '%s/%s\n' "${LOG_ROOT}" "$(sanitize "${task}")"
    fi
}

done_file_for() {
    local task="$1"
    local run_name="$2"
    local safe_name
    safe_name="$(sanitize "${run_name}")"
    printf '%s/%s.done\n' "$(job_log_root "${task}")" "${safe_name}"
}

has_resumable_checkpoint() {
    local task="$1"
    local algo="$2"
    local run_name="$3"
    local marker checkpoint_root
    checkpoint_root="${LOG_DIR}/${algo}/${LOG_TAG}/${task}/${run_name}/ckpt"
    for marker in "${checkpoint_root}"/[0-9]*/_SUCCESS; do
        [[ -f "${marker}" ]] && return 0
    done
    return 1
}

validate_inputs() {
    local gpu algo seed task flag value limit_spec limit_algo limit_value
    local -A seen_gpus=()
    local -A seen_tasks=()

    if (( ${#GPUS[@]} == 0 || ${#TASKS[@]} == 0 )); then
        echo "At least one GPU and one task are required." >&2
        return 1
    fi
    for gpu in "${GPUS[@]}"; do
        if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "GPU ids must be non-negative integers, got: ${gpu}" >&2
            return 1
        fi
        if [[ -v "seen_gpus[${gpu}]" ]]; then
            echo "Duplicate physical GPU id: ${gpu}" >&2
            return 1
        fi
        seen_gpus[${gpu}]=1
    done
    for seed in "${SEEDS[@]}"; do
        if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
            echo "Seeds must be non-negative integers, got: ${seed}" >&2
            return 1
        fi
    done
    for flag in WANDB_ENABLED RESUME_LOCAL SAVE_CKPT AUTO_RESUME CHECK_GPU_MEM DRY_RUN; do
        value="${!flag}"
        if [[ "${value}" != "0" && "${value}" != "1" ]]; then
            echo "${flag} must be 0 or 1, got: ${value}" >&2
            return 1
        fi
    done
    if ! [[ "${JOB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "JOB_TIMEOUT_SECONDS must be a non-negative integer." >&2
        return 1
    fi
    if ! [[ "${INTERRUPT_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "INTERRUPT_GRACE_SECONDS must be a non-negative integer." >&2
        return 1
    fi
    if ! [[ "${MIN_FREE_MEM_MB}" =~ ^[0-9]+$ ]]; then
        echo "MIN_FREE_MEM_MB must be a non-negative integer." >&2
        return 1
    fi
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        return 1
    fi
    if (( JOB_TIMEOUT_SECONDS > 0 )) && ! command -v timeout >/dev/null 2>&1; then
        echo "timeout is required when JOB_TIMEOUT_SECONDS is nonzero." >&2
        return 1
    fi
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
    for algo in "${ALGOS[@]}"; do
        if [[ ! -v "ALGO_WEIGHTS[${algo}]" ]]; then
            echo "No scheduling weight configured for algorithm: ${algo}" >&2
            return 1
        fi
        if [[ ! -f "examples/online/config/isaaclab_onpolicy/algo/${algo}.yaml" ]]; then
            echo "Missing algorithm config: examples/online/config/isaaclab_onpolicy/algo/${algo}.yaml" >&2
            return 1
        fi
    done

    ALGO_LIMIT=()
    for limit_spec in ${ALGO_MAX_CONCURRENT}; do
        if [[ "${limit_spec}" != *=* ]]; then
            echo "Invalid ALGO_MAX_CONCURRENT entry: ${limit_spec}; expected algo=count." >&2
            return 1
        fi
        limit_algo="${limit_spec%%=*}"
        limit_value="${limit_spec#*=}"
        if [[ ! "${limit_value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Concurrency limit must be positive: ${limit_spec}" >&2
            return 1
        fi
        if [[ ! -v "ALGO_WEIGHTS[${limit_algo}]" ]]; then
            echo "Concurrency limit references unknown algorithm: ${limit_algo}" >&2
            return 1
        fi
        ALGO_LIMIT[${limit_algo}]="${limit_value}"
    done
}

build_sweep_config() {
    SWEEP_CONFIG_TEXT="$(
        printf 'sweep_id=%s\n' "${SWEEP_ID}"
        printf 'tasks=%s\n' "${TASKS[*]}"
        printf 'algos=%s\n' "${ALGOS[*]}"
        printf 'seeds=%s\n' "${SEEDS[*]}"
        printf 'train_frames=%s\n' "${TRAIN_FRAMES}"
        printf 'eval_frames=%s\n' "${EVAL_FRAMES}"
        printf 'log_dir=%s\n' "${LOG_DIR}"
        printf 'log_tag=%s\n' "${LOG_TAG}"
        printf 'log_group=%s\n' "${LOG_GROUP}"
        printf 'run_name_suffix=%s\n' "${RUN_NAME_SUFFIX}"
        printf 'wandb=%s/%s enabled=%s mode=%s\n' "${WANDB_ENTITY}" "${WANDB_PROJECT_NAME}" "${WANDB_ENABLED}" "${WANDB_MODE}"
        printf 'genpo_extra_args=%s\n' "${GENPO_EXTRA_ARGS}"
        printf 'extra_args=%s\n' "${EXTRA_ARGS}"
        printf 'algo_max_concurrent=%s\n' "${ALGO_MAX_CONCURRENT}"
    )"
}

validate_existing_sweep_config() {
    local config_file="${LOG_ROOT}/sweep-config.txt"
    if [[ -f "${config_file}" ]] && ! cmp -s <(printf '%s\n' "${SWEEP_CONFIG_TEXT}") "${config_file}"; then
        echo "Sweep configuration differs from ${config_file}." >&2
        if command -v diff >/dev/null 2>&1; then
            diff -u "${config_file}" <(printf '%s\n' "${SWEEP_CONFIG_TEXT}") >&2 || true
        fi
        echo "Use a new SWEEP_ID or LOG_ROOT for a different experiment." >&2
        return 1
    fi
}

prepare_jobs() {
    local -a records=()
    local task algo seed run_name done_file weight

    SKIPPED_DONE=0
    for task in "${TASKS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_name="$(run_name_for "${task}" "${algo}" "${seed}")"
                done_file="$(done_file_for "${task}" "${run_name}")"
                if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
                    ((SKIPPED_DONE += 1))
                    continue
                fi
                weight="${ALGO_WEIGHTS[${algo}]}"
                if has_resumable_checkpoint "${task}" "${algo}" "${run_name}"; then
                    weight=$((weight + CHECKPOINT_PRIORITY_BOOST))
                fi
                records+=("${weight}"$'\t'"${task}"$'\t'"${algo}"$'\t'"${seed}")
            done
        done
    done

    if (( ${#records[@]} == 0 )); then
        JOBS=()
        return 0
    fi
    mapfile -t JOBS < <(
        printf '%s\n' "${records[@]}" |
            LC_ALL=C sort -s -t $'\t' -k1,1nr
    )
}

algo_extra_args() {
    local algo="$1"
    if [[ "${algo}" == "genpo" ]]; then
        printf '%s %s\n' "${GENPO_EXTRA_ARGS}" "${EXTRA_ARGS}"
    else
        printf '%s\n' "${EXTRA_ARGS}"
    fi
}

build_command() {
    local task="$1"
    local algo="$2"
    local seed="$3"
    local run_name="$4"
    local merged_extra_args

    CMD=(
        "${PYTHON_BIN}" examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=${algo}"
        "seed=${seed}"
        "device=0"
        "train_frames=${TRAIN_FRAMES}"
        "eval_frames=${EVAL_FRAMES}"
        "log.tag=${LOG_TAG}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tags=[${task},${algo},isaaclab,onpolicy,baseline,action-smoothness,4seed,${#GPUS[@]}gpu]"
        "log.wandb_mode=${WANDB_MODE}"
        "log.save_ckpt=${SAVE_CKPT}"
        "log.resume=${AUTO_RESUME}"
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

    merged_extra_args="$(algo_extra_args "${algo}")"
    if [[ -n "${merged_extra_args//[[:space:]]/}" ]]; then
        # shellcheck disable=SC2206
        local extra_args_array=( ${merged_extra_args} )
        CMD+=("${extra_args_array[@]}")
    fi
}

print_plan() {
    echo "Repo: ${REPO_ROOT}"
    echo "Entry: examples/online/main_isaaclab_onpolicy.py"
    echo "Tasks (${#TASKS[@]}): ${TASKS[*]}"
    echo "Algorithms: ${ALGOS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Schedule: global cross-task queue + dynamic GPU refill"
    echo "Algorithm concurrency limits: ${ALGO_MAX_CONCURRENT:-none}"
    echo "Priority: resumable checkpoints first, then slowest algorithms"
    echo "Configured runs: ${TOTAL_CONFIGURED}"
    echo "Skipped local done: ${SKIPPED_DONE}"
    echo "Pending runs: ${#JOBS[@]}"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "Eval frames: ${EVAL_FRAMES}"
    echo "Checkpoint/resume: save=${SAVE_CKPT} auto_resume=${AUTO_RESUME}"
    echo "Interrupt grace: ${INTERRUPT_GRACE_SECONDS}s"
    echo "GenPO extra args: ${GENPO_EXTRA_ARGS}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} enabled=${WANDB_ENABLED} mode=${WANDB_MODE}"
    echo "W&B group: ${LOG_GROUP}"
    echo "Logs: ${LOG_ROOT}"
}

print_initial_dispatch() {
    local gpu_idx job_idx weight task algo seed limit active
    local -A selected=()
    local -A simulated_active=()

    echo "Initial cap-aware dispatch:"
    for gpu_idx in "${!GPUS[@]}"; do
        for job_idx in "${!JOBS[@]}"; do
            [[ -v "selected[${job_idx}]" ]] && continue
            IFS=$'\t' read -r weight task algo seed <<<"${JOBS[${job_idx}]}"
            limit="${ALGO_LIMIT[${algo}]:-0}"
            active="${simulated_active[${algo}]:-0}"
            if (( limit != 0 && active >= limit )); then
                continue
            fi
            selected[${job_idx}]=1
            simulated_active[${algo}]=$((active + 1))
            echo "  gpu=${GPUS[${gpu_idx}]} task=${task} algo=${algo} seed=${seed}"
            break
        done
    done
}

print_dry_run() {
    local job_idx weight task algo seed run_name
    for job_idx in "${!JOBS[@]}"; do
        IFS=$'\t' read -r weight task algo seed <<<"${JOBS[${job_idx}]}"
        run_name="$(run_name_for "${task}" "${algo}" "${seed}")"
        build_command "${task}" "${algo}" "${seed}" "${run_name}"
        printf 'queue_rank=%q weight=%q gpu=assigned_at_runtime PYTHONPATH=%q WANDB_MODE=%q WANDB_PROJECT=%q WANDB_ENTITY=%q FLOWRL_ISAACLAB_CLOSE_APP=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ' \
            "$((job_idx + 1))" "${weight}" "${PYTHONPATH_VALUE}" "${WANDB_MODE}" "${WANDB_PROJECT_NAME}" "${WANDB_ENTITY}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    done
}

check_gpu_memory() {
    if [[ "${CHECK_GPU_MEM}" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the IsaacLab startup check." >&2
        return 1
    fi

    local gpu line free_mb total_mb
    for gpu in "${GPUS[@]}"; do
        line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
        if [[ -z "${line}" ]]; then
            echo "Could not query GPU ${gpu}." >&2
            return 1
        fi
        IFS=',' read -r free_mb total_mb <<<"${line}"
        free_mb="${free_mb//[[:space:]]/}"
        total_mb="${total_mb//[[:space:]]/}"
        echo "GPU ${gpu}: free=${free_mb} MiB total=${total_mb} MiB"
        if (( free_mb < MIN_FREE_MEM_MB )); then
            echo "GPU ${gpu} has less than MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB} MiB free." >&2
            return 1
        fi
    done
}

validate_finished_log() {
    local log_file="$1"
    local algo="$2"
    grep -Fq "training: 100%" "${log_file}" &&
        grep -Fq "train_frames: ${TRAIN_FRAMES}" "${log_file}" &&
        grep -Fq "name: ${algo}" "${log_file}"
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
    local pid any_alive deadline
    local -a active_pids=("${!ACTIVE_GPU[@]}")
    if (( STOPPING != 0 )); then
        for pid in "${active_pids[@]}"; do
            kill_tree "${pid}" KILL
        done
        exit "${status}"
    fi
    STOPPING=1
    trap - INT TERM
    echo "[$(date '+%F %T')] requesting checkpointed shutdown of active IsaacLab jobs ..." >&2
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
            if kill -0 "${pid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        (( any_alive == 0 )) && break
        sleep 1
    done
    for pid in "${active_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[$(date '+%F %T')] grace period expired; killing pid=${pid}" >&2
            kill_tree "${pid}" KILL
        fi
        wait "${pid}" 2>/dev/null || true
    done
    exit "${status}"
}

trap 'stop_all 130' INT
trap 'stop_all 143' TERM

run_one() {
    local task="$1"
    local algo="$2"
    local seed="$3"
    local gpu="$4"
    local rank="$5"
    local run_name task_log_root
    run_name="$(run_name_for "${task}" "${algo}" "${seed}")"
    task_log_root="$(job_log_root "${task}")"
    local safe_name
    safe_name="$(sanitize "${run_name}")"
    local log_file="${task_log_root}/${run_name}.gpu${gpu}.log"
    local done_file="${task_log_root}/${safe_name}.done"
    local failed_file="${task_log_root}/${safe_name}.failed"

    mkdir -p "${task_log_root}"

    if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
        echo "[$(date '+%F %T')] skip done | rank=${rank}/${#JOBS[@]} gpu=${gpu} algo=${algo} seed=${seed}"
        return 0
    fi
    rm -f "${done_file}" "${failed_file}"
    build_command "${task}" "${algo}" "${seed}" "${run_name}"

    if [[ -f "${log_file}" ]]; then
        local archived_log="${log_file}.attempt-$(date '+%Y%m%d-%H%M%S')"
        mv "${log_file}" "${archived_log}"
        echo "[$(date '+%F %T')] archived previous attempt: ${archived_log}"
    fi

    echo "[$(date '+%F %T')] start | rank=${rank}/${#JOBS[@]} gpu=${gpu} algo=${algo} seed=${seed} log=${log_file}"
    local start_ts status elapsed
    start_ts="$(date +%s)"
    if (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EGL_VISIBLE_DEVICES="${gpu}"
        export PYTHONPATH="${PYTHONPATH_VALUE}"
        export WANDB_MODE="${WANDB_MODE}"
        export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
        export WANDB_ENTITY="${WANDB_ENTITY}"
        export FLOWRL_ISAACLAB_CLOSE_APP=0
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONUNBUFFERED=1
        if (( JOB_TIMEOUT_SECONDS > 0 )); then
            timeout --preserve-status --signal=TERM --kill-after=120s "${JOB_TIMEOUT_SECONDS}" "${CMD[@]}"
        else
            "${CMD[@]}"
        fi
    ) >"${log_file}" 2>&1; then
        status=0
    else
        status=$?
    fi
    elapsed=$(( $(date +%s) - start_ts ))

    if (( status == 0 )) && ! validate_finished_log "${log_file}" "${algo}"; then
        status=90
    fi
    if (( status == 0 )); then
        date --iso-8601=seconds >"${done_file}"
        echo "[$(date '+%F %T')] done | rank=${rank}/${#JOBS[@]} gpu=${gpu} algo=${algo} seed=${seed} elapsed=${elapsed}s"
        return 0
    fi

    {
        echo "time=$(date --iso-8601=seconds)"
        echo "gpu=${gpu}"
        echo "task=${task}"
        echo "algo=${algo}"
        echo "seed=${seed}"
        echo "exit_code=${status}"
        echo "elapsed_seconds=${elapsed}"
        echo "log_file=${log_file}"
    } >"${failed_file}"
    echo "[$(date '+%F %T')] failed | rank=${rank}/${#JOBS[@]} gpu=${gpu} algo=${algo} seed=${seed} exit=${status} elapsed=${elapsed}s log=${log_file}" >&2
    tail -n 40 "${log_file}" >&2 || true
    return "${status}"
}

launch_job() {
    local gpu_idx="$1"
    local job_idx="$2"
    local weight task algo seed gpu pid
    IFS=$'\t' read -r weight task algo seed <<<"${JOBS[${job_idx}]}"
    gpu="${GPUS[${gpu_idx}]}"
    JOB_STATE[${job_idx}]=1
    FREE_GPU[${gpu_idx}]=0
    ACTIVE_BY_ALGO[${algo}]=$(( ${ACTIVE_BY_ALGO[${algo}]:-0} + 1 ))
    run_one "${task}" "${algo}" "${seed}" "${gpu}" "$((job_idx + 1))" &
    pid=$!
    ACTIVE_GPU[${pid}]="${gpu_idx}"
    ACTIVE_JOB[${pid}]="${job_idx}"
    ACTIVE_ALGO[${pid}]="${algo}"
}

find_next_eligible_job() {
    local job_idx weight task algo seed limit active
    for job_idx in "${!JOBS[@]}"; do
        (( ${JOB_STATE[${job_idx}]:-0} == 0 )) || continue
        IFS=$'\t' read -r weight task algo seed <<<"${JOBS[${job_idx}]}"
        limit="${ALGO_LIMIT[${algo}]:-0}"
        active="${ACTIVE_BY_ALGO[${algo}]:-0}"
        if (( limit == 0 || active < limit )); then
            printf '%s\n' "${job_idx}"
            return 0
        fi
    done
    return 1
}

fill_free_gpus() {
    local gpu_idx job_idx
    for gpu_idx in "${!GPUS[@]}"; do
        (( ${FREE_GPU[${gpu_idx}]:-0} == 1 )) || continue
        if job_idx="$(find_next_eligible_job)"; then
            launch_job "${gpu_idx}" "${job_idx}"
        fi
    done
}

run_scheduler() {
    local job_idx gpu_idx pid process_state rc finished_pid finished_gpu finished_job finished_algo
    FAILED_RUNS=0
    JOB_STATE=()
    FREE_GPU=()
    ACTIVE_GPU=()
    ACTIVE_JOB=()
    ACTIVE_ALGO=()
    ACTIVE_BY_ALGO=()

    for job_idx in "${!JOBS[@]}"; do
        JOB_STATE[${job_idx}]=0
    done
    for gpu_idx in "${!GPUS[@]}"; do
        FREE_GPU[${gpu_idx}]=1
    done
    fill_free_gpus
    mark_ready running

    while (( ${#ACTIVE_GPU[@]} > 0 )); do
        finished_pid=""
        rc=0
        while [[ -z "${finished_pid}" ]]; do
            for pid in "${!ACTIVE_GPU[@]}"; do
                process_state=""
                if [[ -r "/proc/${pid}/stat" ]]; then
                    read -r _ _ process_state _ <"/proc/${pid}/stat" || true
                fi
                if [[ -z "${process_state}" || "${process_state}" == "Z" ]]; then
                    finished_pid="${pid}"
                    if wait "${pid}"; then
                        rc=0
                    else
                        rc=$?
                    fi
                    break
                fi
            done
            [[ -n "${finished_pid}" ]] || sleep 1
        done
        finished_gpu="${ACTIVE_GPU[${finished_pid}]}"
        finished_job="${ACTIVE_JOB[${finished_pid}]}"
        finished_algo="${ACTIVE_ALGO[${finished_pid}]}"
        unset 'ACTIVE_GPU['"${finished_pid}"']'
        unset 'ACTIVE_JOB['"${finished_pid}"']'
        unset 'ACTIVE_ALGO['"${finished_pid}"']'
        FREE_GPU[${finished_gpu}]=1
        JOB_STATE[${finished_job}]=2
        ACTIVE_BY_ALGO[${finished_algo}]=$(( ${ACTIVE_BY_ALGO[${finished_algo}]} - 1 ))
        if (( rc != 0 )); then
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo "[$(date '+%F %T')] scheduler recorded failure | job=$((finished_job + 1)) exit=${rc}" >&2
        fi
        fill_free_gpus
    done
    for job_idx in "${!JOBS[@]}"; do
        if (( ${JOB_STATE[${job_idx}]:-0} == 0 )); then
            echo "Scheduler left pending job $((job_idx + 1)) without an eligible slot." >&2
            return 1
        fi
    done
    (( FAILED_RUNS == 0 ))
}

write_metadata() {
    printf '%s\n' "${SWEEP_CONFIG_TEXT}" >"${LOG_ROOT}/sweep-config.txt"
    {
        echo "created_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "python_executable=${PYTHON_BIN}"
        echo "python_version=$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
        echo "bash_version=${BASH_VERSION}"
        echo "scheduler=global_cross_task_dynamic_proc_poll"
        echo "tasks=${TASKS[*]}"
        echo "algo_max_concurrent=${ALGO_MAX_CONCURRENT}"
        echo "save_ckpt=${SAVE_CKPT}"
        echo "auto_resume=${AUTO_RESUME}"
        echo "interrupt_grace_seconds=${INTERRUPT_GRACE_SECONDS}"
        echo "gpus=${GPUS[*]}"
        echo "pending_runs=${#JOBS[@]}"
        echo "skipped_done=${SKIPPED_DONE}"
    } >"${LOG_ROOT}/manifest.txt"
    git status --short >"${LOG_ROOT}/git-status.txt" 2>/dev/null || true
}

main() {
    resolve_python
    validate_inputs
    build_sweep_config
    validate_existing_sweep_config
    prepare_jobs
    print_plan

    if [[ "${DRY_RUN}" == "1" ]]; then
        print_initial_dispatch
        print_dry_run
        return 0
    fi

    if [[ "${PYTHON_PREFLIGHT_DONE}" != "1" ]]; then
        preflight_python
    fi
    if [[ "${GRAPHICS_PREFLIGHT_DONE}" != "1" ]]; then
        preflight_graphics
    fi
    check_gpu_memory
    mkdir -p "${LOG_ROOT}"
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${LOG_ROOT}/launcher.lock"
        if ! flock -n 9; then
            echo "Another launcher holds ${LOG_ROOT}/launcher.lock." >&2
            return 2
        fi
    fi
    write_metadata

    if (( ${#JOBS[@]} == 0 )); then
        mark_ready complete
        echo "All configured runs already have local .done markers."
        return 0
    fi
    if run_scheduler; then
        echo "[$(date '+%F %T')] all ${#JOBS[@]} globally scheduled runs completed"
        return 0
    fi
    echo "[$(date '+%F %T')] sweep completed with ${FAILED_RUNS} failed runs" >&2
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
