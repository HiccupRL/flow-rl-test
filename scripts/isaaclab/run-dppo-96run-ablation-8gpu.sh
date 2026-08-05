#!/usr/bin/env bash
set -uo pipefail

# DPPO diagnostic sweep:
#   4 representative tasks * 8 one-factor variants * 3 seeds = 96 runs.
#
# The diagnostic reference follows the official scratch DPPO defaults. Every
# other variant changes one high-impact setting relative to that reference.
#
# Usage:
#   bash scripts/isaaclab/run-dppo-96run-ablation-8gpu.sh
#
# Inspect all 96 commands without creating files or starting IsaacLab:
#   DRY_RUN=1 bash scripts/isaaclab/run-dppo-96run-ablation-8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "This scheduler requires Bash >= 4.3." >&2
    exit 2
fi

PYTHON_BIN_REQUESTED="${PYTHON_BIN:-}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a TASKS <<<"${TASKS:-Isaac-Cartpole-v0 Isaac-Ant-v0 Isaac-Humanoid-v0 Isaac-Lift-Cube-Franka-v0}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"

TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
SWEEP_ID="${SWEEP_ID:-isaaclab-dppo-96run-ablation-200m-3seed-8gpu-v1}"
LOG_ROOT="${LOG_ROOT:-run_logs/${SWEEP_ID}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-${SWEEP_ID}}"
LOG_GROUP="${LOG_GROUP:-${SWEEP_ID}}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-dppo-ablation-200m-3seed-8gpu-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-dppo-ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
SAVE_CKPT="${SAVE_CKPT:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_LOCAL="${RESUME_LOCAL:-1}"
DISABLE_BOOTSTRAP="${DISABLE_BOOTSTRAP:-0}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-40000}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

declare -a VARIANTS=(
    ref
    legacy_zero_eval
    uniform_clip
    lr_3e5
    lr_3e4
    std_005
    std_020
    steps_05
)

# All variants inherit the corrected reference values and change one field.
declare -A VARIANT_ARGS=(
    [ref]=""
    [legacy_zero_eval]="algo.diffusion.eval_zero_xT=true"
    [uniform_clip]="algo.clip_epsilon=0.2 algo.clip_epsilon_base=0.2"
    [lr_3e5]="algo.actor_lr=0.00003"
    [lr_3e4]="algo.actor_lr=0.0003"
    [std_005]="algo.diffusion.min_logprob_denoising_std=0.05"
    [std_020]="algo.diffusion.min_logprob_denoising_std=0.2"
    [steps_05]="algo.diffusion.steps=5"
)

declare -A VARIANT_DESCRIPTION=(
    [ref]="official scratch reference: uniform clip=0.1, lr=1e-4, std=0.1, K=10"
    [legacy_zero_eval]="zero-prior evaluation ablation: DDPM initialized from xT=0"
    [uniform_clip]="uniform epsilon=0.2 at every denoising step"
    [lr_3e5]="lower actor learning rate: 3e-5"
    [lr_3e4]="higher actor learning rate: 3e-4"
    [std_005]="lower minimum denoising log-prob std: 0.05"
    [std_020]="higher minimum denoising log-prob std: 0.20"
    [steps_05]="shorter denoising chain: K=5"
)

declare -A TASK_WEIGHT=(
    [Isaac-Humanoid-v0]=400
    [Isaac-Lift-Cube-Franka-v0]=300
    [Isaac-Ant-v0]=200
    [Isaac-Cartpole-v0]=100
)

declare -a JOBS=()
declare -a JOB_STATE=()
declare -a FREE_GPU=()
declare -a CMD=()
declare -A ACTIVE_GPU=()
declare -A ACTIVE_JOB=()

FAILED_RUNS=0
SKIPPED_DONE=0
STOPPING=0
TOTAL_CONFIGURED=$(( ${#TASKS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]} ))

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

preflight_python() {
    local summary
    if ! summary="$(PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" -c '
import sys
if sys.version_info < (3, 11):
    raise RuntimeError(f"Python >=3.11 is required, got {sys.version}")
import hydra
import jax
import jaxlib
import examples.online.main_isaaclab_onpolicy
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
    if ! command -v vulkaninfo >/dev/null 2>&1; then
        echo "Warning: vulkaninfo unavailable; skipping graphics preflight." >&2
        return 0
    fi
    summary="$(vulkaninfo --summary 2>&1 || true)"
    if ! grep -Eiq 'vendorID[[:space:]]*= 0x10de|deviceName[[:space:]]*= NVIDIA' <<<"${summary}"; then
        echo "Isaac Sim graphics preflight failed: Vulkan cannot see an NVIDIA GPU." >&2
        return 1
    fi
}

validate_inputs() {
    local gpu task seed variant flag value
    local -A seen_gpu=()
    local -A seen_task=()
    local -A seen_seed=()

    if (( ${#GPUS[@]} != 8 )); then
        echo "This sweep is designed for exactly 8 GPUs; got: ${GPUS[*]}" >&2
        return 1
    fi
    if (( ${#TASKS[@]} != 4 || ${#SEEDS[@]} != 3 || ${#VARIANTS[@]} != 8 )); then
        echo "The fixed 96-run design requires 4 tasks, 8 variants, and 3 seeds." >&2
        return 1
    fi
    if (( TOTAL_CONFIGURED != 96 )); then
        echo "Internal matrix error: expected 96 runs, got ${TOTAL_CONFIGURED}." >&2
        return 1
    fi
    for gpu in "${GPUS[@]}"; do
        if ! [[ "${gpu}" =~ ^[0-9]+$ ]] || [[ -v "seen_gpu[${gpu}]" ]]; then
            echo "GPU ids must be unique non-negative integers: ${GPUS[*]}" >&2
            return 1
        fi
        seen_gpu[${gpu}]=1
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
        if [[ ! -v "TASK_WEIGHT[${task}]" ]]; then
            echo "Missing scheduling weight for task: ${task}" >&2
            return 1
        fi
    done
    for seed in "${SEEDS[@]}"; do
        if ! [[ "${seed}" =~ ^[0-9]+$ ]] || [[ -v "seen_seed[${seed}]" ]]; then
            echo "Seeds must be unique non-negative integers: ${SEEDS[*]}" >&2
            return 1
        fi
        seen_seed[${seed}]=1
    done
    for variant in "${VARIANTS[@]}"; do
        if [[ ! -v "VARIANT_ARGS[${variant}]" || ! -v "VARIANT_DESCRIPTION[${variant}]" ]]; then
            echo "Incomplete variant definition: ${variant}" >&2
            return 1
        fi
    done
    for flag in WANDB_ENABLED SAVE_CKPT AUTO_RESUME RESUME_LOCAL DISABLE_BOOTSTRAP CHECK_GPU_MEM DRY_RUN; do
        value="${!flag}"
        if ! is_bool "${value}"; then
            echo "${flag} must be 0 or 1, got ${value}." >&2
            return 1
        fi
    done
    if ! [[ "${TRAIN_FRAMES}" =~ ^[1-9][0-9]*$ && "${EVAL_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_FRAMES and EVAL_FRAMES must be positive integers." >&2
        return 1
    fi
    if ! [[ "${MIN_FREE_MEM_MB}" =~ ^[0-9]+$ && "${JOB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "MIN_FREE_MEM_MB and JOB_TIMEOUT_SECONDS must be non-negative integers." >&2
        return 1
    fi
}

run_name_for() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    printf '%s-dppo-%s-seed%s-%s\n' "${task}" "${variant}" "${seed}" "${RUN_NAME_SUFFIX}"
}

task_log_root() {
    printf '%s/%s\n' "${LOG_ROOT}" "$(sanitize "$1")"
}

done_file_for() {
    local task="$1"
    local run_name="$2"
    printf '%s/%s.done\n' "$(task_log_root "${task}")" "$(sanitize "${run_name}")"
}

has_resumable_checkpoint() {
    local task="$1"
    local run_name="$2"
    local marker
    local checkpoint_root="${LOG_DIR}/dppo/${LOG_TAG}/${task}/${run_name}/ckpt"
    for marker in "${checkpoint_root}"/[0-9]*/_SUCCESS; do
        [[ -f "${marker}" ]] && return 0
    done
    return 1
}

prepare_jobs() {
    local task variant seed run_name done_file weight
    local -a records=()
    SKIPPED_DONE=0
    for task in "${TASKS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_name="$(run_name_for "${task}" "${variant}" "${seed}")"
                done_file="$(done_file_for "${task}" "${run_name}")"
                if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
                    ((SKIPPED_DONE += 1))
                    continue
                fi
                weight="${TASK_WEIGHT[${task}]}"
                [[ "${variant}" == "steps_05" ]] && weight=$((weight - 20))
                if has_resumable_checkpoint "${task}" "${run_name}"; then
                    weight=$((weight + 10000))
                fi
                records+=("${weight}"$'\t'"${task}"$'\t'"${variant}"$'\t'"${seed}")
            done
        done
    done
    if (( ${#records[@]} == 0 )); then
        JOBS=()
    else
        mapfile -t JOBS < <(
            printf '%s\n' "${records[@]}" |
                LC_ALL=C sort -s -t $'\t' -k1,1nr
        )
    fi
}

build_command() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    local run_name="$4"
    local bootstrap_value=false
    local -a reference_args=(
        "algo.actor_lr=0.0001"
        "algo.clip_epsilon=0.1"
        "algo.clip_epsilon_base=0.1"
        "algo.clip_epsilon_rate=3.0"
        "algo.gamma_denoising=1.0"
        "algo.diffusion.min_sampling_denoising_std=0.1"
        "algo.diffusion.min_logprob_denoising_std=0.1"
        "algo.diffusion.randn_clip_value=3.0"
        "algo.diffusion.steps=10"
        "algo.diffusion.eval_zero_xT=false"
    )
    local -a variant_args=()
    local -a extra_args=()

    [[ "${DISABLE_BOOTSTRAP}" == "1" ]] && bootstrap_value=true
    [[ -n "${VARIANT_ARGS[${variant}]}" ]] &&
        read -r -a variant_args <<<"${VARIANT_ARGS[${variant}]}"
    [[ -n "${EXTRA_ARGS//[[:space:]]/}" ]] &&
        read -r -a extra_args <<<"${EXTRA_ARGS}"

    CMD=(
        "${PYTHON_BIN}" examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=dppo"
        "seed=${seed}"
        "device=0"
        "train_frames=${TRAIN_FRAMES}"
        "eval_frames=${EVAL_FRAMES}"
        "disable_bootstrap=${bootstrap_value}"
        "log.tag=${LOG_TAG}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tags=[${task},dppo,dppo-ablation,${variant},3seed,8gpu]"
        "log.wandb_mode=${WANDB_MODE}"
        "log.save_ckpt=${SAVE_CKPT}"
        "log.resume=${AUTO_RESUME}"
        "${reference_args[@]}"
        "${variant_args[@]}"
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
    local variant
    echo "DPPO 96-run diagnostic sweep"
    echo "Tasks (${#TASKS[@]}): ${TASKS[*]}"
    echo "Seeds (${#SEEDS[@]}): ${SEEDS[*]}"
    echo "GPUs (${#GPUS[@]}): ${GPUS[*]}"
    echo "Configured/pending/skipped: ${TOTAL_CONFIGURED}/${#JOBS[@]}/${SKIPPED_DONE}"
    echo "Train/eval frames: ${TRAIN_FRAMES}/${EVAL_FRAMES}"
    echo "Time-limit handling: disable_bootstrap=${DISABLE_BOOTSTRAP}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} mode=${WANDB_MODE}"
    echo "Logs: ${LOG_ROOT}"
    echo "Variants:"
    for variant in "${VARIANTS[@]}"; do
        echo "  ${variant}: ${VARIANT_DESCRIPTION[${variant}]}"
    done
}

print_dry_run() {
    local index weight task variant seed run_name
    for index in "${!JOBS[@]}"; do
        IFS=$'\t' read -r weight task variant seed <<<"${JOBS[${index}]}"
        run_name="$(run_name_for "${task}" "${variant}" "${seed}")"
        build_command "${task}" "${variant}" "${seed}" "${run_name}"
        printf 'queue_rank=%q weight=%q gpu=assigned_at_runtime CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=%q ' \
            "$((index + 1))" "${weight}" "${PYTHONPATH_VALUE}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
    done
}

check_gpu_memory() {
    local gpu line free_mb total_mb
    [[ "${CHECK_GPU_MEM}" == "1" ]] || return 0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the GPU memory preflight." >&2
        return 1
    fi
    for gpu in "${GPUS[@]}"; do
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

validate_finished_log() {
    local log_file="$1"
    grep -Fq "training: 100%" "${log_file}" &&
        grep -Fq "train_frames: ${TRAIN_FRAMES}" "${log_file}" &&
        grep -Fq "name: dppo" "${log_file}"
}

run_one() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    local gpu="$4"
    local rank="$5"
    local run_name task_root safe_name log_file done_file failed_file
    local start_ts status elapsed archived_log

    run_name="$(run_name_for "${task}" "${variant}" "${seed}")"
    task_root="$(task_log_root "${task}")"
    safe_name="$(sanitize "${run_name}")"
    log_file="${task_root}/${run_name}.gpu${gpu}.log"
    done_file="${task_root}/${safe_name}.done"
    failed_file="${task_root}/${safe_name}.failed"
    mkdir -p "${task_root}"

    if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
        echo "[$(date '+%F %T')] skip done | rank=${rank} gpu=${gpu} task=${task} variant=${variant} seed=${seed}"
        return 0
    fi
    if [[ -f "${log_file}" ]]; then
        archived_log="${log_file}.attempt-$(date '+%Y%m%d-%H%M%S')"
        mv "${log_file}" "${archived_log}"
    fi
    build_command "${task}" "${variant}" "${seed}" "${run_name}"
    echo "[$(date '+%F %T')] start | rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} variant=${variant} seed=${seed}"
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
        date --iso-8601=seconds >"${done_file}"
        [[ -f "${failed_file}" ]] && mv "${failed_file}" "${failed_file}.recovered-$(date '+%Y%m%d-%H%M%S')"
        echo "[$(date '+%F %T')] done | rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} variant=${variant} seed=${seed} elapsed=${elapsed}s"
        return 0
    fi

    {
        echo "time=$(date --iso-8601=seconds)"
        echo "gpu=${gpu}"
        echo "task=${task}"
        echo "variant=${variant}"
        echo "seed=${seed}"
        echo "exit_code=${status}"
        echo "elapsed_seconds=${elapsed}"
        echo "log_file=${log_file}"
    } >"${failed_file}"
    echo "[$(date '+%F %T')] failed | rank=${rank}/${#JOBS[@]} gpu=${gpu} task=${task} variant=${variant} seed=${seed} exit=${status}" >&2
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

stop_all() {
    local status="$1"
    local pid deadline
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
        kill_tree "${pid}" TERM
    done
    deadline=$((SECONDS + INTERRUPT_GRACE_SECONDS))
    while (( SECONDS < deadline )); do
        local any_alive=0
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
    local weight task variant seed gpu pid
    IFS=$'\t' read -r weight task variant seed <<<"${JOBS[${job_index}]}"
    gpu="${GPUS[${gpu_index}]}"
    JOB_STATE[${job_index}]=1
    FREE_GPU[${gpu_index}]=0
    run_one "${task}" "${variant}" "${seed}" "${gpu}" "$((job_index + 1))" &
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
                    if wait "${pid}"; then rc=0; else rc=$?; fi
                    break
                fi
            done
            [[ -n "${finished_pid}" ]] || sleep 1
        done
        finished_gpu="${ACTIVE_GPU[${finished_pid}]}"
        finished_job="${ACTIVE_JOB[${finished_pid}]}"
        unset 'ACTIVE_GPU['"${finished_pid}"']'
        unset 'ACTIVE_JOB['"${finished_pid}"']'
        FREE_GPU[${finished_gpu}]=1
        JOB_STATE[${finished_job}]=2
        if (( rc != 0 )); then
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
        fill_free_gpus
    done
    (( FAILED_RUNS == 0 ))
}

write_metadata() {
    local variant
    if [[ -f "${LOG_ROOT}/manifest.txt" ]]; then
        return 0
    fi
    mkdir -p "${LOG_ROOT}/source_snapshot"
    {
        echo "sweep_id=${SWEEP_ID}"
        echo "tasks=${TASKS[*]}"
        echo "seeds=${SEEDS[*]}"
        echo "gpus=${GPUS[*]}"
        echo "train_frames=${TRAIN_FRAMES}"
        echo "eval_frames=${EVAL_FRAMES}"
        echo "disable_bootstrap=${DISABLE_BOOTSTRAP}"
        echo "reference=actor_lr=0.0001 clip=0.1 clip_base=0.1 clip_rate=3.0 gamma_denoising=1.0 min_sampling_std=0.1 min_logprob_std=0.1 randn_clip=3.0 steps=10 eval_zero_xT=false"
        for variant in "${VARIANTS[@]}"; do
            echo "variant.${variant}.args=${VARIANT_ARGS[${variant}]}"
            echo "variant.${variant}.description=${VARIANT_DESCRIPTION[${variant}]}"
        done
    } >"${LOG_ROOT}/sweep-config.txt"
    {
        echo "created_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "python_executable=${PYTHON_BIN}"
        echo "python_version=$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
        echo "configured_runs=${TOTAL_CONFIGURED}"
        echo "pending_runs=${#JOBS[@]}"
        echo "skipped_done=${SKIPPED_DONE}"
    } >"${LOG_ROOT}/manifest.txt"
    git status --short >"${LOG_ROOT}/git-status.txt" 2>/dev/null || true
    git diff -- \
        flowrl/agent/online/dppo.py \
        flowrl/config/online/algo/dppo.py \
        examples/online/config/isaaclab_onpolicy/algo/dppo.yaml \
        >"${LOG_ROOT}/source_snapshot/dppo.patch" 2>/dev/null || true
    cp flowrl/agent/online/dppo.py \
        "${LOG_ROOT}/source_snapshot/dppo_agent.py"
    cp flowrl/config/online/algo/dppo.py \
        "${LOG_ROOT}/source_snapshot/dppo_config.py"
    cp examples/online/config/isaaclab_onpolicy/algo/dppo.yaml \
        "${LOG_ROOT}/source_snapshot/dppo.yaml"
    cp "${BASH_SOURCE[0]}" \
        "${LOG_ROOT}/source_snapshot/run-dppo-96run-ablation-8gpu.sh"
}

verify_resume_source() {
    local snapshot_root="${LOG_ROOT}/source_snapshot"
    local -a source_pairs=(
        "flowrl/agent/online/dppo.py:${snapshot_root}/dppo_agent.py"
        "flowrl/config/online/algo/dppo.py:${snapshot_root}/dppo_config.py"
        "examples/online/config/isaaclab_onpolicy/algo/dppo.yaml:${snapshot_root}/dppo.yaml"
    )
    local pair current snapshot

    [[ -f "${LOG_ROOT}/manifest.txt" ]] || return 0
    for pair in "${source_pairs[@]}"; do
        current="${pair%%:*}"
        snapshot="${pair#*:}"
        if [[ ! -f "${snapshot}" ]] || ! cmp -s "${current}" "${snapshot}"; then
            echo "Refusing to mix DPPO implementations in ${LOG_ROOT}." >&2
            echo "Source drift detected: ${current}" >&2
            echo "Use a new SWEEP_ID/LOG_ROOT for the corrected baseline." >&2
            return 1
        fi
    done
}

main() {
    resolve_python || return 1
    validate_inputs || return 1
    prepare_jobs
    print_plan
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_dry_run
        return 0
    fi
    preflight_python || return 1
    preflight_graphics || return 1
    check_gpu_memory || return 1
    mkdir -p "${LOG_ROOT}"
    verify_resume_source || return 1
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${LOG_ROOT}/launcher.lock"
        if ! flock -n 9; then
            echo "Another launcher holds ${LOG_ROOT}/launcher.lock." >&2
            return 2
        fi
    fi
    write_metadata
    if (( ${#JOBS[@]} == 0 )); then
        echo "All 96 runs already have local .done markers."
        return 0
    fi
    if run_scheduler; then
        echo "[$(date '+%F %T')] all ${#JOBS[@]} pending DPPO runs completed"
        return 0
    fi
    echo "[$(date '+%F %T')] sweep completed with ${FAILED_RUNS} failed runs" >&2
    return 1
}

main "$@"
