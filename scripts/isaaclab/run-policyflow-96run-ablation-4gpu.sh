#!/usr/bin/env bash
set -uo pipefail

# PolicyFlow method-only diagnostic sweep:
#   4 shared IsaacLab tasks * 8 one-factor variants * 3 seeds = 96 runs.
#
# Fair-comparison protocol (identical for every run and aligned with the other
# IsaacLab baselines):
#   1024 envs, rollout 24, batch 6144, 4 minibatches, 4 epochs,
#   normalized observations, and 200M training frames.
#
# Only PolicyFlow-specific choices vary across variants. Task YAMLs continue to
# own environment settings such as action bounds and timeout handling.
#
# Usage:
#   bash scripts/isaaclab/run-policyflow-96run-ablation-4gpu.sh
#
# Print all commands without creating logs or starting IsaacLab:
#   DRY_RUN=1 bash scripts/isaaclab/run-policyflow-96run-ablation-4gpu.sh
#
# Use another four physical GPUs:
#   GPUS="4 5 6 7" bash scripts/isaaclab/run-policyflow-96run-ablation-4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    echo "This scheduler requires Bash >= 4.3." >&2
    exit 2
fi

PYTHON_BIN_REQUESTED="${PYTHON_BIN:-}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3}"
read -r -a TASKS <<<"${TASKS:-Isaac-Cartpole-v0 Isaac-Ant-v0 Isaac-Humanoid-v0 Isaac-Lift-Cube-Franka-v0}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"

# These are intentionally constants: variants must not change the shared
# environment/training protocol.
readonly NUM_ENVS=1024
readonly ROLLOUT_LENGTH=24
readonly NUM_MINIBATCHES=4
readonly NUM_EPOCHS=4
readonly BATCH_SIZE=6144
readonly TRAIN_FRAMES=200000000
readonly EVAL_FRAMES=5000000

SWEEP_ID="${SWEEP_ID:-isaaclab-policyflow-96run-method-only-200m-3seed-4gpu-v1}"
LOG_ROOT="${LOG_ROOT:-run_logs/${SWEEP_ID}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-${SWEEP_ID}}"
LOG_GROUP="${LOG_GROUP:-${SWEEP_ID}}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-policyflow-method-only-200m-3seed-4gpu-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-policyflow-method-ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
SAVE_CKPT="${SAVE_CKPT:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_LOCAL="${RESUME_LOCAL:-1}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
INTERRUPT_GRACE_SECONDS="${INTERRUPT_GRACE_SECONDS:-180}"
DRY_RUN="${DRY_RUN:-0}"
PYTHONPATH_VALUE="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

declare -a VARIANTS=(
    official_ref
    legacy_arch
    paper_zero_output
    paper_steps_12
    fixed_lr
    no_brownian
    paper_brownian_l2
    ppo_value_clip
)

declare -A VARIANT_ARGS=(
    [official_ref]=""
    [legacy_arch]="algo.flow.architecture=legacy"
    [paper_zero_output]="algo.flow.zero_init_output=true"
    [paper_steps_12]="algo.flow.steps=12"
    [fixed_lr]="algo.adaptive_learning_rate=false"
    [no_brownian]="algo.brownian_reg_loss_scale=0.0"
    [paper_brownian_l2]="algo.brownian_reduction=paper_l2"
    [ppo_value_clip]="algo.value_clip_mode=ppo"
)

declare -A VARIANT_DESCRIPTION=(
    [official_ref]="released-code method reference under the shared baseline protocol"
    [legacy_arch]="legacy raw-observation concatenation, learnable Fourier features, and nonzero actor output"
    [paper_zero_output]="exact zero actor output initialization stated in the paper"
    [paper_steps_12]="12 RK2 solver steps stated in the paper instead of the released code's 10"
    [fixed_lr]="fixed learning rate instead of desired-KL adaptive learning rate"
    [no_brownian]="Brownian coefficient set to zero"
    [paper_brownian_l2]="paper squared-L2 Brownian reduction instead of released-code elementwise mean"
    [ppo_value_clip]="standard PPO max(unclipped, clipped) value loss instead of direct value clipping"
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

validate_inputs() {
    local gpu task seed variant flag value
    local -A seen_gpu=()
    local -A seen_task=()
    local -A seen_seed=()

    if (( ${#GPUS[@]} != 4 )); then
        echo "This sweep is designed for exactly 4 GPUs; got: ${GPUS[*]}" >&2
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
    for flag in WANDB_ENABLED SAVE_CKPT AUTO_RESUME RESUME_LOCAL CHECK_GPU_MEM DRY_RUN; do
        value="${!flag}"
        if ! is_bool "${value}"; then
            echo "${flag} must be 0 or 1, got ${value}." >&2
            return 1
        fi
    done
    if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ &&
            "${ROLLOUT_LENGTH}" =~ ^[1-9][0-9]*$ &&
            "${NUM_MINIBATCHES}" =~ ^[1-9][0-9]*$ &&
            "${NUM_EPOCHS}" =~ ^[1-9][0-9]*$ &&
            "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ &&
            "${TRAIN_FRAMES}" =~ ^[1-9][0-9]*$ &&
            "${EVAL_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "The shared protocol contains an invalid non-positive integer." >&2
        return 1
    fi
    if (( NUM_ENVS * ROLLOUT_LENGTH != NUM_MINIBATCHES * BATCH_SIZE )); then
        echo "Batch invariant failed: envs*rollout must equal minibatches*batch_size." >&2
        return 1
    fi
    if ! [[ "${MIN_FREE_MEM_MB}" =~ ^[0-9]+$ &&
            "${JOB_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "MIN_FREE_MEM_MB and JOB_TIMEOUT_SECONDS must be non-negative integers." >&2
        return 1
    fi
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
from flowrl.agent.online.policyflow import PolicyFlowAgent
print(f"{sys.executable} | jax={jax.__version__} jaxlib={jaxlib.__version__}")
' 2>&1)"; then
        echo "Python preflight failed for ${PYTHON_BIN}" >&2
        printf '%s\n' "${summary}" >&2
        return 1
    fi
    echo "Python preflight: ${summary}"
}

run_name_for() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    printf '%s-policyflow-%s-seed%s-%s\n' \
        "${task}" "${variant}" "${seed}" "${RUN_NAME_SUFFIX}"
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

prepare_jobs() {
    local variant seed task run_name done_file
    SKIPPED_DONE=0
    for variant in "${VARIANTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for task in "${TASKS[@]}"; do
                run_name="$(run_name_for "${task}" "${variant}" "${seed}")"
                done_file="$(done_file_for "${task}" "${run_name}")"
                if [[ "${RESUME_LOCAL}" == "1" && -f "${done_file}" ]]; then
                    ((SKIPPED_DONE += 1))
                    continue
                fi
                JOBS+=("${task}"$'\t'"${variant}"$'\t'"${seed}")
                JOB_STATE+=(0)
            done
        done
    done
}

build_command() {
    local task="$1"
    local variant="$2"
    local seed="$3"
    local run_name="$4"
    local -a variant_args=()

    [[ -n "${VARIANT_ARGS[${variant}]}" ]] &&
        read -r -a variant_args <<<"${VARIANT_ARGS[${variant}]}"

    CMD=(
        "${PYTHON_BIN}" examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=policyflow"
        "seed=${seed}"
        "device=0"
        "train_frames=${TRAIN_FRAMES}"
        "eval_frames=${EVAL_FRAMES}"
        "norm_obs=true"
        "log.dir=${LOG_DIR}"
        "log.tag=${LOG_TAG}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tags=[${task},policyflow,method-only-ablation,${variant},200m,3seed,4gpu]"
        "log.wandb_mode=${WANDB_MODE}"
        "log.save_ckpt=${SAVE_CKPT}"
        "log.resume=${AUTO_RESUME}"
        "algo.num_envs=${NUM_ENVS}"
        "algo.rollout_length=${ROLLOUT_LENGTH}"
        "algo.num_minibatches=${NUM_MINIBATCHES}"
        "algo.num_epochs=${NUM_EPOCHS}"
        "algo.batch_size=${BATCH_SIZE}"
        "algo.backbone_cls=mlp"
        "algo.critic_hidden_dims=[256,256,256]"
        "algo.flow.hidden_dims=[256,256,256]"
        "algo.critic_activation=elu"
        "algo.flow.activation=elu"
        "algo.critic_lr=0.001"
        "algo.flow.lr=0.0001"
        "algo.gamma=0.99"
        "algo.gae_lambda=0.95"
        "algo.ratio_clip=0.2"
        "algo.reward_scaling=1.0"
        "algo.normalize_advantage=true"
        "algo.gaussian_entropy_loss_scale=0.004"
        "algo.brownian_reg_loss_scale=0.002"
        "algo.brownian_reduction=official_mean"
        "algo.value_loss_scale=1.0"
        "algo.clip_predicted_values=true"
        "algo.value_clip=0.2"
        "algo.value_clip_mode=official"
        "algo.clip_grad_norm=1.0"
        "algo.weight_decay=0.00001"
        "algo.time_limit_bootstrap=true"
        "algo.critic_output_init_scale=0.01"
        "algo.adaptive_learning_rate=true"
        "algo.desired_kl=0.01"
        "algo.learning_rate_min=0.000001"
        "algo.learning_rate_max=0.01"
        "algo.learning_rate_factor=1.5"
        "algo.flow.time_dim=64"
        "algo.flow.steps=10"
        "algo.flow.clip_sampler=false"
        "algo.flow.init_logstd=0.0"
        "algo.flow.logstd_min=-20.0"
        "algo.flow.logstd_max=4.0"
        "algo.flow.interpolation_type=rectified_flow"
        "algo.flow.time_sampling=discrete"
        "algo.flow.architecture=official"
        "algo.flow.zero_init_output=false"
        "algo.flow.output_init_scale=0.01"
        "algo.flow.fourier_scale=16.0"
        "algo.flow.eval_zero_x0=false"
        "algo.flow.eval_sample_delta=true"
        "${variant_args[@]}"
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
    echo "PolicyFlow 96-run method-only diagnostic sweep"
    echo "Tasks (${#TASKS[@]}): ${TASKS[*]}"
    echo "Seeds (${#SEEDS[@]}): ${SEEDS[*]}"
    echo "GPUs (${#GPUS[@]}): ${GPUS[*]}"
    echo "Configured/pending/skipped: ${TOTAL_CONFIGURED}/${#JOBS[@]}/${SKIPPED_DONE}"
    echo "Shared protocol: envs=${NUM_ENVS}, rollout=${ROLLOUT_LENGTH}, batch=${BATCH_SIZE}, minibatches=${NUM_MINIBATCHES}, epochs=${NUM_EPOCHS}, norm_obs=true"
    echo "Training horizon: ${TRAIN_FRAMES} frames; eval every ${EVAL_FRAMES} frames"
    echo "Environment action bounds and timeout handling: inherited from each task YAML"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} mode=${WANDB_MODE}"
    echo "Logs: ${LOG_ROOT}"
    echo "Variants:"
    for variant in "${VARIANTS[@]}"; do
        echo "  ${variant}: ${VARIANT_DESCRIPTION[${variant}]}"
    done
}

print_dry_run() {
    local index task variant seed run_name
    for index in "${!JOBS[@]}"; do
        IFS=$'\t' read -r task variant seed <<<"${JOBS[${index}]}"
        run_name="$(run_name_for "${task}" "${variant}" "${seed}")"
        build_command "${task}" "${variant}" "${seed}" "${run_name}"
        printf 'queue_rank=%q CUDA_VISIBLE_DEVICES=<assigned-at-runtime> PYTHONPATH=%q ' \
            "$((index + 1))" "${PYTHONPATH_VALUE}"
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
    local expected
    grep -Fq "training: 100%" "${log_file}" || return 1
    for expected in \
        "name: policyflow" \
        "train_frames: ${TRAIN_FRAMES}" \
        "norm_obs: true" \
        "num_envs: ${NUM_ENVS}" \
        "rollout_length: ${ROLLOUT_LENGTH}" \
        "num_minibatches: ${NUM_MINIBATCHES}" \
        "num_epochs: ${NUM_EPOCHS}" \
        "batch_size: ${BATCH_SIZE}"; do
        grep -Fq "${expected}" "${log_file}" || return 1
    done
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
        if [[ -f "${failed_file}" ]]; then
            mv "${failed_file}" "${failed_file}.recovered-$(date '+%Y%m%d-%H%M%S')"
        fi
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
        kill_tree "${pid}" TERM
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
    local task variant seed gpu pid
    IFS=$'\t' read -r task variant seed <<<"${JOBS[${job_index}]}"
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

reap_finished_jobs() {
    local pid gpu_index job_index status reaped=0
    local -a pids=("${!ACTIVE_GPU[@]}")
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            continue
        fi
        gpu_index="${ACTIVE_GPU[${pid}]}"
        job_index="${ACTIVE_JOB[${pid}]}"
        if wait "${pid}"; then
            status=0
        else
            status=$?
        fi
        JOB_STATE[${job_index}]=2
        FREE_GPU[${gpu_index}]=1
        unset "ACTIVE_GPU[${pid}]"
        unset "ACTIVE_JOB[${pid}]"
        (( status != 0 )) && ((FAILED_RUNS += 1))
        reaped=1
    done
    return $(( 1 - reaped ))
}

run_scheduler() {
    local gpu_index job_index launched
    for gpu_index in "${!GPUS[@]}"; do
        FREE_GPU[${gpu_index}]=1
    done

    while true; do
        launched=0
        for gpu_index in "${!GPUS[@]}"; do
            if (( ${FREE_GPU[${gpu_index}]:-0} == 1 )); then
                if job_index="$(find_next_job)"; then
                    launch_job "${gpu_index}" "${job_index}"
                    launched=1
                fi
            fi
        done

        if (( ${#ACTIVE_GPU[@]} == 0 )); then
            break
        fi
        if ! reap_finished_jobs; then
            sleep 1
        fi
        (( launched == 1 )) && continue
    done
}

main() {
    resolve_python || exit 2
    validate_inputs || exit 2
    prepare_jobs
    print_plan

    if [[ "${DRY_RUN}" == "1" ]]; then
        print_dry_run
        return 0
    fi
    preflight_python || exit 2
    check_gpu_memory || exit 2
    mkdir -p "${LOG_ROOT}"

    if (( ${#JOBS[@]} == 0 )); then
        echo "All 96 PolicyFlow runs already have local done markers."
        return 0
    fi
    run_scheduler
    echo "PolicyFlow sweep finished: failed=${FAILED_RUNS}, skipped=${SKIPPED_DONE}, total=${TOTAL_CONFIGURED}"
    (( FAILED_RUNS == 0 ))
}

main "$@"
