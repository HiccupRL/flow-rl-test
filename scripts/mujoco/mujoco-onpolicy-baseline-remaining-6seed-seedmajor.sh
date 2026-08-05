#!/usr/bin/env bash
set -euo pipefail

# Continue the MuJoCo on-policy baseline sweep from 4 seeds to 10 seeds.
#
# Default coverage:
#   6 algos * 5 tasks * 6 seeds = 180 runs.
#
# Scheduling:
#   Run one seed at a time. Within a seed, algo/task runs are distributed across
#   the configured GPU worker slots. The next seed starts only after every run
#   for the current seed has completed successfully.
#
# Usage:
#   GPUS="0 1 2 3" RUNS_PER_GPU=3 \
#     bash scripts/mujoco/mujoco-onpolicy-baseline-remaining-6seed-seedmajor.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RAW_GPUS=(${GPUS:-0 1 2 3})
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS_PER_GPU must be a positive integer, got: ${RUNS_PER_GPU}" >&2
    exit 1
fi

GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
    for ((slot = 0; slot < RUNS_PER_GPU; slot += 1)); do
        GPUS+=("${gpu}")
    done
done

ALGOS=(${ALGOS:-ppo dppo fpo fpopp genpo policyflow})
TASKS=(${TASKS:-Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5 Humanoid-v5})
SEEDS=(${SEEDS:-4 5 6 7 8 9})
TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-5000000}"
LOG_FRAMES="${LOG_FRAMES:-100000}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-mujoco-onpolicy-200m-10seed}"
LOG_GROUP="${LOG_GROUP:-mujoco-onpolicy-baseline-200m-4seed}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-log/parallel/mujoco-onpolicy-baseline-200m-seed4-9-seedmajor}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT:-mujoco-onpolicy-baseline-final}}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
ENV_MODE="${ENV_MODE:-async}"
ASYNC_SHARED_MEMORY="${ASYNC_SHARED_MEMORY:-true}"
ASYNC_CONTEXT="${ASYNC_CONTEXT:-}"
DEBUG_FINITE_CHECKS="${DEBUG_FINITE_CHECKS:-false}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

PIDS=()
STOPPING=0
PER_SEED_RUNS=$(( ${#ALGOS[@]} * ${#TASKS[@]} ))
TOTAL_RUNS=$(( PER_SEED_RUNS * ${#SEEDS[@]} ))

sanitize() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

kill_tree() {
    local parent="$1"
    local signal="$2"
    local child

    if command -v pgrep >/dev/null 2>&1; then
        while IFS= read -r child; do
            [[ -n "${child}" ]] && kill_tree "${child}" "${signal}"
        done < <(pgrep -P "${parent}" 2>/dev/null || true)
    fi
    kill "-${signal}" "${parent}" 2>/dev/null || true
}

stop_all() {
    local status="$1"
    if (( STOPPING != 0 )); then
        exit "${status}"
    fi
    STOPPING=1
    trap - INT TERM
    echo "[$(date '+%F %T')] stopping MuJoCo on-policy seed-major workers ..."
    for pid in "${PIDS[@]:-}"; do
        kill_tree "${pid}" TERM
    done
    sleep 3
    for pid in "${PIDS[@]:-}"; do
        kill_tree "${pid}" KILL
    done
    wait 2>/dev/null || true
    exit "${status}"
}

trap 'stop_all 130' INT
trap 'stop_all 143' TERM

run_one() {
    local algo="$1"
    local task="$2"
    local seed="$3"
    local gpu="$4"
    local progress_idx="$5"
    local run_name="${task}-${algo}-seed${seed}"
    local safe_name
    safe_name="$(sanitize "${run_name}")"
    local run_dir="${RUN_LOG_ROOT}/${algo}/${task}"
    local run_log="${run_dir}/seed${seed}.log"
    local done_file="${RUN_LOG_ROOT}/${safe_name}.done"
    local failed_file="${RUN_LOG_ROOT}/${safe_name}.failed"

    mkdir -p "${run_dir}" "${RUN_LOG_ROOT}"

    if [[ -f "${done_file}" ]]; then
        echo "[$(date '+%F %T')] skip done | ${progress_idx}/${TOTAL_RUNS} | algo=${algo} task=${task} seed=${seed}"
        return 0
    fi
    rm -f "${failed_file}"

    local cmd=(
        "${PYTHON_BIN}" examples/online/main_mujoco_onpolicy.py
        "task=${task}"
        "algo=${algo}"
        "seed=${seed}"
        "device=0"
        "train_frames=${TRAIN_FRAMES}"
        "eval_frames=${EVAL_FRAMES}"
        "log_frames=${LOG_FRAMES}"
        "env_mode=${ENV_MODE}"
        "async_shared_memory=${ASYNC_SHARED_MEMORY}"
        "debug_finite_checks=${DEBUG_FINITE_CHECKS}"
        "log.dir=${LOG_DIR}"
        "log.tag=${LOG_TAG}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tags=[${task},${algo},mujoco,onpolicy,baseline,200m]"
        "log.wandb_mode=${WANDB_MODE}"
    )

    if [[ "${WANDB_ENABLED}" == "0" || -z "${WANDB_PROJECT_NAME}" ]]; then
        cmd+=("log.wandb=false")
    else
        cmd+=("log.wandb=true" "log.project=${WANDB_PROJECT_NAME}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
            cmd+=("log.entity=${WANDB_ENTITY}")
        fi
    fi

    if [[ -n "${ASYNC_CONTEXT}" ]]; then
        cmd+=("async_context=${ASYNC_CONTEXT}")
    fi

    if [[ -n "${EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        local extra=( ${EXTRA_ARGS} )
        cmd+=("${extra[@]}")
    fi

    echo "[$(date '+%F %T')] start | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} log=${run_log}"
    if [[ "${DRY_RUN}" != "0" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=%q WANDB_MODE=%q WANDB_PROJECT=%q WANDB_ENTITY=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q XLA_FLAGS=%q ' "${gpu}" "${REPO_ROOT}:${PYTHONPATH:-}" "${WANDB_MODE}" "${WANDB_PROJECT_NAME}" "${WANDB_ENTITY}" "${OMP_NUM_THREADS}" "${MKL_NUM_THREADS}" "${OPENBLAS_NUM_THREADS}" "${XLA_FLAGS}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    local start_ts status elapsed
    start_ts="$(date +%s)"
    if (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
        export WANDB_MODE="${WANDB_MODE}"
        export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
        export WANDB_ENTITY="${WANDB_ENTITY}"
        export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
        export HYDRA_FULL_ERROR XLA_FLAGS
        "${cmd[@]}"
    ) >"${run_log}" 2>&1; then
        status=0
    else
        status=$?
    fi
    elapsed=$(( $(date +%s) - start_ts ))

    if (( status == 0 )); then
        touch "${done_file}"
        echo "[$(date '+%F %T')] done | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} elapsed=${elapsed}s"
        return 0
    fi

    {
        echo "task=${task}"
        echo "algo=${algo}"
        echo "seed=${seed}"
        echo "gpu=${gpu}"
        echo "exit_code=${status}"
        echo "log=${run_log}"
    } >"${failed_file}"
    echo "[$(date '+%F %T')] failed | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} exit=${status} elapsed=${elapsed}s log=${run_log}" >&2
    tail -n 40 "${run_log}" >&2 || true
    return "${status}"
}

worker_seed() {
    local worker_idx="$1"
    local gpu="$2"
    local num_workers="$3"
    local seed="$4"
    local seed_idx="$5"
    local job_idx=0
    local status=0

    for algo in "${ALGOS[@]}"; do
        for task in "${TASKS[@]}"; do
            if (( job_idx % num_workers == worker_idx )); then
                run_one "${algo}" "${task}" "${seed}" "${gpu}" "$((seed_idx * PER_SEED_RUNS + job_idx + 1))" || status=1
            fi
            ((job_idx += 1))
        done
    done
    return "${status}"
}

run_seed() {
    local seed="$1"
    local seed_idx="$2"
    local num_workers="${#GPUS[@]}"
    local status=0

    PIDS=()
    echo "[$(date '+%F %T')] seed start | seed=${seed} | $((seed_idx + 1))/${#SEEDS[@]} | runs=${PER_SEED_RUNS}"

    for i in "${!GPUS[@]}"; do
        worker_seed "${i}" "${GPUS[$i]}" "${num_workers}" "${seed}" "${seed_idx}" &
        PIDS+=("$!")
    done

    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done

    PIDS=()
    if (( status == 0 )); then
        echo "[$(date '+%F %T')] seed done | seed=${seed}"
        return 0
    fi

    echo "[$(date '+%F %T')] seed failed | seed=${seed}; stopping before next seed" >&2
    return 1
}

main() {
    local num_gpus="${#GPUS[@]}"
    local seed_idx=0

    if (( num_gpus == 0 )); then
        echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
        exit 1
    fi

    mkdir -p "${RUN_LOG_ROOT}"
    echo "Repo: ${REPO_ROOT}"
    echo "Entry: examples/online/main_mujoco_onpolicy.py"
    echo "Schedule: seed-major; complete each seed before starting the next"
    echo "Algorithms: ${ALGOS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Physical GPUs: ${RAW_GPUS[*]}"
    echo "Runs per GPU: ${RUNS_PER_GPU}"
    echo "Worker GPU slots: ${GPUS[*]}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} enabled=${WANDB_ENABLED} mode=${WANDB_MODE}"
    echo "W&B group: ${LOG_GROUP}"
    echo "W&B tags: <task>, <algo>, mujoco, onpolicy, baseline, 200m"
    echo "Env mode: ${ENV_MODE}"
    echo "Async shared memory: ${ASYNC_SHARED_MEMORY}"
    echo "Debug finite checks: ${DEBUG_FINITE_CHECKS}"
    echo "Thread caps: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "Eval frames: ${EVAL_FRAMES}"
    echo "Runs per seed: ${PER_SEED_RUNS}"
    echo "Total runs: ${TOTAL_RUNS}"
    echo "Run logs: ${RUN_LOG_ROOT}"

    for seed in "${SEEDS[@]}"; do
        if ! run_seed "${seed}" "${seed_idx}"; then
            echo "[$(date '+%F %T')] MuJoCo seed-major sweep stopped with failures" >&2
            exit 1
        fi
        ((seed_idx += 1))
    done

    echo "[$(date '+%F %T')] all MuJoCo seed-major runs completed"
}

main "$@"
