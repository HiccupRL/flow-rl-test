#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_PER_GPU="${RUNS_PER_GPU:-4}"
ALGOS=(${ALGOS:-ppo dppo fpo fpopp genpo policyflow})
TASKS=(
    ${TASKS:-
        ball_in_cup-catch
        cartpole-balance
        cheetah-run
        finger-spin
        finger-turn_easy
        finger-turn_hard
        fish-swim
        point_mass-easy
        reacher-easy
        reacher-hard
    }
)
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})

FRAME_SKIP="${FRAME_SKIP:-2}"
FRAME_STACK="${FRAME_STACK:-1}"
HORIZON="${HORIZON:-1000}"
TRAIN_FRAMES="${TRAIN_FRAMES:-983040}"
EVAL_FRAMES="${EVAL_FRAMES:-98304}"
LOG_FRAMES="${LOG_FRAMES:-49152}"
ENV_MODE="${ENV_MODE:-sync}"
DEBUG_FINITE_CHECKS="${DEBUG_FINITE_CHECKS:-true}"
SEED_BARRIER="${SEED_BARRIER:-0}"

LOG_DIR="${LOG_DIR:-logs}"
LOG_TAG="${LOG_TAG:-dmc-onpolicy-983040f-10seed-1024env-16epoch}"
LOG_GROUP="${LOG_GROUP:-dmc-onpolicy-baseline-1m-10seed-1024env-16epoch}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-run_logs/dmc-onpolicy-baseline-983040f-10seed-1024env-16epoch-32way-v1}"
BUDGET_TAG="${BUDGET_TAG:-1m}"
SCHEDULE_TAG="${SCHEDULE_TAG:-seed-barrier-${SEED_BARRIER}}"
SEED_PROTOCOL_TAG="${SEED_PROTOCOL_TAG:-independent-seed-blocks-v1}"
POLICYFLOW_TIMEOUT_TAG="${POLICYFLOW_TIMEOUT_TAG:-timeout-bootstrap-official-current-value}"
EVAL_RNG_TAG="${EVAL_RNG_TAG:-isolated-fixed-eval-rng-v1}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT:-dmc-onpolicy-baseline-final}}"
WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"

RAW_GPUS=()
WORKER_GPUS=()
if [[ "${STATUS_ONLY}" == "0" ]]; then
    if [[ -n "${GPUS:-}" ]]; then
        read -r -a RAW_GPUS <<<"${GPUS}"
    else
        mapfile -t RAW_GPUS < <(
            nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        )
    fi

    if ! [[ "${RUNS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RUNS_PER_GPU must be a positive integer, got: ${RUNS_PER_GPU}" >&2
        exit 1
    fi
    if ! [[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_ATTEMPTS must be a positive integer, got: ${MAX_ATTEMPTS}" >&2
        exit 1
    fi
    if ! [[ "${RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "RETRY_DELAY_SECONDS must be a non-negative integer, got: ${RETRY_DELAY_SECONDS}" >&2
        exit 1
    fi
    if [[ "${SEED_BARRIER}" != "0" && "${SEED_BARRIER}" != "1" ]]; then
        echo "SEED_BARRIER must be 0 or 1, got: ${SEED_BARRIER}" >&2
        exit 1
    fi
    if (( ${#RAW_GPUS[@]} == 0 )); then
        echo "No GPUs detected. Set GPUS explicitly, for example GPUS=\"0 1\"." >&2
        exit 1
    fi

    for ((slot = 0; slot < RUNS_PER_GPU; slot += 1)); do
        for gpu in "${RAW_GPUS[@]}"; do
            WORKER_GPUS+=("${gpu}")
        done
    done
fi

TOTAL_RUNS=$(( ${#ALGOS[@]} * ${#TASKS[@]} * ${#SEEDS[@]} ))
PIDS=()
STOPPING=0

sanitize() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

marker_count() {
    local suffix="$1"
    if [[ ! -d "${RUN_LOG_ROOT}" ]]; then
        echo 0
        return
    fi
    find "${RUN_LOG_ROOT}" -maxdepth 1 -type f -name "*.${suffix}" 2>/dev/null |
        wc -l
}

print_status() {
    local done_count failed_count running_count
    done_count="$(marker_count done)"
    failed_count="$(marker_count failed)"
    running_count="$(marker_count running)"
    echo "total=${TOTAL_RUNS} done=${done_count} failed=${failed_count} running=${running_count} remaining=$((TOTAL_RUNS - done_count))"
}

if [[ "${STATUS_ONLY}" != "0" ]]; then
    print_status
    exit 0
fi

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
    echo "[$(date '+%F %T')] stopping DMC on-policy workers ..."
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

preflight() {
    command -v "${PYTHON_BIN}" >/dev/null
    command -v nvidia-smi >/dev/null
    command -v flock >/dev/null

    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
from dm_control import suite
from examples.online.main_dmc_onpolicy import DMControlOnPolicyTrainer, SUPPORTED_AGENTS

tasks = {
    "ball_in_cup-catch",
    "cartpole-balance",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "point_mass-easy",
    "reacher-easy",
    "reacher-hard",
}
registered = {f"{domain}-{task}" for domain, task in suite.ALL_TASKS}
missing = sorted(tasks - registered)
if missing:
    raise RuntimeError(f"Missing DMControl tasks: {missing}")
expected_algos = {"ppo", "dppo", "fpo", "fpopp", "genpo", "policyflow"}
if set(SUPPORTED_AGENTS) != expected_algos:
    raise RuntimeError(f"Unexpected on-policy agents: {sorted(SUPPORTED_AGENTS)}")
print("DMC on-policy preflight: OK")
PY

    for algo in "${ALGOS[@]}"; do
        [[ -f "examples/online/config/dmc_onpolicy/algo/${algo}.yaml" ]]
    done
}

experiment_spec() {
    cat <<EOF
schema_version=3
git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
python=$(${PYTHON_BIN} -c 'import os,sys; print(os.path.realpath(sys.executable))')
python_version=$(${PYTHON_BIN} -c 'import platform; print(platform.python_version())')
algos=${ALGOS[*]}
tasks=${TASKS[*]}
seeds=${SEEDS[*]}
train_frames=${TRAIN_FRAMES}
eval_frames=${EVAL_FRAMES}
log_frames=${LOG_FRAMES}
frame_skip=${FRAME_SKIP}
frame_stack=${FRAME_STACK}
horizon=${HORIZON}
env_mode=${ENV_MODE}
debug_finite_checks=${DEBUG_FINITE_CHECKS}
seed_barrier=${SEED_BARRIER}
log_dir=${LOG_DIR}
log_tag=${LOG_TAG}
log_group=${LOG_GROUP}
budget_tag=${BUDGET_TAG}
schedule_tag=${SCHEDULE_TAG}
seed_protocol_tag=${SEED_PROTOCOL_TAG}
policyflow_timeout_tag=${POLICYFLOW_TIMEOUT_TAG}
eval_rng_tag=${EVAL_RNG_TAG}
wandb_enabled=${WANDB_ENABLED}
wandb_mode=${WANDB_MODE}
wandb_project=${WANDB_PROJECT_NAME}
wandb_entity=${WANDB_ENTITY}
extra_args=${EXTRA_ARGS}
xla_flags=${XLA_FLAGS}
omp_num_threads=${OMP_NUM_THREADS}
mkl_num_threads=${MKL_NUM_THREADS}
openblas_num_threads=${OPENBLAS_NUM_THREADS}
numexpr_num_threads=${NUMEXPR_NUM_THREADS}
EOF
}

source_manifest() {
    {
        find flowrl -type f -name '*.py' -print0
        find examples/online/config/dmc_onpolicy -type f -name '*.yaml' -print0
        printf '%s\0' \
            examples/online/main_dmc_onpolicy.py \
            examples/online/main_mujoco_onpolicy.py \
            scripts/dmc/dmc-onpolicy-baseline-10seed.sh
    } | sort -z | xargs -0 sha256sum
}

write_manifest() {
    {
        echo -e "index\tseed\ttask\talgo"
        local index=0 seed task algo
        for seed in "${SEEDS[@]}"; do
            for task in "${TASKS[@]}"; do
                for algo in "${ALGOS[@]}"; do
                    index=$((index + 1))
                    echo -e "${index}\t${seed}\t${task}\t${algo}"
                done
            done
        done
    } >"${RUN_LOG_ROOT}/manifest.tsv"
}

freeze_source() {
    FROZEN_ROOT="${RUN_LOG_ROOT}/frozen_source"
    FROZEN_ENTRY="${FROZEN_ROOT}/examples/online/main_dmc_onpolicy.py"
    if [[ -e "${FROZEN_ROOT}" ]]; then
        echo "Refusing to overwrite incomplete frozen source: ${FROZEN_ROOT}" >&2
        exit 1
    fi

    mkdir -p "${FROZEN_ROOT}/examples/online/config" "${FROZEN_ROOT}/scripts/dmc"
    cp -a flowrl "${FROZEN_ROOT}/"
    cp examples/online/main_dmc_onpolicy.py "${FROZEN_ROOT}/examples/online/"
    cp examples/online/main_mujoco_onpolicy.py "${FROZEN_ROOT}/examples/online/"
    cp -a examples/online/config/dmc_onpolicy "${FROZEN_ROOT}/examples/online/config/"
    cp scripts/dmc/dmc-onpolicy-baseline-10seed.sh "${FROZEN_ROOT}/scripts/dmc/"

    (
        cd "${FROZEN_ROOT}"
        PYTHONPATH="${FROZEN_ROOT}" "${PYTHON_BIN}" -c \
            'from examples.online.main_dmc_onpolicy import DMControlOnPolicyTrainer'
    )
}

prepare_provenance() {
    local fingerprint_file="${RUN_LOG_ROOT}/launch-fingerprint.txt"
    local stored_fingerprint fingerprint_tmp
    CONFIG_FINGERPRINT="$(experiment_spec | sha256sum | awk '{print $1}')"
    SOURCE_FINGERPRINT="$(source_manifest | sha256sum | awk '{print $1}')"
    LAUNCH_FINGERPRINT="$({
        echo "config=${CONFIG_FINGERPRINT}"
        echo "source=${SOURCE_FINGERPRINT}"
    } | sha256sum | awk '{print $1}')"

    FROZEN_ROOT="${RUN_LOG_ROOT}/frozen_source"
    FROZEN_ENTRY="${FROZEN_ROOT}/examples/online/main_dmc_onpolicy.py"
    if [[ -f "${fingerprint_file}" ]]; then
        stored_fingerprint="$(awk -F= '$1 == "launch" {print $2}' "${fingerprint_file}")"
        if [[ "${stored_fingerprint}" != "${LAUNCH_FINGERPRINT}" ]]; then
            echo "Refusing mixed-version resume in ${RUN_LOG_ROOT}." >&2
            echo "stored launch fingerprint: ${stored_fingerprint}" >&2
            echo "current launch fingerprint: ${LAUNCH_FINGERPRINT}" >&2
            echo "Use the frozen launcher or a new RUN_LOG_ROOT." >&2
            exit 1
        fi
        if [[ ! -f "${FROZEN_ENTRY}" ]]; then
            echo "Missing frozen entry for existing sweep: ${FROZEN_ENTRY}" >&2
            exit 1
        fi
    else
        if [[ -f "${RUN_LOG_ROOT}/run-config.txt" ]]; then
            echo "Refusing legacy/incomplete RUN_LOG_ROOT without fingerprint: ${RUN_LOG_ROOT}" >&2
            exit 1
        fi

        freeze_source
        {
            echo "created_at=$(date --iso-8601=seconds)"
            echo "repo_root=${REPO_ROOT}"
            experiment_spec
            echo "initial_gpus=${RAW_GPUS[*]}"
            echo "initial_runs_per_gpu=${RUNS_PER_GPU}"
            echo "initial_worker_slots=${WORKER_GPUS[*]}"
            echo "max_attempts_per_launch=${MAX_ATTEMPTS}"
            echo "retry_delay_seconds=${RETRY_DELAY_SECONDS}"
        } >"${RUN_LOG_ROOT}/run-config.txt"
        source_manifest >"${RUN_LOG_ROOT}/source-hashes.txt"
        git status --short >"${RUN_LOG_ROOT}/git-status.txt" 2>/dev/null || true
        git diff --stat >"${RUN_LOG_ROOT}/git-diff-stat.txt" 2>/dev/null || true
        "${PYTHON_BIN}" -m pip freeze >"${RUN_LOG_ROOT}/pip-freeze.txt" 2>/dev/null || true
        nvidia-smi >"${RUN_LOG_ROOT}/nvidia-smi.txt"
        write_manifest

        fingerprint_tmp="${fingerprint_file}.tmp-${BASHPID}"
        {
            echo "config=${CONFIG_FINGERPRINT}"
            echo "source=${SOURCE_FINGERPRINT}"
            echo "launch=${LAUNCH_FINGERPRINT}"
        } >"${fingerprint_tmp}"
        mv "${fingerprint_tmp}" "${fingerprint_file}"
    fi

    echo "started_at=$(date --iso-8601=seconds) gpus=${RAW_GPUS[*]} runs_per_gpu=${RUNS_PER_GPU} worker_slots=${WORKER_GPUS[*]}" \
        >>"${RUN_LOG_ROOT}/launcher-history.log"
}

run_one() {
    local algo="$1"
    local task="$2"
    local seed="$3"
    local gpu="$4"
    local progress_idx="$5"
    local run_name="${task}-${algo}-seed${seed}"
    local safe_name run_dir run_log done_file failed_file running_file
    local attempt_file attempt_tmp stamp previous_attempt attempt attempt_name
    local wandb_run_id=""
    safe_name="$(sanitize "${run_name}")"
    local timeout_bootstrap_tag="timeout-bootstrap-standard-final-next-value"
    run_dir="${RUN_LOG_ROOT}/${algo}/${task}"
    run_log="${run_dir}/seed${seed}.log"
    done_file="${RUN_LOG_ROOT}/${safe_name}.done"
    failed_file="${RUN_LOG_ROOT}/${safe_name}.failed"
    running_file="${RUN_LOG_ROOT}/${safe_name}.running"
    attempt_file="${RUN_LOG_ROOT}/${safe_name}.attempt"

    mkdir -p "${run_dir}"
    if [[ -f "${done_file}" ]]; then
        echo "[$(date '+%F %T')] skip done | ${progress_idx}/${TOTAL_RUNS} | algo=${algo} task=${task} seed=${seed}"
        return 0
    fi

    previous_attempt=0
    if [[ -f "${attempt_file}" ]]; then
        previous_attempt="$(<"${attempt_file}")"
        if ! [[ "${previous_attempt}" =~ ^[0-9]+$ ]]; then
            echo "Invalid attempt counter: ${attempt_file}" >&2
            return 1
        fi
    fi
    attempt=$((previous_attempt + 1))
    attempt_name="${run_name}"
    if (( attempt > 1 )); then
        attempt_name="${run_name}-retry${attempt}"
    fi
    if [[ "${WANDB_ENABLED}" != "0" && -n "${WANDB_PROJECT_NAME}" ]]; then
        wandb_run_id="${LAUNCH_FINGERPRINT:0:12}-${safe_name}-a${attempt}"
    fi

    if [[ "${algo}" == "policyflow" ]]; then
        timeout_bootstrap_tag="${POLICYFLOW_TIMEOUT_TAG}"
    fi
    local cmd=(
        "${PYTHON_BIN}" "${FROZEN_ENTRY}"
        "task=${task}"
        "algo=${algo}"
        "seed=${seed}"
        "device=0"
        "frame_skip=${FRAME_SKIP}"
        "frame_stack=${FRAME_STACK}"
        "horizon=${HORIZON}"
        "train_frames=${TRAIN_FRAMES}"
        "eval_frames=${EVAL_FRAMES}"
        "log_frames=${LOG_FRAMES}"
        "env_mode=${ENV_MODE}"
        "debug_finite_checks=${DEBUG_FINITE_CHECKS}"
        "log.dir=${LOG_DIR}"
        "log.tag=${LOG_TAG}"
        "log.group=${LOG_GROUP}"
        "log.name=${attempt_name}"
        "log.tags=[${task},${algo},dmc,onpolicy,baseline,${BUDGET_TAG},${SCHEDULE_TAG},${SEED_PROTOCOL_TAG},${EVAL_RNG_TAG},${timeout_bootstrap_tag},10seed,1024env,16epoch,attempt${attempt}]"
        "log.wandb_mode=${WANDB_MODE}"
        "log.save_ckpt=false"
    )

    if [[ "${WANDB_ENABLED}" == "0" || -z "${WANDB_PROJECT_NAME}" ]]; then
        cmd+=("log.wandb=false")
    else
        cmd+=("log.wandb=true" "log.project=${WANDB_PROJECT_NAME}")
        if [[ -n "${WANDB_ENTITY}" ]]; then
            cmd+=("log.entity=${WANDB_ENTITY}")
        fi
    fi
    if [[ -n "${EXTRA_ARGS}" ]]; then
        local extra=( ${EXTRA_ARGS} )
        cmd+=("${extra[@]}")
    fi

    if [[ "${DRY_RUN}" != "0" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q WANDB_RUN_ID=%q WANDB_RESUME=never ' \
            "${gpu}" "${wandb_run_id}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    stamp="$(date '+%Y%m%d-%H%M%S')-${BASHPID}"
    if [[ -f "${failed_file}" ]]; then
        mv "${failed_file}" "${failed_file}.previous-${stamp}"
    fi
    if [[ -f "${running_file}" ]]; then
        mv "${running_file}" "${running_file}.stale-${stamp}"
    fi
    if [[ -f "${run_log}" ]]; then
        mv "${run_log}" "${run_log}.previous-${stamp}"
    fi

    attempt_tmp="${attempt_file}.tmp-${BASHPID}"
    printf '%s\n' "${attempt}" >"${attempt_tmp}"
    mv "${attempt_tmp}" "${attempt_file}"
    {
        echo "task=${task}"
        echo "algo=${algo}"
        echo "seed=${seed}"
        echo "gpu=${gpu}"
        echo "worker_pid=${BASHPID}"
        echo "attempt=${attempt}"
        echo "wandb_run_id=${wandb_run_id}"
        echo "started_at=$(date --iso-8601=seconds)"
        echo "log=${run_log}"
    } >"${running_file}"

    echo "[$(date '+%F %T')] start | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} attempt=${attempt} wandb_id=${wandb_run_id:-disabled}"
    local start_ts status elapsed
    start_ts="$(date +%s)"
    if (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONPATH="${FROZEN_ROOT}:${PYTHONPATH:-}"
        export WANDB_MODE WANDB_PROJECT_NAME WANDB_ENTITY
        if [[ -n "${wandb_run_id}" ]]; then
            export WANDB_RUN_ID="${wandb_run_id}"
            export WANDB_RESUME=never
        else
            unset WANDB_RUN_ID WANDB_RESUME
        fi
        export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
        export HYDRA_FULL_ERROR XLA_FLAGS
        "${cmd[@]}"
    ) >"${run_log}" 2>&1; then
        status=0
    else
        status=$?
    fi
    elapsed=$(( $(date +%s) - start_ts ))

    {
        echo "exit_code=${status}"
        echo "elapsed_seconds=${elapsed}"
        echo "finished_at=$(date --iso-8601=seconds)"
    } >>"${running_file}"

    if (( status == 0 )); then
        mv "${running_file}" "${done_file}"
        echo "[$(date '+%F %T')] done | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} attempt=${attempt} elapsed=${elapsed}s"
        return 0
    fi

    mv "${running_file}" "${failed_file}"
    echo "[$(date '+%F %T')] failed | ${progress_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} attempt=${attempt} exit=${status} elapsed=${elapsed}s log=${run_log}" >&2
    tail -n 60 "${run_log}" >&2 || true
    return "${status}"
}

worker() {
    local worker_idx="$1"
    local gpu="$2"
    local num_workers="$3"
    local job_idx=0
    local status=0
    local seed task algo launch_attempt completed

    for seed in "${SEEDS[@]}"; do
        for task in "${TASKS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                if (( job_idx % num_workers == worker_idx )); then
                    completed=0
                    for ((launch_attempt = 1; launch_attempt <= MAX_ATTEMPTS; launch_attempt += 1)); do
                        if run_one "${algo}" "${task}" "${seed}" "${gpu}" "$((job_idx + 1))"; then
                            completed=1
                            break
                        fi
                        if (( launch_attempt < MAX_ATTEMPTS )); then
                            echo "[$(date '+%F %T')] retrying | $((job_idx + 1))/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} in ${RETRY_DELAY_SECONDS}s" >&2
                            if (( RETRY_DELAY_SECONDS > 0 )); then
                                sleep "${RETRY_DELAY_SECONDS}"
                            fi
                        fi
                    done
                    if (( completed == 0 )); then
                        status=1
                    fi
                fi
                job_idx=$((job_idx + 1))
            done
        done
    done
    return "${status}"
}

claim_seed_job() {
    local queue_state="$1"
    local queue_fd="$2"
    local jobs_per_seed="$3"
    local next_job

    flock -x "${queue_fd}"
    next_job="$(<"${queue_state}")"
    if (( next_job >= jobs_per_seed )); then
        flock -u "${queue_fd}"
        return 1
    fi
    printf '%s\n' "$((next_job + 1))" >"${queue_state}"
    flock -u "${queue_fd}"
    CLAIMED_SEED_JOB="${next_job}"
}

seed_worker() {
    local worker_idx="$1"
    local gpu="$2"
    local seed="$3"
    local seed_idx="$4"
    local queue_state="$5"
    local queue_lock="$6"
    local jobs_per_seed="$7"
    local num_algos="${#ALGOS[@]}"
    local num_tasks="${#TASKS[@]}"
    local status=0
    local local_job_idx global_job_idx task_idx algo_idx task algo
    local launch_attempt completed

    exec 8>>"${queue_lock}"
    while claim_seed_job "${queue_state}" 8 "${jobs_per_seed}"; do
        local_job_idx="${CLAIMED_SEED_JOB}"
        # Algorithm-major dispatch lets a caller put the slowest algorithms first,
        # reducing the long tail at the end of each strict seed stage.
        algo_idx=$((local_job_idx / num_tasks))
        task_idx=$((local_job_idx % num_tasks))
        task="${TASKS[$task_idx]}"
        algo="${ALGOS[$algo_idx]}"
        global_job_idx=$((seed_idx * jobs_per_seed + task_idx * num_algos + algo_idx + 1))

        completed=0
        for ((launch_attempt = 1; launch_attempt <= MAX_ATTEMPTS; launch_attempt += 1)); do
            if run_one "${algo}" "${task}" "${seed}" "${gpu}" "${global_job_idx}"; then
                completed=1
                break
            fi
            if (( launch_attempt < MAX_ATTEMPTS )); then
                echo "[$(date '+%F %T')] retrying | ${global_job_idx}/${TOTAL_RUNS} | gpu=${gpu} algo=${algo} task=${task} seed=${seed} in ${RETRY_DELAY_SECONDS}s" >&2
                if (( RETRY_DELAY_SECONDS > 0 )); then
                    sleep "${RETRY_DELAY_SECONDS}"
                fi
            fi
        done
        if (( completed == 0 )); then
            status=1
        fi
    done
    echo "[$(date '+%F %T')] seed worker drained | seed=${seed} worker=${worker_idx} gpu=${gpu}"
    return "${status}"
}

run_seed_stage() {
    local seed="$1"
    local seed_idx="$2"
    local num_workers="$3"
    local jobs_per_seed="$4"
    local queue_state="${RUN_LOG_ROOT}/seed${seed}.queue"
    local queue_lock="${RUN_LOG_ROOT}/seed${seed}.queue.lock"
    local seed_status=0
    local done_count=0
    local i pid task algo safe_name

    printf '0\n' >"${queue_state}"
    : >"${queue_lock}"
    PIDS=()
    echo "[$(date '+%F %T')] seed stage start | seed=${seed} index=$((seed_idx + 1))/${#SEEDS[@]} jobs=${jobs_per_seed}" |
        tee -a "${RUN_LOG_ROOT}/seed-stages.log"

    for i in "${!WORKER_GPUS[@]}"; do
        seed_worker "${i}" "${WORKER_GPUS[$i]}" "${seed}" "${seed_idx}" \
            "${queue_state}" "${queue_lock}" "${jobs_per_seed}" &
        PIDS+=("$!")
    done
    printf 'seed=%s started_at=%s pids=%s\n' \
        "${seed}" "$(date --iso-8601=seconds)" "${PIDS[*]}" \
        >>"${RUN_LOG_ROOT}/worker-pids.log"

    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            seed_status=1
        fi
    done
    PIDS=()

    if [[ "${DRY_RUN}" != "0" ]]; then
        done_count="${jobs_per_seed}"
    else
        for task in "${TASKS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                safe_name="$(sanitize "${task}-${algo}-seed${seed}")"
                if [[ -f "${RUN_LOG_ROOT}/${safe_name}.done" ]]; then
                    done_count=$((done_count + 1))
                fi
            done
        done
    fi

    if (( seed_status != 0 || done_count != jobs_per_seed )); then
        echo "[$(date '+%F %T')] seed stage failed | seed=${seed} done=${done_count}/${jobs_per_seed}; later seeds will not start" |
            tee -a "${RUN_LOG_ROOT}/seed-stages.log" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] seed stage complete | seed=${seed} done=${done_count}/${jobs_per_seed}" |
        tee -a "${RUN_LOG_ROOT}/seed-stages.log"
}

main() {
    mkdir -p "${RUN_LOG_ROOT}"
    exec 9>"${RUN_LOG_ROOT}/launcher.lock"
    if ! flock -n 9; then
        echo "Another launcher holds ${RUN_LOG_ROOT}/launcher.lock" >&2
        print_status
        exit 1
    fi

    preflight
    prepare_provenance

    echo "Repo: ${REPO_ROOT}"
    echo "Frozen entry: ${FROZEN_ENTRY}"
    echo "Algorithms: ${ALGOS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Physical GPUs: ${RAW_GPUS[*]}"
    echo "Runs per GPU: ${RUNS_PER_GPU}"
    echo "Worker GPU slots: ${WORKER_GPUS[*]}"
    echo "Per update: 1024 envs x 24 steps = 24576 samples; 4 minibatches x 6144; 16 epochs"
    echo "Frame budget: train=${TRAIN_FRAMES} eval=${EVAL_FRAMES} log=${LOG_FRAMES} frame_skip=${FRAME_SKIP}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME} enabled=${WANDB_ENABLED} mode=${WANDB_MODE} group=${LOG_GROUP}"
    echo "Launch fingerprint: ${LAUNCH_FINGERPRINT}"
    echo "Max attempts per run this launch: ${MAX_ATTEMPTS}"
    echo "Strict seed barrier: ${SEED_BARRIER}"
    echo "Total runs: ${TOTAL_RUNS}"
    echo "Run logs: ${RUN_LOG_ROOT}"

    local num_workers="${#WORKER_GPUS[@]}"
    local status=0
    local jobs_per_seed=$(( ${#TASKS[@]} * ${#ALGOS[@]} ))
    local seed_idx seed i pid
    if [[ "${SEED_BARRIER}" == "1" ]]; then
        for seed_idx in "${!SEEDS[@]}"; do
            seed="${SEEDS[$seed_idx]}"
            if ! run_seed_stage "${seed}" "${seed_idx}" "${num_workers}" "${jobs_per_seed}"; then
                status=1
                break
            fi
        done
    else
        for i in "${!WORKER_GPUS[@]}"; do
            worker "${i}" "${WORKER_GPUS[$i]}" "${num_workers}" &
            PIDS+=("$!")
        done
        printf '%s\n' "${PIDS[@]}" >"${RUN_LOG_ROOT}/worker-pids.txt"

        for pid in "${PIDS[@]}"; do
            if ! wait "${pid}"; then
                status=1
            fi
        done
    fi

    print_status
    if (( status == 0 )); then
        echo "[$(date '+%F %T')] all DMC on-policy runs completed"
    else
        echo "[$(date '+%F %T')] some DMC on-policy runs failed" >&2
    fi
    exit "${status}"
}

main "$@"
