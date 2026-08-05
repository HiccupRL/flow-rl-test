#!/usr/bin/env bash
set -euo pipefail

# Complete the final IsaacLab baseline matrix on one 4-GPU machine.
#
# Target matrix:
#   6 algos * 12 tasks * 10 seeds = 720 expected W&B runs.
#
# This script queries hiccupnudt/isaaclab-baseline-final first and only launches
# combinations whose W&B state is selected by WANDB_RUN_STATES. Finished runs are
# skipped even if local markers are absent.
#
# Defaults:
#   - W&B project: hiccupnudt/isaaclab-baseline-final
#   - W&B mode: online
#   - train_frames: 200M
#   - GPUS: 0 1 2 3
#   - algos: ppo dppo fpo fpopp genpo policyflow
#   - retry states: missing crashed failed killed
#
# Usage:
#   bash scripts/isaaclab/isaaclab-baseline-final-complete-12task-4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
ALGOS=(${ALGOS:-ppo dppo fpo fpopp genpo policyflow})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
TASKS=(${TASKS:-Isaac-Ant-v0 Isaac-Cartpole-v0 Isaac-Humanoid-v0 Isaac-Lift-Cube-Franka-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Repose-Cube-Shadow-Direct-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Rough-Unitree-Go2-v0 Isaac-Velocity-Rough-H1-v0 Isaac-Quadcopter-Direct-v0})

WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-baseline-final}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_TAG="${LOG_TAG:-isaaclab-baseline-final-200m}"
LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_baseline_final_complete_12task_4gpu_10seed_200m}"
TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-}"
GENPO_EXTRA_ARGS="${GENPO_EXTRA_ARGS:-algo.batch_size=4096 algo.num_minibatches=6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RESUME_WANDB="${RESUME_WANDB:-1}"
ALLOW_WANDB_RESUME_FALLBACK="${ALLOW_WANDB_RESUME_FALLBACK:-0}"
WANDB_RUN_STATES="${WANDB_RUN_STATES:-missing crashed failed killed}"
WANDB_STATUS_FILE="${WANDB_STATUS_FILE:-${LOG_ROOT}/wandb_run_status.tsv}"
WANDB_ATTEMPT_FILE="${WANDB_ATTEMPT_FILE:-${LOG_ROOT}/wandb_runs_to_attempt.tsv}"
WANDB_FINISHED_FILE="${WANDB_FINISHED_FILE:-${LOG_ROOT}/wandb_finished_runs.txt}"

CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
ALLOW_LOW_MEM="${ALLOW_LOW_MEM:-0}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-64800}"

ALLOW_NON_FINAL_WANDB_TARGET="${ALLOW_NON_FINAL_WANDB_TARGET:-0}"
ALLOW_NON_FINAL_PROTOCOL="${ALLOW_NON_FINAL_PROTOCOL:-0}"
ALLOW_OFFLINE_WANDB="${ALLOW_OFFLINE_WANDB:-0}"

TOTAL_EXPECTED=$(( ${#ALGOS[@]} * ${#TASKS[@]} * ${#SEEDS[@]} ))

mkdir -p "${LOG_ROOT}"

run_name_for() {
    local task="$1"
    local algo="$2"
    local seed="$3"
    echo "${task}-${algo}-seed${seed}"
}

algo_extra_args() {
    local algo="$1"
    if [[ "${algo}" == "genpo" ]]; then
        echo "${GENPO_EXTRA_ARGS} ${EXTRA_ARGS}"
    else
        echo "${EXTRA_ARGS}"
    fi
}

validate_final_protocol() {
    local errors=0

    if [[ "${WANDB_ENTITY}" != "hiccupnudt" || "${WANDB_PROJECT_NAME}" != "isaaclab-baseline-final" ]]; then
        echo "W&B target is ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}, not hiccupnudt/isaaclab-baseline-final." >&2
        errors=1
    fi

    if [[ "${TRAIN_FRAMES}" != "200000000" ]]; then
        echo "TRAIN_FRAMES=${TRAIN_FRAMES}, not the final baseline value 200000000." >&2
        errors=1
    fi

    if [[ "${WANDB_MODE}" != "online" ]]; then
        echo "WANDB_MODE=${WANDB_MODE}; final baseline runs must sync online." >&2
        errors=1
    fi

    if (( errors == 0 )); then
        return 0
    fi

    if [[ "${ALLOW_NON_FINAL_WANDB_TARGET}" == "1" || "${ALLOW_NON_FINAL_PROTOCOL}" == "1" || "${ALLOW_OFFLINE_WANDB}" == "1" ]]; then
        if [[ "${WANDB_ENTITY}" != "hiccupnudt" || "${WANDB_PROJECT_NAME}" != "isaaclab-baseline-final" ]] && [[ "${ALLOW_NON_FINAL_WANDB_TARGET}" != "1" ]]; then
            return 1
        fi
        if [[ "${TRAIN_FRAMES}" != "200000000" ]] && [[ "${ALLOW_NON_FINAL_PROTOCOL}" != "1" ]]; then
            return 1
        fi
        if [[ "${WANDB_MODE}" != "online" ]] && [[ "${ALLOW_OFFLINE_WANDB}" != "1" ]]; then
            return 1
        fi
        echo "Proceeding with explicit non-final override flags." >&2
        return 0
    fi

    echo "Refusing to launch a non-final baseline completion run. Set the matching ALLOW_* override only for debugging." >&2
    return 1
}

validate_task_configs() {
    local missing=0
    local task
    for task in "${TASKS[@]}"; do
        if [[ ! -f "examples/online/config/isaaclab_onpolicy/task/${task}.yaml" ]]; then
            echo "Missing task config: examples/online/config/isaaclab_onpolicy/task/${task}.yaml" >&2
            missing=1
        fi
    done
    return "${missing}"
}

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
    for gpu in "${GPUS[@]}"; do
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

write_wandb_status_cache() {
    : >"${WANDB_STATUS_FILE}"
    : >"${WANDB_ATTEMPT_FILE}"
    : >"${WANDB_FINISHED_FILE}"

    if [[ "${RESUME_WANDB}" != "1" ]]; then
        echo "W&B resume: disabled; scheduling every configured combination."
        {
            echo -e "task\talgo\tseed\tstate\tname"
            for algo in "${ALGOS[@]}"; do
                for task in "${TASKS[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        echo -e "${task}\t${algo}\t${seed}\tforced\t$(run_name_for "${task}" "${algo}" "${seed}")"
                    done
                done
            done
        } >"${WANDB_ATTEMPT_FILE}"
        return 0
    fi

    if [[ -z "${WANDB_ENTITY}" ]]; then
        echo "W&B resume requires WANDB_ENTITY." >&2
        return 1
    fi

    echo "W&B resume: querying ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    export WANDB_PROJECT_NAME WANDB_ENTITY WANDB_STATUS_FILE WANDB_ATTEMPT_FILE WANDB_FINISHED_FILE WANDB_RUN_STATES
    export TASKS_JOINED="${TASKS[*]}"
    export SEEDS_JOINED="${SEEDS[*]}"
    export ALGOS_JOINED="${ALGOS[*]}"

    if ! python3 - <<'PY_WANDB'
import os
import re
import sys
from collections import Counter, defaultdict

try:
    import wandb
except Exception as exc:
    print(f"failed to import wandb: {exc}", file=sys.stderr)
    sys.exit(2)

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT_NAME"]
tasks = os.environ["TASKS_JOINED"].split()
algos = os.environ["ALGOS_JOINED"].split()
seeds = os.environ["SEEDS_JOINED"].split()
status_file = os.environ["WANDB_STATUS_FILE"]
attempt_file = os.environ["WANDB_ATTEMPT_FILE"]
finished_file = os.environ["WANDB_FINISHED_FILE"]
target_states = {
    state.strip()
    for state in os.environ.get("WANDB_RUN_STATES", "missing").replace(",", " ").split()
    if state.strip()
}
if not target_states:
    target_states = {"missing"}

expected = {
    f"{task}-{algo}-seed{seed}": (task, algo, seed)
    for task in tasks
    for algo in algos
    for seed in seeds
}
expected_names = set(expected)
api = wandb.Api(timeout=60)
by_name = defaultdict(list)

try:
    for algo in algos:
        seen_ids = set()
        filters = (
            {"tags": algo},
            {"display_name": {"$regex": f"-{re.escape(algo)}-seed[0-9]"}},
        )
        for filt in filters:
            for run in api.runs(f"{entity}/{project}", filters=filt, per_page=100):
                if run.id in seen_ids:
                    continue
                seen_ids.add(run.id)
                name = run.name or ""
                if name in expected_names:
                    by_name[name].append((run.state, run.id, run.name or "", run.url, run.created_at or ""))
except Exception as exc:
    print(f"failed to query wandb runs: {exc}", file=sys.stderr)
    sys.exit(3)

status_rows = []
attempt_rows = []
finished_names = []
state_counts = Counter()

for task in tasks:
    for algo in algos:
        for seed in seeds:
            name = f"{task}-{algo}-seed{seed}"
            matches = by_name.get(name, [])
            if matches:
                states = {run[0] for run in matches}
                newest = sorted(matches, key=lambda r: r[4], reverse=True)[0]
                if "finished" in states:
                    state = "finished"
                    finished = [run for run in matches if run[0] == "finished"]
                    newest = sorted(finished, key=lambda r: r[4], reverse=True)[0]
                    finished_names.append(name)
                else:
                    state = newest[0]
                run_id, run_name, run_url = newest[1], newest[2], newest[3]
            else:
                state = "missing"
                run_id, run_name, run_url = "", name, ""

            state_counts[state] += 1
            status_rows.append((task, algo, seed, state, run_id, run_name, run_url))
            if state in target_states:
                attempt_rows.append((task, algo, seed, state, name))

with open(status_file, "w", encoding="utf-8") as f:
    f.write("task\talgo\tseed\tstate\trun_id\tname\turl\n")
    for row in status_rows:
        f.write("\t".join(row) + "\n")

with open(attempt_file, "w", encoding="utf-8") as f:
    f.write("task\talgo\tseed\tstate\tname\n")
    for row in attempt_rows:
        f.write("\t".join(row) + "\n")

with open(finished_file, "w", encoding="utf-8") as f:
    for name in sorted(set(finished_names)):
        f.write(name + "\n")

counts_str = ",".join(f"{k}:{state_counts[k]}" for k in sorted(state_counts))
print(
    f"wandb expected={len(expected)} finished={len(set(finished_names))} "
    f"target_states={sorted(target_states)} attempt_count={len(attempt_rows)} state_counts={counts_str}"
)
PY_WANDB
    then
        if [[ "${ALLOW_WANDB_RESUME_FALLBACK}" == "1" ]]; then
            echo "W&B resume query failed; scheduling every configured combination because fallback is enabled." >&2
            RESUME_WANDB=0
            write_wandb_status_cache
            return 0
        fi
        echo "W&B resume query failed; aborting to avoid duplicate or mis-targeted final runs." >&2
        return 1
    fi
}

validate_finished_log() {
    local log_file="$1"
    local algo="$2"
    local errors=0

    if grep -Eq 'wandb_mode: offline|W&B syncing is set to `offline`|wandb sync ' "${log_file}"; then
        echo "Post-run validation failed: ${log_file} contains offline W&B markers." >&2
        errors=1
    fi
    if ! grep -Fq "project: ${WANDB_PROJECT_NAME}" "${log_file}"; then
        echo "Post-run validation failed: ${log_file} does not show project: ${WANDB_PROJECT_NAME}." >&2
        errors=1
    fi
    if ! grep -Fq "train_frames: ${TRAIN_FRAMES}" "${log_file}"; then
        echo "Post-run validation failed: ${log_file} does not show train_frames: ${TRAIN_FRAMES}." >&2
        errors=1
    fi
    if ! grep -Fq "name: ${algo}" "${log_file}"; then
        echo "Post-run validation failed: ${log_file} does not show algo name ${algo}." >&2
        errors=1
    fi

    return "${errors}"
}

print_plan() {
    local attempt_count
    attempt_count=$(( $(wc -l <"${WANDB_ATTEMPT_FILE}") - 1 ))
    if (( attempt_count < 0 )); then
        attempt_count=0
    fi

    echo "Repo: ${REPO_ROOT}"
    echo "Algos: ${ALGOS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Total expected matrix: ${TOTAL_EXPECTED}"
    echo "Runs scheduled this launch: ${attempt_count}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    echo "W&B group: isaaclab-<algo>-baseline-final"
    echo "W&B tags: <task>, <algo>, final, 200m"
    echo "W&B mode: ${WANDB_MODE}"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "Eval frames override: ${EVAL_FRAMES:-<config default>}"
    echo "GENPO extra args: ${GENPO_EXTRA_ARGS}"
    echo "Extra args: ${EXTRA_ARGS:-<none>}"
    echo "Stdout logs: ${LOG_ROOT}"
    echo "W&B status cache: ${WANDB_STATUS_FILE}"
    echo "W&B attempt queue: ${WANDB_ATTEMPT_FILE}"
    echo "Resume policy: RESUME_WANDB=${RESUME_WANDB}; attempting W&B states: ${WANDB_RUN_STATES}"
    echo "Job timeout seconds: ${JOB_TIMEOUT_SECONDS}"
}

run_job() {
    local gpu="$1"
    local task="$2"
    local algo="$3"
    local seed="$4"
    local source_state="$5"
    local run_index="$6"
    local total_runs="$7"
    local run_name
    run_name="$(run_name_for "${task}" "${algo}" "${seed}")"
    local log_file="${LOG_ROOT}/${run_name}.gpu${gpu}.log"
    local done_file="${LOG_ROOT}/${run_name}.done"
    local failed_file="${LOG_ROOT}/${run_name}.failed"

    local cmd=(
        python3 examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=${algo}"
        "seed=${seed}"
        "device=0"
        "log.project=${WANDB_PROJECT_NAME}"
        "log.entity=${WANDB_ENTITY}"
        "log.group=isaaclab-${algo}-baseline-final"
        "log.name=${run_name}"
        "log.tag=${LOG_TAG}"
        "log.tags=[${task},${algo},final,200m]"
        "log.wandb=true"
        "log.wandb_mode=${WANDB_MODE}"
        "train_frames=${TRAIN_FRAMES}"
    )
    if [[ -n "${EVAL_FRAMES}" ]]; then
        cmd+=("eval_frames=${EVAL_FRAMES}")
    fi

    local merged_extra_args
    merged_extra_args="$(algo_extra_args "${algo}")"
    if [[ -n "${merged_extra_args}" ]]; then
        # shellcheck disable=SC2206
        local extra_args_array=( ${merged_extra_args} )
        cmd+=("${extra_args_array[@]}")
    fi

    echo "[$(date '+%F %T')] run ${run_index}/${total_runs} GPU ${gpu}: ${run_name} (wandb_state=${source_state})"
    if [[ -n "${DRY_RUN:-}" ]]; then
        local dry_prefix=""
        local dry_cmd=""
        printf -v dry_prefix 'CUDA_VISIBLE_DEVICES=%q EGL_VISIBLE_DEVICES=%q PYTHONPATH=%q WANDB_MODE=%q WANDB_PROJECT=%q WANDB_ENTITY=%q FLOWRL_ISAACLAB_CLOSE_APP=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ' "${gpu}" "${gpu}" "${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${WANDB_MODE}" "${WANDB_PROJECT_NAME}" "${WANDB_ENTITY}"
        printf -v dry_cmd '%q ' "${cmd[@]}"
        printf '%s%s\n' "${dry_prefix}" "${dry_cmd}"
        return 0
    fi

    set +e
    if (( JOB_TIMEOUT_SECONDS > 0 )); then
        timeout --preserve-status --signal=TERM --kill-after=120s "${JOB_TIMEOUT_SECONDS}" bash -c '
            export CUDA_VISIBLE_DEVICES="$1"
            export EGL_VISIBLE_DEVICES="$1"
            export PYTHONPATH="$2"
            export WANDB_MODE="$3"
            export WANDB_PROJECT="$4"
            export WANDB_ENTITY="$5"
            export FLOWRL_ISAACLAB_CLOSE_APP=0
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export PYTHONUNBUFFERED=1
            shift 5
            "$@"
        ' bash "${gpu}" "${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${WANDB_MODE}" "${WANDB_PROJECT_NAME}" "${WANDB_ENTITY}" "${cmd[@]}" >"${log_file}" 2>&1
    else
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export EGL_VISIBLE_DEVICES="${gpu}"
            export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
            export WANDB_MODE="${WANDB_MODE}"
            export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
            export WANDB_ENTITY="${WANDB_ENTITY}"
            export FLOWRL_ISAACLAB_CLOSE_APP=0
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export PYTHONUNBUFFERED=1
            "${cmd[@]}"
        ) >"${log_file}" 2>&1
    fi
    local rc=$?
    set -e

    if (( rc == 0 )) && ! validate_finished_log "${log_file}" "${algo}"; then
        rc=90
    fi

    if (( rc == 0 )); then
        date '+%F %T' >"${done_file}"
        rm -f "${failed_file}"
    else
        {
            echo "time=$(date '+%F %T')"
            echo "gpu=${gpu}"
            echo "task=${task}"
            echo "algo=${algo}"
            echo "seed=${seed}"
            echo "source_wandb_state=${source_state}"
            echo "exit_code=${rc}"
            echo "log_file=${log_file}"
        } >"${failed_file}"
    fi
    return "${rc}"
}

worker() {
    local worker_id="$1"
    local gpu="$2"
    local total_runs="$3"
    local idx=0
    local failures=0
    local header_seen=0
    local task algo seed source_state name

    while IFS=$'\t' read -r task algo seed source_state name; do
        if (( header_seen == 0 )); then
            header_seen=1
            continue
        fi
        if (( idx % ${#GPUS[@]} == worker_id )); then
            if ! run_job "${gpu}" "${task}" "${algo}" "${seed}" "${source_state}" "$((idx + 1))" "${total_runs}"; then
                echo "FAILED: run $((idx + 1))/${total_runs} task=${task} algo=${algo} seed=${seed} gpu=${gpu}" >&2
                failures=$((failures + 1))
            fi
        fi
        idx=$((idx + 1))
    done <"${WANDB_ATTEMPT_FILE}"

    return "${failures}"
}

if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
    exit 1
fi

validate_final_protocol
validate_task_configs
check_gpu_memory
write_wandb_status_cache
print_plan

attempt_count=$(( $(wc -l <"${WANDB_ATTEMPT_FILE}") - 1 ))
if (( attempt_count <= 0 )); then
    echo "No runs need to be launched for selected states: ${WANDB_RUN_STATES}"
    exit 0
fi

pids=()
for worker_id in "${!GPUS[@]}"; do
    worker "${worker_id}" "${GPUS[$worker_id]}" "${attempt_count}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if (( status == 0 )); then
    echo "All scheduled final IsaacLab baseline completion jobs finished."
else
    echo "Some final IsaacLab baseline completion jobs failed; inspect ${LOG_ROOT}." >&2
fi

if (( status != 0 )); then
    exit "${status}"
fi
exit 0
