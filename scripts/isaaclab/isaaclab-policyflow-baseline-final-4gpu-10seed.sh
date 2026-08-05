#!/usr/bin/env bash
set -euo pipefail

# Final IsaacLab PolicyFlow baseline sweep.
#
# Coverage:
#   1 algo * 8 tasks * 10 seeds = 80 runs.
#
# Defaults match the final IsaacLab baseline protocol:
#   - W&B project: hiccupnudt/isaaclab-baseline-final
#   - train_frames: 200M
#   - eval_frames: inherited from examples/online/config/isaaclab_onpolicy/config.yaml
#   - run names: <task>-policyflow-seed<seed>
#   - W&B group: isaaclab-policyflow-baseline-final
#   - W&B tags: <task>, policyflow, final, 200m
#   - W&B mode must be online unless explicitly overridden.
#
# Usage:
#   WANDB_ENTITY=hiccupnudt GPUS="0 1 2 3" \
#     bash scripts/isaaclab/isaaclab-policyflow-baseline-final-4gpu-10seed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
TASKS=(${TASKS:-Isaac-Humanoid-v0 Isaac-Cartpole-v0 Isaac-Ant-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Lift-Cube-Franka-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0})

WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-baseline-final}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_GROUP="${LOG_GROUP:-isaaclab-policyflow-baseline-final}"
LOG_TAG="${LOG_TAG:-isaaclab-baseline-final-200m}"
LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_policyflow_baseline_final_4gpu_10seed_200m}"
TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RESUME_WANDB="${RESUME_WANDB:-1}"
ALLOW_WANDB_RESUME_FALLBACK="${ALLOW_WANDB_RESUME_FALLBACK:-0}"
WANDB_RUN_STATES="${WANDB_RUN_STATES:-missing crashed failed killed}"
WANDB_SKIP_FILE="${WANDB_SKIP_FILE:-${LOG_ROOT}/wandb_finished_runs.txt}"
WANDB_STATUS_FILE="${WANDB_STATUS_FILE:-${LOG_ROOT}/wandb_run_status.tsv}"
WANDB_ATTEMPT_FILE="${WANDB_ATTEMPT_FILE:-${LOG_ROOT}/wandb_runs_to_attempt.txt}"

CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
ALLOW_LOW_MEM="${ALLOW_LOW_MEM:-0}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-43200}"

ALLOW_NON_FINAL_WANDB_TARGET="${ALLOW_NON_FINAL_WANDB_TARGET:-0}"
ALLOW_NON_FINAL_PROTOCOL="${ALLOW_NON_FINAL_PROTOCOL:-0}"
ALLOW_OFFLINE_WANDB="${ALLOW_OFFLINE_WANDB:-0}"

TOTAL_RUNS=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))

mkdir -p "${LOG_ROOT}"

run_name_for() {
    local task="$1"
    local seed="$2"
    echo "${task}-policyflow-seed${seed}"
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

    echo "Refusing to launch a non-final PolicyFlow baseline. Set the matching ALLOW_* override only for debugging." >&2
    return 1
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
    : >"${WANDB_SKIP_FILE}"
    : >"${WANDB_STATUS_FILE}"
    : >"${WANDB_ATTEMPT_FILE}"

    if [[ "${RESUME_WANDB}" != "1" ]]; then
        echo "W&B resume: disabled"
        return 0
    fi

    if [[ -z "${WANDB_ENTITY}" ]]; then
        echo "W&B resume: WANDB_ENTITY is empty; only local .done markers will be used." >&2
        RESUME_WANDB=0
        return 0
    fi

    echo "W&B resume: querying ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    export WANDB_PROJECT_NAME WANDB_ENTITY WANDB_SKIP_FILE WANDB_STATUS_FILE WANDB_ATTEMPT_FILE WANDB_RUN_STATES
    export TASKS_JOINED="${TASKS[*]}"
    export SEEDS_JOINED="${SEEDS[*]}"

    if ! python3 - <<'PY_WANDB'
import os
import re
import sys
from collections import defaultdict

try:
    import wandb
except Exception as exc:
    print(f"failed to import wandb: {exc}", file=sys.stderr)
    sys.exit(2)

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT_NAME"]
tasks = os.environ["TASKS_JOINED"].split()
seeds = os.environ["SEEDS_JOINED"].split()
skip_file = os.environ["WANDB_SKIP_FILE"]
status_file = os.environ["WANDB_STATUS_FILE"]
attempt_file = os.environ["WANDB_ATTEMPT_FILE"]
target_states = {
    state.strip()
    for state in os.environ.get("WANDB_RUN_STATES", "missing").replace(",", " ").split()
    if state.strip()
}
if not target_states:
    target_states = {"missing"}

expected = [f"{task}-policyflow-seed{seed}" for task in tasks for seed in seeds]
expected_set = set(expected)

def write_missing_project_status():
    selected = []
    with open(status_file, "w", encoding="utf-8") as status_f:
        status_f.write("task\talgo\tseed\tstate\trun_id\tname\turl\n")
        for task in tasks:
            for seed in seeds:
                name = f"{task}-policyflow-seed{seed}"
                status_f.write(f"{task}\tpolicyflow\t{seed}\tmissing\t\t{name}\t\n")
                if "missing" in target_states:
                    selected.append(name)
    with open(skip_file, "w", encoding="utf-8"):
        pass
    with open(attempt_file, "w", encoding="utf-8") as attempt_f:
        for name in sorted(set(selected)):
            attempt_f.write(name + "\n")
    print(
        f"wandb expected={len(expected)} finished=0 target_states={sorted(target_states)} "
        f"attempt_count={len(set(selected))} state_counts=missing:{len(expected)}"
    )

try:
    runs = list(wandb.Api().runs(f"{entity}/{project}"))
except Exception as exc:
    msg = str(exc)
    if "Could not find project" in msg or "project not found" in msg.lower():
        print(f"wandb project {entity}/{project} does not exist yet; treating all expected runs as missing.")
        write_missing_project_status()
        sys.exit(0)
    print(f"failed to query wandb runs: {exc}", file=sys.stderr)
    sys.exit(3)

runs_by_name = defaultdict(list)
fallback_by_task_seed = defaultdict(list)
for run in runs:
    name = run.name or ""
    tags = set(run.tags or [])
    if name in expected_set:
        runs_by_name[name].append(run)
        continue
    if "policyflow" not in tags:
        continue
    task = next((tag for tag in tags if tag in tasks), None)
    match = re.search(r"(?:-policyflow)?-seed(\d+)(?:-|$)", name)
    if task is not None and match is not None:
        fallback_by_task_seed[(task, match.group(1))].append(run)

finished = []
selected = []
state_counts = defaultdict(int)
with open(status_file, "w", encoding="utf-8") as status_f:
    status_f.write("task\talgo\tseed\tstate\trun_id\tname\turl\n")
    for task in tasks:
        for seed in seeds:
            name = f"{task}-policyflow-seed{seed}"
            matches = runs_by_name.get(name, []) + fallback_by_task_seed.get((task, seed), [])
            if matches:
                states = {run.state for run in matches}
                newest = sorted(matches, key=lambda r: r.created_at or "", reverse=True)[0]
                if "finished" in states:
                    state = "finished"
                    finished.append(name)
                else:
                    state = newest.state
                run_id = newest.id
                run_name = newest.name
                run_url = newest.url
            else:
                state = "missing"
                run_id = ""
                run_name = name
                run_url = ""

            state_counts[state] += 1
            if state in target_states:
                selected.append(name)
            status_f.write(f"{task}\tpolicyflow\t{seed}\t{state}\t{run_id}\t{run_name}\t{run_url}\n")

with open(skip_file, "w", encoding="utf-8") as skip_f:
    for name in sorted(set(finished)):
        skip_f.write(name + "\n")

with open(attempt_file, "w", encoding="utf-8") as attempt_f:
    for name in sorted(set(selected)):
        attempt_f.write(name + "\n")

counts_str = ",".join(f"{k}:{state_counts[k]}" for k in sorted(state_counts))
print(
    f"wandb expected={len(expected)} finished={len(set(finished))} "
    f"target_states={sorted(target_states)} attempt_count={len(set(selected))} state_counts={counts_str}"
)
PY_WANDB
    then
        if [[ "${ALLOW_WANDB_RESUME_FALLBACK}" == "1" ]]; then
            echo "W&B resume query failed; continuing with local .done markers only." >&2
            : >"${WANDB_SKIP_FILE}"
            : >"${WANDB_ATTEMPT_FILE}"
            RESUME_WANDB=0
            return 0
        fi
        echo "W&B resume query failed; set ALLOW_WANDB_RESUME_FALLBACK=1 to ignore this and run from local markers only." >&2
        return 1
    fi
}

should_skip_run() {
    local run_name="$1"
    local done_file="${LOG_ROOT}/${run_name}.done"

    if [[ "${RESUME_WANDB}" == "1" ]]; then
        if [[ -f "${WANDB_ATTEMPT_FILE}" ]] && grep -Fxq "${run_name}" "${WANDB_ATTEMPT_FILE}"; then
            return 1
        fi
        if [[ -f "${WANDB_SKIP_FILE}" ]] && grep -Fxq "${run_name}" "${WANDB_SKIP_FILE}"; then
            echo "wandb finished"
        else
            echo "wandb state not selected"
        fi
        return 0
    fi

    if [[ -f "${done_file}" ]]; then
        echo "local done marker"
        return 0
    fi
    return 1
}

validate_finished_log() {
    local log_file="$1"
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
    if ! grep -Fq "name: policyflow" "${log_file}"; then
        echo "Post-run validation failed: ${log_file} does not show algo name policyflow." >&2
        errors=1
    fi

    return "${errors}"
}

print_plan() {
    echo "Repo: ${REPO_ROOT}"
    echo "Algo: policyflow"
    echo "Tasks: ${TASKS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Total runs: ${TOTAL_RUNS}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    echo "W&B group: ${LOG_GROUP}"
    echo "W&B tags: <task>, policyflow, final, 200m"
    echo "W&B mode: ${WANDB_MODE}"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "Eval frames override: ${EVAL_FRAMES:-<config default>}"
    echo "Extra args: ${EXTRA_ARGS:-<none>}"
    echo "Stdout logs: ${LOG_ROOT}"
    echo "W&B status cache: ${WANDB_STATUS_FILE}"
    echo "W&B attempt cache: ${WANDB_ATTEMPT_FILE}"
    echo "Resume policy: RESUME_WANDB=${RESUME_WANDB}; attempting W&B states: ${WANDB_RUN_STATES}"
    echo "Job timeout seconds: ${JOB_TIMEOUT_SECONDS}"
}

run_job() {
    local gpu="$1"
    local task="$2"
    local seed="$3"
    local run_index="$4"
    local run_name
    run_name="$(run_name_for "${task}" "${seed}")"
    local log_file="${LOG_ROOT}/${run_name}.gpu${gpu}.log"
    local done_file="${LOG_ROOT}/${run_name}.done"
    local failed_file="${LOG_ROOT}/${run_name}.failed"
    local skip_reason=""

    if skip_reason="$(should_skip_run "${run_name}")"; then
        echo "[$(date '+%F %T')] run ${run_index}/${TOTAL_RUNS} GPU ${gpu}: SKIP ${run_name} (${skip_reason})"
        return 0
    fi

    local cmd=(
        python3 examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=policyflow"
        "seed=${seed}"
        "device=0"
        "log.project=${WANDB_PROJECT_NAME}"
        "log.entity=${WANDB_ENTITY}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tag=${LOG_TAG}"
        "log.tags=[${task},policyflow,final,200m]"
        "log.wandb=true"
        "log.wandb_mode=${WANDB_MODE}"
        "train_frames=${TRAIN_FRAMES}"
    )
    if [[ -n "${EVAL_FRAMES}" ]]; then
        cmd+=("eval_frames=${EVAL_FRAMES}")
    fi
    if [[ -n "${EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        local extra_args_array=( ${EXTRA_ARGS} )
        cmd+=("${extra_args_array[@]}")
    fi

    echo "[$(date '+%F %T')] run ${run_index}/${TOTAL_RUNS} GPU ${gpu}: ${run_name}"
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

    if (( rc == 0 )) && ! validate_finished_log "${log_file}"; then
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
            echo "algo=policyflow"
            echo "seed=${seed}"
            echo "exit_code=${rc}"
            echo "log_file=${log_file}"
        } >"${failed_file}"
    fi
    return "${rc}"
}

worker() {
    local worker_id="$1"
    local gpu="$2"
    local idx=0
    local failures=0

    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if (( idx % ${#GPUS[@]} == worker_id )); then
                if ! run_job "${gpu}" "${task}" "${seed}" "$((idx + 1))"; then
                    echo "FAILED: run $((idx + 1))/${TOTAL_RUNS} task=${task} algo=policyflow seed=${seed} gpu=${gpu}" >&2
                    failures=$((failures + 1))
                fi
            fi
            idx=$((idx + 1))
        done
    done

    return "${failures}"
}

if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
    exit 1
fi

validate_final_protocol
check_gpu_memory
print_plan
write_wandb_status_cache

pids=()
for worker_id in "${!GPUS[@]}"; do
    worker "${worker_id}" "${GPUS[$worker_id]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if (( status == 0 )); then
    echo "All final PolicyFlow IsaacLab baseline jobs completed."
else
    echo "Some final PolicyFlow IsaacLab baseline jobs failed; inspect ${LOG_ROOT}." >&2
fi

if (( status != 0 )); then
    exit "${status}"
fi
exit 0
