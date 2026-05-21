#!/usr/bin/env bash
set -euo pipefail

# Run IsaacLab GENPO baselines on one 4-GPU machine.
#
# Default count:
#   8 tasks * 10 seeds = 80 runs.
#
# Usage:
#   WANDB_ENTITY=hiccupnudt GPUS="0 1 2 3" bash scripts/isaaclab/isaaclab-genpo-baseline-4gpu-10seed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
TASKS=(${TASKS:-Isaac-Ant-v0 Isaac-Cartpole-v0 Isaac-Humanoid-v0 Isaac-Lift-Cube-Franka-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0})

WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-ppo-baseline}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_GROUP="${LOG_GROUP:-isaaclab-genpo-baseline}"
LOG_TAG="${LOG_TAG:-genpo-baseline-10seed}"
LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_genpo_baseline_4gpu_10seed}"
TRAIN_FRAMES="${TRAIN_FRAMES:-100000000}"
EVAL_FRAMES="${EVAL_FRAMES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RESUME_WANDB="${RESUME_WANDB:-1}"
ALLOW_WANDB_RESUME_FALLBACK="${ALLOW_WANDB_RESUME_FALLBACK:-0}"
WANDB_SKIP_FILE="${WANDB_SKIP_FILE:-${LOG_ROOT}/wandb_finished_runs.txt}"
WANDB_STATUS_FILE="${WANDB_STATUS_FILE:-${LOG_ROOT}/wandb_run_status.tsv}"

mkdir -p "${LOG_ROOT}"

run_name_for() {
    local task="$1"
    local seed="$2"
    echo "${task}-genpo-seed${seed}"
}

write_wandb_status_cache() {
    : >"${WANDB_SKIP_FILE}"
    : >"${WANDB_STATUS_FILE}"

    if [[ "${RESUME_WANDB}" != "1" ]]; then
        echo "W&B resume: disabled"
        return 0
    fi

    if [[ -z "${WANDB_ENTITY}" ]]; then
        echo "W&B resume: WANDB_ENTITY is empty; only local .done markers will be used." >&2
        return 0
    fi

    echo "W&B resume: querying ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    export WANDB_PROJECT_NAME WANDB_ENTITY WANDB_SKIP_FILE WANDB_STATUS_FILE
    export TASKS_JOINED="${TASKS[*]}"
    export SEEDS_JOINED="${SEEDS[*]}"

    if ! python3 - <<'PY'
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
expected = [f"{task}-genpo-seed{seed}" for task in tasks for seed in seeds]
expected_set = set(expected)

try:
    runs = list(wandb.Api().runs(f"{entity}/{project}"))
except Exception as exc:
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
    if "genpo" not in tags:
        continue
    task = next((tag for tag in tags if tag in tasks), None)
    match = re.search(r"(?:-genpo)?-seed(\d+)(?:-|$)", name)
    if task is not None and match is not None:
        fallback_by_task_seed[(task, match.group(1))].append(run)

finished = []
with open(status_file, "w", encoding="utf-8") as status_f:
    status_f.write("task\talgo\tseed\tstate\trun_id\tname\turl\n")
    for task in tasks:
        for seed in seeds:
            name = f"{task}-genpo-seed{seed}"
            matches = runs_by_name.get(name, []) + fallback_by_task_seed.get((task, seed), [])
            if not matches:
                status_f.write(f"{task}\tgenpo\t{seed}\tmissing\t\t{name}\t\n")
                continue
            states = {run.state for run in matches}
            if "finished" in states:
                finished.append(name)
            newest = sorted(matches, key=lambda r: r.created_at or "", reverse=True)[0]
            status_f.write(
                f"{task}\tgenpo\t{seed}\t{newest.state}\t{newest.id}\t{newest.name}\t{newest.url}\n"
            )

with open(skip_file, "w", encoding="utf-8") as skip_f:
    for name in sorted(set(finished)):
        skip_f.write(name + "\n")

print(f"wandb expected={len(expected)} finished={len(set(finished))} resume_remaining={len(expected)-len(set(finished))}")
PY
    then
        if [[ "${ALLOW_WANDB_RESUME_FALLBACK}" == "1" ]]; then
            echo "W&B resume query failed; continuing with local .done markers only." >&2
            : >"${WANDB_SKIP_FILE}"
            return 0
        fi
        echo "W&B resume query failed; set ALLOW_WANDB_RESUME_FALLBACK=1 to ignore this and run from local markers only." >&2
        return 1
    fi
}

should_skip_run() {
    local run_name="$1"
    local done_file="${LOG_ROOT}/${run_name}.done"

    if [[ -f "${done_file}" ]]; then
        echo "local done marker"
        return 0
    fi
    if [[ "${RESUME_WANDB}" == "1" && -f "${WANDB_SKIP_FILE}" ]] && grep -Fxq "${run_name}" "${WANDB_SKIP_FILE}"; then
        echo "wandb finished"
        return 0
    fi
    return 1
}

print_plan() {
    local total_runs=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
    echo "Repo: ${REPO_ROOT}"
    echo "Tasks: ${TASKS[*]}"
    echo "Algo: genpo"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Total runs: ${total_runs}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    echo "W&B group: ${LOG_GROUP}"
    echo "W&B tags: <task>, genpo"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "Stdout logs: ${LOG_ROOT}"
    echo "W&B status cache: ${WANDB_STATUS_FILE}"
}

run_job() {
    local gpu="$1"
    local task="$2"
    local seed="$3"
    local run_name
    run_name="$(run_name_for "${task}" "${seed}")"
    local log_file="${LOG_ROOT}/${run_name}.gpu${gpu}.log"
    local done_file="${LOG_ROOT}/${run_name}.done"
    local failed_file="${LOG_ROOT}/${run_name}.failed"
    local skip_reason=""

    if skip_reason="$(should_skip_run "${run_name}")"; then
        echo "[$(date '+%F %T')] GPU ${gpu}: SKIP ${run_name} (${skip_reason})"
        return 0
    fi

    local cmd=(
        python3 examples/online/main_isaaclab_onpolicy.py
        "task=${task}"
        "algo=genpo"
        "seed=${seed}"
        "device=0"
        "log.project=${WANDB_PROJECT_NAME}"
        "log.entity=${WANDB_ENTITY}"
        "log.group=${LOG_GROUP}"
        "log.name=${run_name}"
        "log.tag=${LOG_TAG}"
        "log.tags=[${task},genpo]"
        "log.wandb=true"
        "log.wandb_mode=${WANDB_MODE}"
        "train_frames=${TRAIN_FRAMES}"
    )
    if [[ -n "${EVAL_FRAMES}" ]]; then
        cmd+=("eval_frames=${EVAL_FRAMES}")
    fi
    if [[ -n "${EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        extra_args_array=( ${EXTRA_ARGS} )
        cmd+=("${extra_args_array[@]}")
    fi

    echo "[$(date '+%F %T')] GPU ${gpu}: ${run_name}"
    if [[ -n "${DRY_RUN:-}" ]]; then
        local dry_prefix=""
        local dry_cmd=""
        printf -v dry_prefix 'CUDA_VISIBLE_DEVICES=%q EGL_VISIBLE_DEVICES=%q PYTHONPATH=%q WANDB_MODE=%q FLOWRL_ISAACLAB_CLOSE_APP=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ' "${gpu}" "${gpu}" "${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${WANDB_MODE}"
        printf -v dry_cmd '%q ' "${cmd[@]}"
        printf '%s%s
' "${dry_prefix}" "${dry_cmd}"
        return 0
    fi

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EGL_VISIBLE_DEVICES="${gpu}"
        export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        export WANDB_MODE="${WANDB_MODE}"
        export FLOWRL_ISAACLAB_CLOSE_APP=0
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PYTHONUNBUFFERED=1
        "${cmd[@]}"
    ) >"${log_file}" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        date '+%F %T' >"${done_file}"
        rm -f "${failed_file}"
    else
        {
            echo "time=$(date '+%F %T')"
            echo "gpu=${gpu}"
            echo "task=${task}"
            echo "algo=genpo"
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
                if ! run_job "${gpu}" "${task}" "${seed}"; then
                    echo "FAILED: task=${task} algo=genpo seed=${seed} gpu=${gpu}" >&2
                    failures=$((failures + 1))
                fi
            fi
            idx=$((idx + 1))
        done
    done

    return "${failures}"
}

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
    echo "All GENPO baseline jobs completed."
else
    echo "Some GENPO baseline jobs failed; inspect ${LOG_ROOT}." >&2
fi

if (( status != 0 )); then
    exit "${status}"
fi
exit 0
