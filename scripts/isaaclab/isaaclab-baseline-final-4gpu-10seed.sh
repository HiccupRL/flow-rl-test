#!/usr/bin/env bash
set -euo pipefail

# Final IsaacLab baseline sweep.
#
# Coverage:
#   5 algos * 8 tasks * 10 seeds = 400 runs.
#
# Defaults:
#   - W&B project: isaaclab-baseline-final
#   - train_frames: 200M for every algorithm
#   - GENPO uses smaller minibatches on all tasks: batch_size=4096, num_minibatches=6
#     This preserves the full 1024*24 rollout coverage while fitting the G1 tasks on 48G GPUs.
#   - Local .done markers are used for same-directory resume; W&B resume is opt-in via RESUME_WANDB=1.
#
# Usage:
#   WANDB_ENTITY=hiccupnudt GPUS="0 1 2 3" \
#     bash scripts/isaaclab/isaaclab-baseline-final-4gpu-10seed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS=(${GPUS:-0 1 2 3})
ALGOS=(${ALGOS:-ppo dppo fpo fpopp genpo})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
TASKS=(${TASKS:-Isaac-Humanoid-v0 Isaac-Cartpole-v0 Isaac-Ant-v0 Isaac-Open-Drawer-Franka-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Lift-Cube-Franka-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0})

WANDB_ENTITY="${WANDB_ENTITY:-hiccupnudt}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-isaaclab-baseline-final}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_TAG="${LOG_TAG:-isaaclab-baseline-final-200m}"
LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_baseline_final_4gpu_10seed_200m}"
TOTAL_RUNS=$(( ${#ALGOS[@]} * ${#TASKS[@]} * ${#SEEDS[@]} ))
TRAIN_FRAMES="${TRAIN_FRAMES:-200000000}"
EVAL_FRAMES="${EVAL_FRAMES:-}"
GENPO_EXTRA_ARGS="${GENPO_EXTRA_ARGS:-algo.batch_size=4096 algo.num_minibatches=6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RESUME_WANDB="${RESUME_WANDB:-0}"
ALLOW_WANDB_RESUME_FALLBACK="${ALLOW_WANDB_RESUME_FALLBACK:-1}"
WANDB_SKIP_FILE="${WANDB_SKIP_FILE:-${LOG_ROOT}/wandb_finished_runs.txt}"
WANDB_STATUS_FILE="${WANDB_STATUS_FILE:-${LOG_ROOT}/wandb_run_status.tsv}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
ALLOW_LOW_MEM="${ALLOW_LOW_MEM:-0}"

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
    export ALGOS_JOINED="${ALGOS[*]}"

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
algos = os.environ["ALGOS_JOINED"].split()
skip_file = os.environ["WANDB_SKIP_FILE"]
status_file = os.environ["WANDB_STATUS_FILE"]
expected = [f"{task}-{algo}-seed{seed}" for algo in algos for task in tasks for seed in seeds]
expected_set = set(expected)

try:
    runs = list(wandb.Api().runs(f"{entity}/{project}"))
except Exception as exc:
    msg = str(exc)
    if "Could not find project" in msg or "project not found" in msg.lower():
        print(f"wandb project {entity}/{project} does not exist yet; first online run will create it.")
        sys.exit(0)
    print(f"failed to query wandb runs: {exc}", file=sys.stderr)
    sys.exit(3)

runs_by_name = defaultdict(list)
fallback_by_key = defaultdict(list)
for run in runs:
    name = run.name or ""
    tags = set(run.tags or [])
    if name in expected_set:
        runs_by_name[name].append(run)
        continue
    task = next((tag for tag in tags if tag in tasks), None)
    algo = next((tag for tag in tags if tag in algos), None)
    match = re.search(r"(?:-seed|seed)(\d+)(?:-|$)", name)
    if task is not None and algo is not None and match is not None:
        fallback_by_key[(task, algo, match.group(1))].append(run)

finished = []
with open(status_file, "w", encoding="utf-8") as status_f:
    status_f.write("task\talgo\tseed\tstate\trun_id\tname\turl\n")
    for algo in algos:
        for task in tasks:
            for seed in seeds:
                name = f"{task}-{algo}-seed{seed}"
                matches = runs_by_name.get(name, []) + fallback_by_key.get((task, algo, seed), [])
                if not matches:
                    status_f.write(f"{task}\t{algo}\t{seed}\tmissing\t\t{name}\t\n")
                    continue
                states = {run.state for run in matches}
                if "finished" in states:
                    finished.append(name)
                newest = sorted(matches, key=lambda r: r.created_at or "", reverse=True)[0]
                status_f.write(
                    f"{task}\t{algo}\t{seed}\t{newest.state}\t{newest.id}\t{newest.name}\t{newest.url}\n"
                )

with open(skip_file, "w", encoding="utf-8") as skip_f:
    for name in sorted(set(finished)):
        skip_f.write(name + "\n")

print(f"wandb expected={len(expected)} finished={len(set(finished))} resume_remaining={len(expected)-len(set(finished))}")
PY_WANDB
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
    local total_runs="${TOTAL_RUNS}"
    echo "Repo: ${REPO_ROOT}"
    echo "Algos: ${ALGOS[*]}"
    echo "Tasks: ${TASKS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs: ${GPUS[*]}"
    echo "Total runs: ${total_runs}"
    echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}"
    echo "W&B group: isaaclab-<algo>-baseline-final"
    echo "W&B tags: <task>, <algo>, final, 200m"
    echo "Train frames: ${TRAIN_FRAMES}"
    echo "GENPO extra args: ${GENPO_EXTRA_ARGS}"
    echo "Stdout logs: ${LOG_ROOT}"
    echo "W&B status cache: ${WANDB_STATUS_FILE}"
}

run_job() {
    local gpu="$1"
    local algo="$2"
    local task="$3"
    local seed="$4"
    local run_index="$5"
    local run_name
    run_name="$(run_name_for "${task}" "${algo}" "${seed}")"
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

    echo "[$(date '+%F %T')] run ${run_index}/${TOTAL_RUNS} GPU ${gpu}: ${run_name}"
    if [[ -n "${DRY_RUN:-}" ]]; then
        local dry_prefix=""
        local dry_cmd=""
        printf -v dry_prefix 'CUDA_VISIBLE_DEVICES=%q EGL_VISIBLE_DEVICES=%q PYTHONPATH=%q WANDB_MODE=%q FLOWRL_ISAACLAB_CLOSE_APP=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ' "${gpu}" "${gpu}" "${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${WANDB_MODE}"
        printf -v dry_cmd '%q ' "${cmd[@]}"
        printf '%s%s\n' "${dry_prefix}" "${dry_cmd}"
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
            echo "algo=${algo}"
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

    for algo in "${ALGOS[@]}"; do
        for task in "${TASKS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if (( idx % ${#GPUS[@]} == worker_id )); then
                    if ! run_job "${gpu}" "${algo}" "${task}" "${seed}" "$((idx + 1))"; then
                        echo "FAILED: run $((idx + 1))/${TOTAL_RUNS} task=${task} algo=${algo} seed=${seed} gpu=${gpu}" >&2
                        failures=$((failures + 1))
                    fi
                fi
                idx=$((idx + 1))
            done
        done
    done

    return "${failures}"
}

if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs configured. Set GPUS=\"0 1 2 3\"." >&2
    exit 1
fi

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
    echo "All final IsaacLab baseline jobs completed."
else
    echo "Some final IsaacLab baseline jobs failed; inspect ${LOG_ROOT}." >&2
fi

if (( status != 0 )); then
    exit "${status}"
fi
exit 0
