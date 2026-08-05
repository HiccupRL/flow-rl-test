#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_LOG_ROOT="${REPO_ROOT}/run_logs/dmc-onpolicy-baseline-29982720f-10seed-seedmajor-1024env-16epoch-32way-v1"
FROZEN_LAUNCHER="${RUN_LOG_ROOT}/frozen_source/scripts/dmc/dmc-onpolicy-baseline-10seed.sh"
LAUNCH_LOG="${RUN_LOG_ROOT}/launcher.log"
PID_FILE="${RUN_LOG_ROOT}/launcher.pid"
EXPECTED_LAUNCH_FINGERPRINT="f0dd3823729543530d651b7f7db203dddfb7d697302f99e25ce8faa3afd7c24a"
EXPECTED_GIT_COMMIT="6ca3ecae50d3e0e52cff5fdcef57552c7402d927"

ALGOS=(genpo dppo fpopp fpo policyflow ppo)
TASKS=(
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
)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

RUN_ENV=(
    "GPUS=0 1 2 3 4 5 6 7"
    "RUNS_PER_GPU=4"
    "ALGOS=${ALGOS[*]}"
    "TASKS=${TASKS[*]}"
    "SEEDS=${SEEDS[*]}"
    "TRAIN_FRAMES=29982720"
    "EVAL_FRAMES=98304"
    "LOG_FRAMES=49152"
    "FRAME_SKIP=2"
    "FRAME_STACK=1"
    "HORIZON=1000"
    "ENV_MODE=sync"
    "DEBUG_FINITE_CHECKS=true"
    "SEED_BARRIER=1"
    "BUDGET_TAG=30m"
    "SCHEDULE_TAG=seed-major"
    "SEED_PROTOCOL_TAG=independent-seed-blocks-v1"
    "EVAL_RNG_TAG=isolated-fixed-eval-rng-v1"
    "POLICYFLOW_TIMEOUT_TAG=timeout-bootstrap-official-current-value"
    "LOG_DIR=${REPO_ROOT}/logs"
    "LOG_TAG=dmc-onpolicy-29982720f-10seed-seedmajor-1024env-16epoch-v1"
    "LOG_GROUP=dmc-onpolicy-baseline-30m-10seed-seedmajor-1024env-16epoch-v1"
    "RUN_LOG_ROOT=${RUN_LOG_ROOT}"
    "WANDB_PROJECT_NAME=dmc-onpolicy-baseline-30m"
    "WANDB_ENTITY=hiccupnudt"
    "WANDB_MODE=online"
    "WANDB_ENABLED=1"
    "MAX_ATTEMPTS=2"
    "RETRY_DELAY_SECONDS=30"
)

marker_count() {
    local suffix="$1"
    find "${RUN_LOG_ROOT}" -maxdepth 1 -type f -name "*.${suffix}" 2>/dev/null |
        wc -l
}

seed_marker_count() {
    local seed="$1"
    local suffix="$2"
    find "${RUN_LOG_ROOT}" -maxdepth 1 -type f -name "*-seed${seed}.${suffix}" 2>/dev/null |
        wc -l
}

launcher_pid() {
    if [[ -s "${PID_FILE}" ]]; then
        sed -n '1p' "${PID_FILE}"
    fi
}

launcher_is_active() {
    local pid cmdline
    pid="$(launcher_pid)"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    [[ "${cmdline}" == *"dmc-onpolicy-baseline-10seed.sh"* ]]
}

live_running_marker_count() {
    local seed_pattern="${1:-*}"
    local marker pid count=0
    shopt -s nullglob
    for marker in "${RUN_LOG_ROOT}"/*-seed${seed_pattern}.running; do
        pid="$(awk -F= '$1 == "worker_pid" {print $2}' "${marker}")"
        if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    shopt -u nullglob
    echo "${count}"
}

print_status() {
    local seed running_markers live_running stale_running seed_running seed_live
    running_markers="$(marker_count running)"
    live_running="$(live_running_marker_count)"
    stale_running=$((running_markers - live_running))
    printf 'total=600 done=%s running_markers=%s live_running=%s stale_running=%s failed=%s attempts=%s\n' \
        "$(marker_count done)" \
        "${running_markers}" \
        "${live_running}" \
        "${stale_running}" \
        "$(marker_count failed)" \
        "$(marker_count attempt)"
    if launcher_is_active; then
        printf 'launcher=running pid=%s\n' "$(launcher_pid)"
    else
        printf 'launcher=stopped\n'
    fi
    for seed in "${SEEDS[@]}"; do
        seed_running="$(seed_marker_count "${seed}" running)"
        seed_live="$(live_running_marker_count "${seed}")"
        printf 'seed=%s done=%s running_markers=%s live_running=%s stale_running=%s failed=%s\n' \
            "${seed}" \
            "$(seed_marker_count "${seed}" done)" \
            "${seed_running}" \
            "${seed_live}" \
            "$((seed_running - seed_live))" \
            "$(seed_marker_count "${seed}" failed)"
    done
}

check_resume_preconditions() {
    local stored_fingerprint current_commit seed0_done
    [[ -d "${RUN_LOG_ROOT}" ]] || {
        echo "Missing run root: ${RUN_LOG_ROOT}" >&2
        exit 1
    }
    [[ -f "${FROZEN_LAUNCHER}" ]] || {
        echo "Missing frozen launcher: ${FROZEN_LAUNCHER}" >&2
        exit 1
    }
    command -v python3 >/dev/null
    command -v nvidia-smi >/dev/null
    command -v flock >/dev/null

    stored_fingerprint="$(
        awk -F= '$1 == "launch" {print $2}' \
            "${RUN_LOG_ROOT}/launch-fingerprint.txt"
    )"
    if [[ "${stored_fingerprint}" != "${EXPECTED_LAUNCH_FINGERPRINT}" ]]; then
        echo "Unexpected launch fingerprint: ${stored_fingerprint}" >&2
        exit 1
    fi

    current_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    if [[ "${current_commit}" != "${EXPECTED_GIT_COMMIT}" ]]; then
        echo "Git HEAD changed since the formal sweep was frozen." >&2
        echo "expected=${EXPECTED_GIT_COMMIT}" >&2
        echo "current=${current_commit}" >&2
        exit 1
    fi

    seed0_done="$(seed_marker_count 0 done)"
    if [[ "${seed0_done}" != "60" ]]; then
        echo "Refusing resume: seed0 has ${seed0_done}/60 successful runs." >&2
        exit 1
    fi

    if launcher_is_active; then
        echo "A DMC launcher is already active (pid=$(launcher_pid))." >&2
        exit 1
    fi

    exec 8>"${RUN_LOG_ROOT}/launcher.lock"
    if ! flock -n 8; then
        echo "The launcher lock is still held: ${RUN_LOG_ROOT}/launcher.lock" >&2
        exit 1
    fi
    flock -u 8
    exec 8>&-
}

run_foreground() {
    printf '%s\n' "$$" >"${PID_FILE}"
    exec env "${RUN_ENV[@]}" bash "${FROZEN_LAUNCHER}"
}

run_detached() {
    local previous_pid pid stamp
    previous_pid="$(launcher_pid)"
    stamp="$(date '+%Y%m%d-%H%M%S')"
    if [[ -s "${PID_FILE}" ]]; then
        mv "${PID_FILE}" "${PID_FILE}.stopped-${stamp}"
    fi

    printf '[%s] resume requested; previous_pid=%s\n' \
        "$(date '+%F %T')" "${previous_pid:-unknown}" >>"${LAUNCH_LOG}"

    setsid -f env \
        "${RUN_ENV[@]}" \
        "RESUME_PID_FILE=${PID_FILE}" \
        "FROZEN_LAUNCHER=${FROZEN_LAUNCHER}" \
        bash -c '
            printf "%s\n" "$$" >"${RESUME_PID_FILE}"
            exec bash "${FROZEN_LAUNCHER}"
        ' >>"${LAUNCH_LOG}" 2>&1 </dev/null

    for _ in $(seq 1 10); do
        if launcher_is_active; then
            pid="$(launcher_pid)"
            echo "DMC 30M resume launcher started: pid=${pid}"
            echo "Log: ${LAUNCH_LOG}"
            echo "W&B: https://wandb.ai/hiccupnudt/dmc-onpolicy-baseline-30m"
            return 0
        fi
        sleep 1
    done

    echo "Resume launcher did not stay alive. Recent log:" >&2
    tail -n 80 "${LAUNCH_LOG}" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [--check|--status|--foreground]

Without arguments, resume seeds 1-9 in a detached session.
The frozen launcher keeps SEEDS=0..9 for fingerprint compatibility and
automatically skips the 60 completed seed0 runs.
EOF
}

case "${1:-}" in
    "")
        check_resume_preconditions
        run_detached
        ;;
    --foreground)
        check_resume_preconditions
        run_foreground
        ;;
    --status)
        print_status
        ;;
    --check)
        check_resume_preconditions
        echo "Resume preflight: OK"
        ;;
    --help|-h)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
