#!/usr/bin/env bash
set -euo pipefail

# Keep the long-running sweep attached by default. Managed compute shells often
# reap nohup/setsid descendants as soon as the invoking command exits, so an
# apparently successful detached launch can disappear during Python preflight.
# Set DETACH=1 only in a persistent host shell; detached mode waits for a READY
# marker written after the scheduler has launched its first GPU jobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${START_TARGET:-${SCRIPT_DIR}/run-all-tasks-smoothness-8gpu.sh}"
LOG_ROOT="${MASTER_LOG_ROOT:-${REPO_ROOT}/run_logs/isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${LOG_ROOT}/launcher.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/launcher.pid}"
READY_FILE="${READY_FILE:-${LOG_ROOT}/launcher.ready}"
DETACH="${DETACH:-0}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-180}"

mkdir -p "${LOG_ROOT}"

if [[ "${DETACH}" != "0" && "${DETACH}" != "1" ]]; then
    echo "DETACH must be 0 or 1, got: ${DETACH}" >&2
    exit 2
fi
if ! [[ "${STARTUP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STARTUP_TIMEOUT_SECONDS must be a positive integer, got: ${STARTUP_TIMEOUT_SECONDS}" >&2
    exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
    read -r old_pid <"${PID_FILE}" || old_pid=""
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        old_cmd="$(tr '\0' ' ' <"/proc/${old_pid}/cmdline" 2>/dev/null || true)"
        if [[ "${old_cmd}" == *"run-all-tasks-smoothness-8gpu.sh"* ]]; then
            echo "Sweep launcher is already running: pid=${old_pid}" >&2
            exit 2
        fi
    fi
fi

cd "${REPO_ROOT}"
rm -f "${READY_FILE}"

if [[ "${DETACH}" == "0" ]]; then
    {
        echo
        echo "[$(date --iso-8601=seconds)] foreground launch requested"
    } >>"${LAUNCHER_LOG}"
    echo "Starting supervised 8-GPU sweep in the foreground."
    echo "Keep this command running; Ctrl-C checkpoints and stops active jobs."
    echo "Launcher log: ${LAUNCHER_LOG}"

    READY_FILE="${READY_FILE}" bash "${TARGET}" > >(tee -a "${LAUNCHER_LOG}") 2>&1 &
    launcher_pid=$!
    echo "${launcher_pid}" >"${PID_FILE}"

    forward_signal() {
        local signal="$1"
        kill -"${signal}" "${launcher_pid}" 2>/dev/null || true
    }
    trap 'forward_signal INT' INT
    trap 'forward_signal TERM' TERM

    if wait "${launcher_pid}"; then
        status=0
    else
        status=$?
    fi
    trap - INT TERM
    rm -f "${PID_FILE}"
    echo "Sweep launcher exited with status ${status}. Log: ${LAUNCHER_LOG}"
    exit "${status}"
fi

if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required for DETACH=1." >&2
    exit 1
fi

{
    echo
    echo "[$(date --iso-8601=seconds)] detached launch requested"
} >>"${LAUNCHER_LOG}"

READY_FILE="${READY_FILE}" nohup setsid bash "${TARGET}" >>"${LAUNCHER_LOG}" 2>&1 </dev/null &
launcher_pid=$!
echo "${launcher_pid}" >"${PID_FILE}"

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
    if [[ -s "${READY_FILE}" ]]; then
        if kill -0 "${launcher_pid}" 2>/dev/null; then
            echo "Started detached 8-GPU sweep: pid=${launcher_pid}"
            echo "Readiness: $(tr '\n' ' ' <"${READY_FILE}")"
            echo "Launcher log: ${LAUNCHER_LOG}"
            exit 0
        fi
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
        echo "Launcher exited before READY; recent log output:" >&2
        tail -n 60 "${LAUNCHER_LOG}" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "Launcher did not become READY within ${STARTUP_TIMEOUT_SECONDS}s; it may be stuck in preflight." >&2
echo "Inspect ${LAUNCHER_LOG}; no startup success is being reported." >&2
exit 3
