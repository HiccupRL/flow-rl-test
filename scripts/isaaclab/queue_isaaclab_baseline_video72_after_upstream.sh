#!/usr/bin/env bash
set -euo pipefail

# Wait for the existing flow-rl-final 12-video job to finish, stop only its
# verified post-run allocator (if present), then start the 8-GPU 200M matrix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

UPSTREAM_ROOT="${UPSTREAM_ROOT:-${REPO_ROOT}/../flow-rl-final/isaaclab_best_return_videos}"
UPSTREAM_REPO="${UPSTREAM_REPO:-${REPO_ROOT}/../flow-rl-final}"
UPSTREAM_LOG_BASE="${UPSTREAM_LOG_BASE:-${UPSTREAM_ROOT}/run_logs/isaaclab_best_return_video12}"
UPSTREAM_EXPECTED="${UPSTREAM_EXPECTED:-12}"
UPSTREAM_LAUNCH_PATTERN="${UPSTREAM_LAUNCH_PATTERN:-scripts/isaaclab/reproduce_and_record_isaaclab_best_return12_8gpu.sh}"
UPSTREAM_JOB_PATTERN="${UPSTREAM_JOB_PATTERN:-scripts/isaaclab/reproduce_isaaclab_best_video_job.py}"
UPSTREAM_TRAIN_PATTERN="${UPSTREAM_TRAIN_PATTERN:-examples/online/train_isaaclab_video_checkpoint.py}"
UPSTREAM_ALLOCATOR_PID="${UPSTREAM_ALLOCATOR_PID:-${UPSTREAM_LOG_BASE}/allocator.pid}"
UPSTREAM_ALLOCATOR_VISIBLE_DEVICES="${UPSTREAM_ALLOCATOR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
UPSTREAM_ROOT="$(realpath "${UPSTREAM_ROOT}")"
UPSTREAM_REPO="$(realpath "${UPSTREAM_REPO}")"
UPSTREAM_LOG_BASE="$(realpath "${UPSTREAM_LOG_BASE}")"

RUNNER="${RUNNER:-scripts/isaaclab/reproduce_and_record_isaaclab_baseline_videos.sh}"
MANIFEST="${MANIFEST:-scripts/isaaclab/search_spaces/isaaclab_baseline_best_final_video72.tsv}"
SOURCE_DIR="${SOURCE_DIR:?SOURCE_DIR must point to the validated smoke-test snapshot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/isaaclab_baseline_final_videos_200m}"
RUN_LABEL="${RUN_LABEL:-isaaclab_baseline_best_final_video72_200m}"
LOG_BASE="${LOG_BASE:-${OUTPUT_ROOT}/run_logs/${RUN_LABEL}}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
EXPECTED="${EXPECTED:-72}"
MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS:-3}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-60}"
VIDEO_STEPS="${VIDEO_STEPS:-1000}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
START_RUNNER="${START_RUNNER:-1}"
HANDOFF_READY_FILE="${HANDOFF_READY_FILE:-}"

log() {
    echo "[$(date '+%F %T')] $*"
}

positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

positive_integer "${UPSTREAM_EXPECTED}" || {
    echo "UPSTREAM_EXPECTED must be a positive integer" >&2
    exit 2
}
positive_integer "${POLL_SECONDS}" || {
    echo "POLL_SECONDS must be a positive integer" >&2
    exit 2
}
[[ "${START_RUNNER}" == "0" || "${START_RUNNER}" == "1" ]] || {
    echo "START_RUNNER must be 0 or 1" >&2
    exit 2
}
if [[ "${START_RUNNER}" == "0" && -z "${HANDOFF_READY_FILE}" ]]; then
    echo "HANDOFF_READY_FILE is required when START_RUNNER=0" >&2
    exit 2
fi
[[ -s "${RUNNER}" && -s "${MANIFEST}" ]] || {
    echo "missing runner or manifest" >&2
    exit 2
}
SOURCE_DIR="$(realpath "${SOURCE_DIR}")"
[[ -f "${SOURCE_DIR}/.baseline_video_source_ready" ]] || {
    echo "invalid frozen SOURCE_DIR: ${SOURCE_DIR}" >&2
    exit 2
}
command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required" >&2
    exit 2
}
[[ -s "${SOURCE_DIR}/SOURCE_FILES.sha256" ]] || {
    echo "missing frozen source checksum list: ${SOURCE_DIR}" >&2
    exit 2
}
(cd "${SOURCE_DIR}" && sha256sum -c SOURCE_FILES.sha256 >/dev/null) || {
    echo "frozen source checksum validation failed: ${SOURCE_DIR}" >&2
    exit 2
}
mkdir -p "${LOG_BASE}"
exec 8>"${LOG_BASE}/.handoff.lock"
flock -n 8 || {
    echo "another handoff process already owns ${LOG_BASE}/.handoff.lock" >&2
    exit 2
}

matching_upstream_processes() {
    local pattern pid args process_cwd
    for pattern in \
        "${UPSTREAM_JOB_PATTERN}" \
        "${UPSTREAM_TRAIN_PATTERN}"; do
        while IFS= read -r pid; do
            [[ "${pid}" =~ ^[0-9]+$ ]] || continue
            args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
            process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
            [[ "${args}" == *"${UPSTREAM_ROOT}"* || \
               "${process_cwd}" == "${UPSTREAM_ROOT}"/* ]] || continue
            printf '%s\n' "${pid}"
        done < <(pgrep -f -- "${pattern}" || true)
    done | sort -u
}

while true; do
    done_count="$(
        find "${UPSTREAM_LOG_BASE}" -maxdepth 1 -type f -name '*.done' \
            -print 2>/dev/null | wc -l
    )"
    running_count="$(
        find "${UPSTREAM_LOG_BASE}" -maxdepth 1 -type f -name '*.running' \
            -print 2>/dev/null | wc -l
    )"
    process_count="$(matching_upstream_processes | wc -l)"
    launcher_count="$(pgrep -f -- "${UPSTREAM_LAUNCH_PATTERN}" | wc -l || true)"
    if (( done_count == UPSTREAM_EXPECTED && running_count == 0 && \
          process_count == 0 && launcher_count == 0 )); then
        log "upstream is complete (${done_count}/${UPSTREAM_EXPECTED})"
        break
    fi
    if (( done_count < UPSTREAM_EXPECTED )) && \
       ! pgrep -f -- "${UPSTREAM_LAUNCH_PATTERN}" >/dev/null 2>&1; then
        echo "upstream launcher exited before completion: done=${done_count}/${UPSTREAM_EXPECTED}" >&2
        exit 1
    fi
    log "waiting for upstream: done=${done_count}/${UPSTREAM_EXPECTED}, running_markers=${running_count}, processes=${process_count}, launchers=${launcher_count}"
    sleep "${POLL_SECONDS}"
done

if [[ -s "${UPSTREAM_ALLOCATOR_PID}" ]]; then
    allocator_pid="$(<"${UPSTREAM_ALLOCATOR_PID}")"
    if [[ "${allocator_pid}" =~ ^[0-9]+$ ]] && \
       kill -0 "${allocator_pid}" 2>/dev/null; then
        allocator_args="$(ps -p "${allocator_pid}" -o args= 2>/dev/null || true)"
        allocator_cwd="$(readlink -f "/proc/${allocator_pid}/cwd" 2>/dev/null || true)"
        allocator_visible_ok=0
        allocator_pattern_ok=0
        if tr '\0' '\n' < "/proc/${allocator_pid}/environ" 2>/dev/null | \
           grep -Fxq "CUDA_VISIBLE_DEVICES=${UPSTREAM_ALLOCATOR_VISIBLE_DEVICES}"; then
            allocator_visible_ok=1
        fi
        if tr '\0' '\n' < "/proc/${allocator_pid}/environ" 2>/dev/null | \
           grep -Fxq "FLOWRL_TRAINING_PROCESS_PATTERN=train_isaaclab_video_checkpoint.py"; then
            allocator_pattern_ok=1
        fi
        if [[ "${allocator_args}" == *"gpu_memory_allocator.py"* && \
              "${allocator_cwd}" == "$(realpath "${UPSTREAM_REPO}")" && \
              "${allocator_visible_ok}" == "1" && "${allocator_pattern_ok}" == "1" ]]; then
            log "stopping verified upstream allocator PID ${allocator_pid}"
            kill -TERM "${allocator_pid}"
            for _ in $(seq 1 30); do
                kill -0 "${allocator_pid}" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "${allocator_pid}" 2>/dev/null; then
                echo "upstream allocator PID ${allocator_pid} did not stop" >&2
                exit 1
            fi
        else
            echo "refusing to stop unverified allocator PID ${allocator_pid}" >&2
            echo "args=${allocator_args}" >&2
            echo "cwd=${allocator_cwd}" >&2
            exit 1
        fi
    fi
fi

if [[ "${START_RUNNER}" == "0" ]]; then
    mkdir -p "$(dirname "${HANDOFF_READY_FILE}")"
    printf 'ready_at=%s\nupstream_done=%s\nallocator_pid=%s\n' \
        "$(date -Iseconds)" "${UPSTREAM_EXPECTED}" "${allocator_pid:-none}" \
        > "${HANDOFF_READY_FILE}"
    log "cleanup-only handoff complete; opened ${HANDOFF_READY_FILE}"
    exit 0
fi

log "starting the 72-cell 200M reproduction on GPUs ${GPUS}"
exec env \
    GPUS="${GPUS}" \
    EXPECTED="${EXPECTED}" \
    MANIFEST="${MANIFEST}" \
    SOURCE_DIR="${SOURCE_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LOG_BASE="${LOG_BASE}" \
    RUN_LABEL="${RUN_LABEL}" \
    TRAIN_FRAMES_OVERRIDE=200000000 \
    MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS}" \
    RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS}" \
    VIDEO_STEPS="${VIDEO_STEPS}" \
    WAIT_FOR_GPU_FREE=1 \
    MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB}" \
    bash "${RUNNER}"
