#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/isaaclab_baseline_final_videos_200m}"
LOG_BASE="${LOG_BASE:-${OUTPUT_ROOT}/run_logs/isaaclab_baseline_best_final_video72_200m}"
RUNNER="${RUNNER:-scripts/isaaclab/reproduce_and_record_isaaclab_baseline_videos.sh}"
HANDOFF="${HANDOFF:-scripts/isaaclab/queue_isaaclab_baseline_video72_after_upstream.sh}"
MANIFEST="${MANIFEST:-scripts/isaaclab/search_spaces/isaaclab_baseline_best_final_video72.tsv}"
SOURCE_DIR="${SOURCE_DIR:-${REPO_ROOT}/artifacts/isaaclab_baseline_video_source_final/source_snapshot/baseline_6ca3ecae50d3_overlay_20260730T162247Z}"
RUN_LABEL="${RUN_LABEL:-isaaclab_baseline_best_final_video72_200m}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
EXPECTED="${EXPECTED:-72}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
IDLE_USED_MEM_MB="${IDLE_USED_MEM_MB:-1000}"
DEFERRED_GPU="${DEFERRED_GPU:-4}"
GPU4_READY_FILE="${GPU4_READY_FILE:-${LOG_BASE}/gpu4.ready}"
RUNNER_PID_FILE="${RUNNER_PID_FILE:-${LOG_BASE}/runner.pid}"
HANDOFF_PID_FILE="${HANDOFF_PID_FILE:-${LOG_BASE}/handoff.pid}"
MONITOR_PID_FILE="${MONITOR_PID_FILE:-${LOG_BASE}/utilization_monitor.pid}"
MONITOR_LOG="${MONITOR_LOG:-${LOG_BASE}/utilization_monitor.log}"
MONITOR_LOCK="${MONITOR_LOCK:-${LOG_BASE}/.utilization_monitor.lock}"
RUNNER_LOCK="${RUNNER_LOCK:-${LOG_BASE}/.runner.lock}"
HANDOFF_LOCK="${HANDOFF_LOCK:-${LOG_BASE}/.handoff.lock}"

mkdir -p "${LOG_BASE}"
exec 9>"${MONITOR_LOCK}"
flock -n 9 || { echo "another utilization monitor is active" >&2; exit 2; }
printf '%s\n' "$$" > "${MONITOR_PID_FILE}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MONITOR_LOG}"
}

positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

positive_integer "${EXPECTED}" || { log "invalid EXPECTED=${EXPECTED}"; exit 2; }
positive_integer "${POLL_SECONDS}" || { log "invalid POLL_SECONDS=${POLL_SECONDS}"; exit 2; }
positive_integer "${IDLE_USED_MEM_MB}" || { log "invalid IDLE_USED_MEM_MB=${IDLE_USED_MEM_MB}"; exit 2; }
[[ -s "${RUNNER}" && -s "${HANDOFF}" && -s "${MANIFEST}" ]] || {
    log "runner, handoff, or manifest is missing"
    exit 2
}
[[ -f "${SOURCE_DIR}/.baseline_video_source_ready" ]] || {
    log "invalid SOURCE_DIR=${SOURCE_DIR}"
    exit 2
}

controller_active() {
    local pid_file="$1" pattern="$2" pid args
    [[ -s "${pid_file}" ]] || return 1
    pid="$(tr -d '[:space:]' < "${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    args="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${args}" == *"${pattern}"* ]]
}

formal_job_count() {
    local pid args count=0
    while IFS= read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        args="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "${args}" == *"${OUTPUT_ROOT}"* ]] || continue
        count=$((count + 1))
    done < <(pgrep -f -- 'reproduce_isaaclab_baseline_video_job.py' || true)
    printf '%s\n' "${count}"
}

count_markers() {
    local suffix="$1"
    find "${LOG_BASE}" -maxdepth 1 -type f -name "*.${suffix}" -print 2>/dev/null | wc -l
}

start_runner() {
    nohup setsid env \
        GPUS="${GPUS}" EXPECTED="${EXPECTED}" MANIFEST="${MANIFEST}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_BASE="${LOG_BASE}" RUN_LABEL="${RUN_LABEL}" \
        SOURCE_DIR="${SOURCE_DIR}" TRAIN_FRAMES_OVERRIDE=200000000 \
        WAIT_FOR_GPU_FREE=1 MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB}" GPU_POLL_SECONDS=30 \
        DEFERRED_GPU="${DEFERRED_GPU}" DEFERRED_GPU_READY_FILE="${GPU4_READY_FILE}" \
        MAX_JOB_ATTEMPTS=3 RETRY_BACKOFF_SECONDS=60 VIDEO_STEPS=1000 \
        bash "${RUNNER}" >> "${LOG_BASE}/runner.log" 2>&1 < /dev/null &
    local pid="$!"
    printf '%s\n' "${pid}" > "${RUNNER_PID_FILE}"
    log "restarted formal runner pid=${pid}"
}

start_handoff() {
    nohup setsid env \
        SOURCE_DIR="${SOURCE_DIR}" MANIFEST="${MANIFEST}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_BASE="${LOG_BASE}" RUN_LABEL="${RUN_LABEL}" \
        GPUS="${GPUS}" EXPECTED="${EXPECTED}" POLL_SECONDS=30 \
        START_RUNNER=0 HANDOFF_READY_FILE="${GPU4_READY_FILE}" \
        bash "${HANDOFF}" >> "${LOG_BASE}/handoff.log" 2>&1 < /dev/null &
    local pid="$!"
    printf '%s\n' "${pid}" > "${HANDOFF_PID_FILE}"
    log "restarted cleanup-only handoff pid=${pid}"
}

ensure_controllers() {
    local done_count jobs
    done_count="$(count_markers done)"
    if (( done_count < EXPECTED )) && ! controller_active "${RUNNER_PID_FILE}" "${RUNNER}"; then
        jobs="$(formal_job_count)"
        if (( jobs == 0 )); then
            if flock -n "${RUNNER_LOCK}" true; then
                start_runner
            else
                log "runner PID is stale but runner lock is held; leaving it untouched"
            fi
        else
            log "runner PID is stale while ${jobs} formal jobs remain; waiting without duplicate restart"
        fi
    fi
    if [[ ! -f "${GPU4_READY_FILE}" ]] && \
       ! controller_active "${HANDOFF_PID_FILE}" "${HANDOFF}"; then
        if flock -n "${HANDOFF_LOCK}" true; then
            start_handoff
        else
            log "handoff PID is stale but handoff lock is held; leaving it untouched"
        fi
    fi
}

gpu_has_assigned_job() {
    local gpu="$1" pid_file pid args
    for pid_file in "${LOG_BASE}"/*.pid; do
        [[ -e "${pid_file}" ]] || continue
        case "${pid_file}" in
            "${RUNNER_PID_FILE}"|"${HANDOFF_PID_FILE}"|"${MONITOR_PID_FILE}") continue ;;
        esac
        pid="$(tr -d '[:space:]' < "${pid_file}")"
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        kill -0 "${pid}" 2>/dev/null || continue
        args="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "${args}" == *"--gpu ${gpu} "* ]] && return 0
    done
    return 1
}

snapshot() {
    local done_count running_count jobs queue_next runner_state handoff_state gpu used util
    done_count="$(count_markers done)"
    running_count="$(count_markers running)"
    jobs="$(formal_job_count)"
    queue_next="unknown"
    [[ -s "${LOG_BASE}/.queue_next" ]] && queue_next="$(tr -d '[:space:]' < "${LOG_BASE}/.queue_next")"
    runner_state="dead"
    handoff_state="dead"
    controller_active "${RUNNER_PID_FILE}" "${RUNNER}" && runner_state="alive"
    controller_active "${HANDOFF_PID_FILE}" "${HANDOFF}" && handoff_state="alive"
    log "matrix done=${done_count}/${EXPECTED} running=${running_count} jobs=${jobs} queue_next=${queue_next} runner=${runner_state} handoff=${handoff_state} gpu4_ready=$([[ -f "${GPU4_READY_FILE}" ]] && echo yes || echo no)"
    while read -r gpu used util; do
        log "gpu=${gpu} used_mib=${used} utilization_pct=${util}"
        if (( done_count < EXPECTED && used < IDLE_USED_MEM_MB )); then
            if gpu_has_assigned_job "${gpu}"; then
                log "NOTICE gpu=${gpu} is low-memory but has a live assigned job (initialization/eval/render)"
            elif [[ "${gpu}" == "${DEFERRED_GPU}" && ! -f "${GPU4_READY_FILE}" ]]; then
                log "NOTICE gpu=${gpu} is intentionally deferred until upstream cleanup"
            else
                log "WARNING gpu=${gpu} appears idle while work remains"
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
}

trap 'log "monitor stopping"; exit 0' INT TERM HUP
log "utilization monitor started; poll_seconds=${POLL_SECONDS}"
while true; do
    ensure_controllers
    snapshot
    sleep "${POLL_SECONDS}"
done
