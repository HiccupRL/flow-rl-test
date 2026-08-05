#!/usr/bin/env bash
set -euo pipefail

# Watch the resumed final IsaacLab baseline launcher and send a minimal Feishu
# alert on actionable failures. Detailed paths, run names, and log tails stay in
# the local monitor log; Feishu receives only a generic heads-up.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-run_logs/isaaclab_baseline_final_remaining_12task_4gpu_10seed_200m}"
LAUNCHER_PID_FILE="${LAUNCHER_PID_FILE:-${LOG_ROOT}/launcher.pid}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${LOG_ROOT}/launcher.nohup.log}"
ATTEMPT_FILE="${ATTEMPT_FILE:-${LOG_ROOT}/wandb_runs_to_attempt.tsv}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STALE_LOG_SECONDS="${STALE_LOG_SECONDS:-7200}"

ALERT_STATE_DIR="${ALERT_STATE_DIR:-${LOG_ROOT}/.alert_monitor}"
ALERT_LOG="${ALERT_LOG:-${ALERT_STATE_DIR}/monitor.log}"
LARK_ALERT_AS="${LARK_ALERT_AS:-bot}"
LARK_ALERT_FALLBACK_AS="${LARK_ALERT_FALLBACK_AS:-user}"
LARK_ALERT_DRY_RUN="${LARK_ALERT_DRY_RUN:-0}"
LARK_ALERT_TEXT="${LARK_ALERT_TEXT:-IsaacLab baseline 实验监控检测到异常，请回到实验机器查看本地监控日志。}"
LARK_COMPLETION_TEXT="${LARK_COMPLETION_TEXT:-IsaacLab baseline 实验已全部结束，请查看 W&B/project 结果。}"

if [[ -z "${FATAL_PATTERN:-}" ]]; then
    FATAL_PATTERN='Traceback|wandb\.errors|CommError|CUDA.*out of memory|out of memory|OOM|No space left on device|Run initialization has timed out|timeout: the monitored command|SIGKILL|Killed|exit_code=([1-9][0-9]*|[0-9][0-9][0-9]+)|Post-run validation failed|W&B resume query failed|Some final IsaacLab baseline completion jobs failed'
fi

mkdir -p "${ALERT_STATE_DIR}"

log_monitor() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"${ALERT_LOG}"
}

sanitize_key() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-120
}

resolve_lark_user_id() {
    if [[ -n "${LARK_ALERT_USER_ID:-}" ]]; then
        printf '%s\n' "${LARK_ALERT_USER_ID}"
        return 0
    fi

    if command -v lark-cli >/dev/null 2>&1; then
        local status_output
        status_output="$(lark-cli auth status 2>/dev/null || true)"
        printf '%s\n' "${status_output}" | sed -n 's/.*"openId": "\(ou_[^"]*\)".*/\1/p' | head -n 1
    fi
}

send_lark_alert_once() {
    local key="$1"
    local local_reason="$2"
    local message_text="${3:-${LARK_ALERT_TEXT}}"
    local sent_file="${ALERT_STATE_DIR}/sent-$(sanitize_key "${key}")"

    log_monitor "alert candidate: key=${key}; reason=${local_reason}"
    if [[ -f "${sent_file}" ]]; then
        return 0
    fi

    local user_id
    user_id="$(resolve_lark_user_id || true)"
    if [[ -z "${user_id}" ]]; then
        log_monitor "alert not sent; could not resolve LARK_ALERT_USER_ID"
        return 1
    fi

    if ! command -v lark-cli >/dev/null 2>&1; then
        log_monitor "alert not sent; lark-cli not found"
        return 1
    fi

    local text
    text="${message_text} UTC: $(date -u '+%F %T')"

    if [[ "${LARK_ALERT_DRY_RUN}" == "1" ]]; then
        printf '%s\n' "${text}" >"${ALERT_STATE_DIR}/dry-run-$(sanitize_key "${key}").txt"
        touch "${sent_file}"
        return 0
    fi

    local output
    if output="$(lark-cli im +messages-send --as "${LARK_ALERT_AS}" --user-id "${user_id}" --text "${text}" 2>&1)"; then
        log_monitor "Feishu alert sent with --as ${LARK_ALERT_AS}"
        printf '%s\n' "${output}" >>"${ALERT_LOG}"
        touch "${sent_file}"
        return 0
    fi

    log_monitor "primary Feishu alert failed with --as ${LARK_ALERT_AS}: ${output}"
    if [[ "${LARK_ALERT_FALLBACK_AS}" != "${LARK_ALERT_AS}" ]]; then
        if output="$(lark-cli im +messages-send --as "${LARK_ALERT_FALLBACK_AS}" --user-id "${user_id}" --text "${text}" 2>&1)"; then
            log_monitor "Feishu alert sent with fallback --as ${LARK_ALERT_FALLBACK_AS}"
            printf '%s\n' "${output}" >>"${ALERT_LOG}"
            touch "${sent_file}"
            return 0
        fi
        log_monitor "fallback Feishu alert failed with --as ${LARK_ALERT_FALLBACK_AS}: ${output}"
    fi

    return 1
}

attempt_count() {
    if [[ ! -f "${ATTEMPT_FILE}" ]]; then
        printf '0\n'
        return 0
    fi
    local lines
    lines="$(wc -l <"${ATTEMPT_FILE}")"
    if (( lines <= 0 )); then
        printf '0\n'
    else
        printf '%s\n' "$((lines - 1))"
    fi
}

marker_count() {
    local suffix="$1"
    find "${LOG_ROOT}" -maxdepth 1 -name "*.${suffix}" 2>/dev/null | wc -l
}

launcher_alive() {
    [[ -f "${LAUNCHER_PID_FILE}" ]] || return 1
    local pid
    pid="$(tr -dc '0-9' <"${LAUNCHER_PID_FILE}" || true)"
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" >/dev/null 2>&1
}

tail_for_local_log() {
    local file="$1"
    local lines="${2:-30}"
    if [[ -f "${file}" ]]; then
        {
            printf '--- %s tail ---\n' "${file}"
            tr '\r' '\n' <"${file}" | tail -n "${lines}"
            printf '%s\n' '--- end tail ---'
        } >>"${ALERT_LOG}"
    fi
}

check_failed_markers() {
    local failed_file
    while IFS= read -r failed_file; do
        [[ -n "${failed_file}" ]] || continue
        local run_name log_file
        run_name="$(basename "${failed_file}" .failed)"
        log_file="$(sed -n 's/^log_file=//p' "${failed_file}" | tail -n 1)"
        log_monitor "failed marker detected: ${failed_file}"
        cat "${failed_file}" >>"${ALERT_LOG}" || true
        tail_for_local_log "${log_file}" 40
        send_lark_alert_once "failed-marker-${run_name}" "failed marker detected" || true
    done < <(find "${LOG_ROOT}" -maxdepth 1 -name '*.failed' 2>/dev/null | sort)
}

check_fatal_log_patterns() {
    local log_file
    while IFS= read -r log_file; do
        [[ -n "${log_file}" ]] || continue
        local base run_name matches
        base="$(basename "${log_file}")"
        run_name="${base%.gpu*.log}"
        [[ ! -f "${LOG_ROOT}/${run_name}.done" ]] || continue
        [[ ! -f "${LOG_ROOT}/${run_name}.failed" ]] || continue

        if command -v rg >/dev/null 2>&1; then
            matches="$(tr '\r' '\n' <"${log_file}" | rg -n -i "${FATAL_PATTERN}" | tail -n 12 || true)"
        else
            matches="$(tr '\r' '\n' <"${log_file}" | grep -Ein "${FATAL_PATTERN}" | tail -n 12 || true)"
        fi

        if [[ -n "${matches}" ]]; then
            log_monitor "fatal pattern detected in ${log_file}"
            printf '%s\n' "${matches}" >>"${ALERT_LOG}"
            tail_for_local_log "${log_file}" 40
            send_lark_alert_once "fatal-log-${run_name}" "fatal log pattern detected" || true
        fi
    done < <(find "${LOG_ROOT}" -maxdepth 1 -name '*.gpu*.log' 2>/dev/null | sort)
}

check_stale_running_logs() {
    local now
    now="$(date +%s)"
    local log_file
    while IFS= read -r log_file; do
        [[ -n "${log_file}" ]] || continue
        local base run_name mtime age
        base="$(basename "${log_file}")"
        run_name="${base%.gpu*.log}"
        [[ ! -f "${LOG_ROOT}/${run_name}.done" ]] || continue
        [[ ! -f "${LOG_ROOT}/${run_name}.failed" ]] || continue

        mtime="$(stat -c '%Y' "${log_file}" 2>/dev/null || printf '0')"
        age=$((now - mtime))
        if (( age >= STALE_LOG_SECONDS )); then
            log_monitor "stale running log detected: ${log_file}; age=${age}; threshold=${STALE_LOG_SECONDS}"
            tail_for_local_log "${log_file}" 40
            send_lark_alert_once "stale-log-${run_name}" "running log is stale" || true
        fi
    done < <(find "${LOG_ROOT}" -maxdepth 1 -name '*.gpu*.log' 2>/dev/null | sort)
}

check_launcher_exit() {
    local total done failed marked
    total="$(attempt_count)"
    done="$(marker_count done)"
    failed="$(marker_count failed)"
    marked=$((done + failed))

    if (( total > 0 && marked >= total )); then
        local completion_text="${LARK_COMPLETION_TEXT}"
        if (( failed > 0 )); then
            completion_text="IsaacLab baseline 实验队列已结束，但存在失败标记，请回实验机器查看本地监控日志。"
        fi
        send_lark_alert_once "all-scheduled-runs-marked" "all scheduled runs marked" "${completion_text}" || true
        log_monitor "all scheduled runs are marked; total=${total} done=${done} failed=${failed}; exiting monitor"
        exit 0
    fi

    if launcher_alive; then
        return 0
    fi

    log_monitor "launcher is not alive before completion; total=${total}; done=${done}; failed=${failed}; marked=${marked}"
    tail_for_local_log "${LAUNCHER_LOG}" 80
    send_lark_alert_once "launcher-exited-before-complete" "launcher stopped before all scheduled runs completed" || true
}

log_monitor "monitor started; log_root=${LOG_ROOT}; poll_seconds=${POLL_SECONDS}; stale_log_seconds=${STALE_LOG_SECONDS}"

while true; do
    if [[ ! -d "${LOG_ROOT}" ]]; then
        log_monitor "missing log root: ${LOG_ROOT}"
        send_lark_alert_once "missing-log-root" "log root is missing" || true
    else
        check_failed_markers
        check_fatal_log_patterns
        check_stale_running_logs
        check_launcher_exit
    fi
    sleep "${POLL_SECONDS}"
done
