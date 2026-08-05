#!/usr/bin/env bash
set -euo pipefail

# Monitor the 288-run IsaacLab smoothness sweep through shared log/marker files.
# A full Feishu report is sent every eight hours. Failures, a stalled queue, and
# final completion are reported immediately. This intentionally does not depend
# on process visibility because the launcher and monitor may run in different
# containers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT="${ROOT:-run_logs/isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs}"
EXPECTED_RUNS="${EXPECTED_RUNS:-288}"
EXPECTED_ACTIVE="${EXPECTED_ACTIVE:-8}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-28800}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ACTIVE_LOG_SECONDS="${ACTIVE_LOG_SECONDS:-1800}"
STATE_DIR="${STATE_DIR:-${ROOT}/feishu-8h-monitor-state}"
MONITOR_LOG="${MONITOR_LOG:-${ROOT}/feishu-8h-monitor.log}"
PID_FILE="${PID_FILE:-${STATE_DIR}/monitor.pid}"
DRY_RUN="${DRY_RUN:-0}"
ONCE="${ONCE:-0}"
LARK_NOTIFY_AS="${LARK_NOTIFY_AS:-bot}"
LARK_NOTIFY_USER_ID="${LARK_NOTIFY_USER_ID:-}"
LARK_SOURCE_HOME="${LARK_SOURCE_HOME:-${HOME}}"
LARK_RUNTIME_HOME="${LARK_RUNTIME_HOME:-/tmp/flowrl-lark-runtime-v2}"

LAST_REPORT_EPOCH_FILE="${STATE_DIR}/last_report_epoch"
LAST_REPORT_DONE_FILE="${STATE_DIR}/last_report_done"
LAST_FAILED_FILE="${STATE_DIR}/last_failed_count"
STARTUP_SENT_FILE="${STATE_DIR}/startup_sent"
STALLED_SENT_FILE="${STATE_DIR}/stalled_sent"
FINAL_SENT_FILE="${STATE_DIR}/final_sent"

mkdir -p "${ROOT}" "${STATE_DIR}"

prepare_lark_runtime() {
    # The mounted Codex home is read-only, while lark-cli needs writable lock,
    # cache, and refreshed-token locations. Copy only the CLI auth material to
    # a private tmpfs home; never place credentials under the repository.
    [[ "${LARK_RUNTIME_HOME}" != "${LARK_SOURCE_HOME}" ]] || return 0
    umask 077
    mkdir -p "${LARK_RUNTIME_HOME}/.lark-cli" \
        "${LARK_RUNTIME_HOME}/.local/share/lark-cli"
    if [[ ! -f "${LARK_RUNTIME_HOME}/.lark-cli/config.json" ]] \
        && [[ -d "${LARK_SOURCE_HOME}/.lark-cli" ]]; then
        cp -a "${LARK_SOURCE_HOME}/.lark-cli/." \
            "${LARK_RUNTIME_HOME}/.lark-cli/"
    fi
    if [[ ! -f "${LARK_RUNTIME_HOME}/.local/share/lark-cli/master.key" ]] \
        && [[ -d "${LARK_SOURCE_HOME}/.local/share/lark-cli" ]]; then
        cp -a "${LARK_SOURCE_HOME}/.local/share/lark-cli/." \
            "${LARK_RUNTIME_HOME}/.local/share/lark-cli/"
    fi
    export HOME="${LARK_RUNTIME_HOME}"
}

prepare_lark_runtime

log() {
    printf '[%s] %s\n' "$(date -u '+%F %T UTC')" "$*" >>"${MONITOR_LOG}"
}

read_nonnegative() {
    local file="$1"
    local fallback="${2:-0}"
    local value
    value="$(sed -n '1p' "${file}" 2>/dev/null || true)"
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s\n' "${fallback}"
    fi
}

count_markers() {
    local suffix="$1"
    find "${ROOT}" -type f -name "*.${suffix}" 2>/dev/null | wc -l | tr -d ' '
}

active_log_files() {
    local now file run_base mtime age
    now="$(date +%s)"
    while IFS= read -r file; do
        run_base="${file%.gpu*.log}"
        [[ ! -f "${run_base}.done" && ! -f "${run_base}.failed" ]] || continue
        mtime="$(stat -c '%Y' "${file}" 2>/dev/null || printf '0')"
        age=$((now - mtime))
        (( age <= ACTIVE_LOG_SECONDS )) || continue
        printf '%s\n' "${file}"
    done < <(find "${ROOT}" -type f -name '*.gpu[0-7].log' 2>/dev/null | sort)
}

active_count() {
    active_log_files | awk 'NF {count++} END {print count + 0}'
}

active_algo_summary() {
    local files algo count out=""
    files="$(active_log_files)"
    for algo in ppo dppo fpo fpopp genpo policyflow; do
        count="$(printf '%s\n' "${files}" | grep -c -- "-${algo}-seed" || true)"
        (( count > 0 )) || continue
        [[ -z "${out}" ]] || out="${out}, "
        out="${out}${algo}=${count}"
    done
    printf '%s\n' "${out:-none}"
}

active_task_summary() {
    active_log_files | while IFS= read -r file; do
        basename "$(dirname "${file}")"
    done | sort | uniq -c | awk '{printf "%s%s=%s", sep, $2, $1; sep=", "} END {print ""}'
}

done_algo_summary() {
    local algo count out=""
    for algo in ppo dppo fpo fpopp genpo policyflow; do
        count="$(find "${ROOT}" -type f -name "*-${algo}-seed*.done" 2>/dev/null | wc -l | tr -d ' ')"
        [[ -z "${out}" ]] || out="${out}, "
        out="${out}${algo}=${count}"
    done
    printf '%s\n' "${out}"
}

latest_checkpoint_epoch() {
    find "${CHECKPOINT_ROOT}" \
        -path '*isaaclab-all-tasks-smoothness-200m-4seed-8gpu-clean-v1*' \
        -type f -name 'checkpoint.msgpack' -printf '%T@\n' 2>/dev/null \
        | sort -nr | sed -n '1{s/\..*//;p;}'
}

last_scheduler_event() {
    if [[ ! -f "${ROOT}/launcher.log" ]]; then
        printf 'launcher.log missing\n'
        return
    fi
    rg '\] (start|done|failed) \||scheduler recorded failure|sweep completed' \
        "${ROOT}/launcher.log" 2>/dev/null | tail -n 1 \
        | sed -E 's/[[:space:]]+log=.*/ /' | cut -c1-300
}

resolve_lark_user_id() {
    if [[ -n "${LARK_NOTIFY_USER_ID}" ]]; then
        printf '%s\n' "${LARK_NOTIFY_USER_ID}"
        return 0
    fi
    command -v lark-cli >/dev/null 2>&1 || return 1
    lark-cli auth status 2>/dev/null \
        | sed -n 's/.*"openId": "\(ou_[^"]*\)".*/\1/p' \
        | head -n 1
}

build_status_message() {
    local title="$1"
    local done failed marked active queued percent last_done delta checkpoint_epoch checkpoint_age
    local bj_time algo_active task_active algo_done scheduler_event
    done="$(count_markers done)"
    failed="$(count_markers failed)"
    marked=$((done + failed))
    active="$(active_count)"
    queued=$((EXPECTED_RUNS - marked - active))
    (( queued >= 0 )) || queued=0
    percent="$(awk -v done="${done}" -v total="${EXPECTED_RUNS}" 'BEGIN {printf "%.1f", total ? 100 * done / total : 0}')"
    last_done="$(read_nonnegative "${LAST_REPORT_DONE_FILE}" "${done}")"
    delta=$((done - last_done))
    checkpoint_epoch="$(latest_checkpoint_epoch)"
    if [[ "${checkpoint_epoch}" =~ ^[0-9]+$ ]]; then
        checkpoint_age=$(( ($(date +%s) - checkpoint_epoch) / 60 ))
    else
        checkpoint_age=-1
    fi
    bj_time="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"
    algo_active="$(active_algo_summary)"
    task_active="$(active_task_summary)"
    algo_done="$(done_algo_summary)"
    scheduler_event="$(last_scheduler_event)"

    cat <<EOF
${title}
北京时间: ${bj_time}
总体完成: ${done}/${EXPECTED_RUNS} (${percent}%), 较上次 +${delta}
失败: ${failed}
当前运行: ${active}/${EXPECTED_ACTIVE}; 排队: ${queued}
运行算法: ${algo_active}
运行任务: ${task_active:-none}
各算法完成: ${algo_done}
最新 checkpoint: ${checkpoint_age} 分钟前
最近调度事件: ${scheduler_event}
EOF
}

send_status() {
    local event="$1"
    local title="$2"
    local message user_id key output
    message="$(build_status_message "${title}")"
    key="flowrl-8h-${event}-$(date +%s)"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s\n' "${message}"
        log "dry-run notification event=${event}"
        return 0
    fi

    user_id="$(resolve_lark_user_id || true)"
    if [[ -z "${user_id}" ]]; then
        log "notification failed event=${event}: no Feishu user id"
        return 1
    fi
    if output="$(lark-cli im +messages-send \
        --as "${LARK_NOTIFY_AS}" \
        --user-id "${user_id}" \
        --text "${message}" \
        --idempotency-key "${key}" 2>&1)"; then
        log "notification sent event=${event}"
        return 0
    fi
    log "notification failed event=${event}: ${output}"
    return 1
}

cleanup() {
    rm -f "${PID_FILE}"
    log "monitor stopped"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" >"${PID_FILE}"

for value_name in EXPECTED_RUNS EXPECTED_ACTIVE REPORT_INTERVAL_SECONDS POLL_SECONDS ACTIVE_LOG_SECONDS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1, got: ${DRY_RUN}" >&2
    exit 2
fi

log "monitor started root=${ROOT} report_interval=${REPORT_INTERVAL_SECONDS}s poll=${POLL_SECONDS}s"

while true; do
    now="$(date +%s)"
    done_count="$(count_markers done)"
    failed_count="$(count_markers failed)"
    active="$(active_count)"
    last_report_epoch="$(read_nonnegative "${LAST_REPORT_EPOCH_FILE}" 0)"
    last_failed="$(read_nonnegative "${LAST_FAILED_FILE}" 0)"

    if [[ ! -f "${STARTUP_SENT_FILE}" ]]; then
        if send_status startup "FlowRL 8卡实验监控已启动"; then
            printf '%s\n' "${now}" >"${LAST_REPORT_EPOCH_FILE}"
            printf '%s\n' "${done_count}" >"${LAST_REPORT_DONE_FILE}"
            date -u '+%F %T UTC' >"${STARTUP_SENT_FILE}"
            last_report_epoch="${now}"
        fi
    elif (( now - last_report_epoch >= REPORT_INTERVAL_SECONDS )); then
        if send_status periodic "FlowRL 8卡实验每8小时进度"; then
            printf '%s\n' "${now}" >"${LAST_REPORT_EPOCH_FILE}"
            printf '%s\n' "${done_count}" >"${LAST_REPORT_DONE_FILE}"
        fi
    fi

    if (( failed_count > last_failed )); then
        if send_status failure "FlowRL 8卡实验检测到失败"; then
            printf '%s\n' "${failed_count}" >"${LAST_FAILED_FILE}"
        fi
    fi

    if (( done_count + failed_count >= EXPECTED_RUNS )); then
        if [[ ! -f "${FINAL_SENT_FILE}" ]] && send_status complete "FlowRL 8卡实验队列已完成"; then
            date -u '+%F %T UTC' >"${FINAL_SENT_FILE}"
        fi
        log "queue fully marked done=${done_count} failed=${failed_count}; exiting"
        exit 0
    fi

    if (( active == 0 )); then
        if [[ ! -f "${STALLED_SENT_FILE}" ]] && send_status stalled "FlowRL 8卡实验疑似停止"; then
            date -u '+%F %T UTC' >"${STALLED_SENT_FILE}"
        fi
    elif [[ -f "${STALLED_SENT_FILE}" ]]; then
        rm -f "${STALLED_SENT_FILE}"
        send_status recovered "FlowRL 8卡实验已恢复运行" || true
    fi

    log "poll done=${done_count}/${EXPECTED_RUNS} failed=${failed_count} active=${active}"
    if [[ "${ONCE}" == "1" ]]; then
        exit 0
    fi
    sleep "${POLL_SECONDS}"
done
