#!/usr/bin/env bash
set -euo pipefail

# Wait for the current MuJoCo 4-seed sweep to finish, then launch the
# seed-major seed=4..9 continuation sweep. Sends Feishu/Lark notifications
# when blocked, when the continuation starts, and when it exits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OLD_PATTERN="${OLD_PATTERN:-scripts/mujoco/mujoco-onpolicy-baseline-[4]seed.sh}"
OLD_RUN_LOG_ROOT="${OLD_RUN_LOG_ROOT:-log/parallel/mujoco-onpolicy-baseline-200m-4seed}"
OLD_TOTAL_RUNS="${OLD_TOTAL_RUNS:-120}"
NEW_SCRIPT="${NEW_SCRIPT:-scripts/mujoco/mujoco-onpolicy-baseline-remaining-6seed-seedmajor.sh}"
NEW_RUN_LOG_ROOT="${NEW_RUN_LOG_ROOT:-log/parallel/mujoco-onpolicy-baseline-200m-seed4-9-seedmajor}"
NEW_LAUNCHER_LOG="${NEW_LAUNCHER_LOG:-${NEW_RUN_LOG_ROOT}/launcher.log}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

GPUS="${GPUS:-0 1 2 3}"
RUNS_PER_GPU="${RUNS_PER_GPU:-3}"

LARK_NOTIFY_AS="${LARK_NOTIFY_AS:-bot}"
LARK_NOTIFY_USER_ID="${LARK_NOTIFY_USER_ID:-ou_391fdc347968ed99c75575050107d63a}"
LARK_NOTIFY_CHAT_ID="${LARK_NOTIFY_CHAT_ID:-}"

mkdir -p "${NEW_RUN_LOG_ROOT}"

ts() {
    date '+%F %T'
}

log() {
    echo "[$(ts)] $*"
}

notify_lark() {
    local title="$1"
    local body="$2"
    local message="[flow-rl] ${title}

${body}"

    if ! command -v lark-cli >/dev/null 2>&1; then
        log "Lark notify skipped: lark-cli not found."
        return 0
    fi

    if [[ -z "${LARK_NOTIFY_CHAT_ID}" && -z "${LARK_NOTIFY_USER_ID}" ]]; then
        log "Lark notify skipped: set LARK_NOTIFY_CHAT_ID or LARK_NOTIFY_USER_ID."
        return 0
    fi

    local cmd=(lark-cli im +messages-send --as "${LARK_NOTIFY_AS}" --text "${message}")
    if [[ -n "${LARK_NOTIFY_CHAT_ID}" ]]; then
        cmd+=(--chat-id "${LARK_NOTIFY_CHAT_ID}")
    else
        cmd+=(--user-id "${LARK_NOTIFY_USER_ID}")
    fi
    cmd+=(--idempotency-key "flowrl-mujoco-autostart-$(date +%s)-${RANDOM}")

    if ! "${cmd[@]}" >/dev/null 2>&1; then
        log "Lark notify failed: ${title}"
    fi
}

old_running() {
    pgrep -f "${OLD_PATTERN}" >/dev/null 2>&1
}

count_markers() {
    local suffix="$1"
    find "${OLD_RUN_LOG_ROOT}" -maxdepth 1 -type f -name "*.${suffix}" 2>/dev/null | wc -l
}

main() {
    log "watcher started"
    log "old pattern: ${OLD_PATTERN}"
    log "old run logs: ${OLD_RUN_LOG_ROOT}"
    log "new script: ${NEW_SCRIPT}"
    log "new launcher log: ${NEW_LAUNCHER_LOG}"
    log "GPUS=${GPUS} RUNS_PER_GPU=${RUNS_PER_GPU}"

    while old_running; do
        local done_count failed_count
        done_count="$(count_markers done)"
        failed_count="$(count_markers failed)"
        log "old sweep still running; done=${done_count}/${OLD_TOTAL_RUNS} failed=${failed_count}; recheck in ${CHECK_INTERVAL_SECONDS}s"
        sleep "${CHECK_INTERVAL_SECONDS}"
    done

    local done_count failed_count
    done_count="$(count_markers done)"
    failed_count="$(count_markers failed)"
    log "old sweep no longer running; done=${done_count}/${OLD_TOTAL_RUNS} failed=${failed_count}"

    if (( failed_count > 0 )); then
        notify_lark \
            "MuJoCo 4seed finished with failures; continuation not started" \
            "旧实验已经退出，但发现 ${failed_count} 个 .failed 标记。新 seed=4..9 实验没有自动启动。
Log root: ${OLD_RUN_LOG_ROOT}"
        exit 1
    fi

    if (( done_count < OLD_TOTAL_RUNS )); then
        notify_lark \
            "MuJoCo 4seed exited before all runs completed; continuation not started" \
            "旧实验已经退出，但只有 ${done_count}/${OLD_TOTAL_RUNS} 个 .done 标记。新 seed=4..9 实验没有自动启动。
Log root: ${OLD_RUN_LOG_ROOT}"
        exit 1
    fi

    notify_lark \
        "MuJoCo 4seed completed; starting seed=4..9 continuation" \
        "旧实验 ${OLD_TOTAL_RUNS}/${OLD_TOTAL_RUNS} 全部完成且没有失败。现在启动 seed-major 续跑脚本。
New script: ${NEW_SCRIPT}
GPUS=${GPUS}
RUNS_PER_GPU=${RUNS_PER_GPU}
Log: ${NEW_LAUNCHER_LOG}"

    log "starting continuation"
    local status=0
    if (
        export GPUS RUNS_PER_GPU
        export RUN_LOG_ROOT="${NEW_RUN_LOG_ROOT}"
        bash "${NEW_SCRIPT}"
    ) >>"${NEW_LAUNCHER_LOG}" 2>&1; then
        status=0
    else
        status=$?
    fi

    if (( status == 0 )); then
        notify_lark \
            "MuJoCo seed=4..9 continuation completed" \
            "新 seed-major 实验已经正常结束。
Log: ${NEW_LAUNCHER_LOG}"
    else
        notify_lark \
            "MuJoCo seed=4..9 continuation failed" \
            "新 seed-major 实验退出码 ${status}。
Log: ${NEW_LAUNCHER_LOG}"
    fi

    log "continuation exited with status=${status}"
    exit "${status}"
}

main "$@"
