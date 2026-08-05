#!/usr/bin/env bash

set -u

# Watch the local completion markers for the final IsaacLab baseline completion
# launch and notify Feishu every N successful completions, plus any failures.

ROOT="${ROOT:-run_logs/isaaclab_baseline_final_complete_12task_4gpu_10seed_200m}"
ATTEMPT_FILE="${ATTEMPT_FILE:-${ROOT}/wandb_runs_to_attempt.tsv}"
FINISHED_FILE="${FINISHED_FILE:-${ROOT}/wandb_finished_runs.txt}"
EXPECTED_RUNS="${EXPECTED_RUNS:-}"
NOTIFY_EVERY="${NOTIFY_EVERY:-48}"
POLL_SECONDS="${POLL_SECONDS:-180}"
STATE_DIR="${STATE_DIR:-${ROOT}/feishu_monitor_state}"
LOG_FILE="${LOG_FILE:-${ROOT}/feishu_monitor.log}"
PID_FILE="${PID_FILE:-${STATE_DIR}/monitor.pid}"

LARK_NOTIFY_AS="${LARK_NOTIFY_AS:-user}"
LARK_NOTIFY_USER_ID="${LARK_NOTIFY_USER_ID:-}"
LARK_NOTIFY_CHAT_ID="${LARK_NOTIFY_CHAT_ID:-}"
LARK_NOTIFY_AUTO_SELF="${LARK_NOTIFY_AUTO_SELF:-1}"

STARTUP_NOTIFY="${STARTUP_NOTIFY:-1}"
NOTIFY_EXISTING_PROGRESS="${NOTIFY_EXISTING_PROGRESS:-0}"
NOTIFY_EXISTING_FAILURES="${NOTIFY_EXISTING_FAILURES:-1}"

mkdir -p "${ROOT}" "${STATE_DIR}"

LAST_DONE_FILE="${STATE_DIR}/last_done_threshold"
SEEN_FAILED_FILE="${STATE_DIR}/seen_failed"
STARTUP_SENT_FILE="${STATE_DIR}/startup_sent"
FINAL_SENT_FILE="${STATE_DIR}/final_sent"
STOPPED_SENT_FILE="${STATE_DIR}/stopped_incomplete_sent"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" >> "${LOG_FILE}"
}

count_files() {
  local pattern="$1"
  if [[ ! -d "${ROOT}" ]]; then
    printf '0\n'
    return
  fi
  find "${ROOT}" -type f -name "${pattern}" | wc -l | tr -d ' '
}

count_attempt_markers() {
  local suffix="$1"
  local count=0
  local task algo seed state name marker

  if [[ ! -f "${ATTEMPT_FILE}" ]]; then
    count_files "*.${suffix}"
    return
  fi

  while IFS=$'	' read -r task algo seed state name; do
    [[ "${task}" == "task" ]] && continue
    [[ -z "${name:-}" ]] && continue
    marker="${ROOT}/${name}.${suffix}"
    if [[ -f "${marker}" ]]; then
      count=$((count + 1))
    fi
  done < "${ATTEMPT_FILE}"

  printf '%s
' "${count}"
}

line_count_minus_header() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    printf '0\n'
    return
  fi
  local lines
  lines="$(wc -l < "${file}" | tr -d ' ')"
  if (( lines > 0 )); then
    printf '%s\n' "$((lines - 1))"
  else
    printf '0\n'
  fi
}

plain_line_count() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    printf '0\n'
    return
  fi
  wc -l < "${file}" | tr -d ' '
}

expected_runs() {
  if [[ -n "${EXPECTED_RUNS}" ]]; then
    printf '%s\n' "${EXPECTED_RUNS}"
    return
  fi
  line_count_minus_header "${ATTEMPT_FILE}"
}

count_active_final_python() {
  ps -ww -eo args= | awk '
    $1 ~ /(^|\/)python([0-9.]+)?$/ &&
    /examples\/online\/main_isaaclab_onpolicy\.py/ &&
    /log.project=isaaclab-baseline-final/ &&
    !/isaaclab-baseline-final-feishu-monitor/ &&
    !/awk/ { count++ }
    END { print count + 0 }
  '
}

count_active_launcher() {
  ps -ww -eo args= | awk '
    /isaaclab-baseline-final-complete-12task-4gpu\.sh/ &&
    !/isaaclab-baseline-final-feishu-monitor/ &&
    !/awk/ { count++ }
    END { print count + 0 }
  '
}

resolve_lark_user_id() {
  if [[ -n "${LARK_NOTIFY_USER_ID}" ]]; then
    printf '%s\n' "${LARK_NOTIFY_USER_ID}"
    return 0
  fi
  if [[ "${LARK_NOTIFY_AUTO_SELF}" != "1" ]]; then
    return 1
  fi
  lark-cli contact +get-user \
    --as "${LARK_NOTIFY_AS}" \
    --jq '.data.user.open_id // .data.open_id // .open_id // .userOpenId' 2>>"${LOG_FILE}"
}

send_feishu() {
  local event="$1"
  local key_suffix="$2"
  local msg="$3"
  local key="flowrl-final-monitor-${event}-${key_suffix}"

  if [[ -n "${LARK_NOTIFY_CHAT_ID}" ]]; then
    lark-cli im +messages-send \
      --as "${LARK_NOTIFY_AS}" \
      --chat-id "${LARK_NOTIFY_CHAT_ID}" \
      --text "${msg}" \
      --idempotency-key "${key}" >>"${LOG_FILE}" 2>&1
    return $?
  fi

  local user_id
  user_id="$(resolve_lark_user_id)"
  if [[ -z "${user_id}" || "${user_id}" == "null" ]]; then
    log "could not resolve Feishu user id; set LARK_NOTIFY_USER_ID or LARK_NOTIFY_CHAT_ID"
    return 1
  fi

  lark-cli im +messages-send \
    --as "${LARK_NOTIFY_AS}" \
    --user-id "${user_id}" \
    --text "${msg}" \
    --idempotency-key "${key}" >>"${LOG_FILE}" 2>&1
}

field_from_failed_file() {
  local file="$1"
  local key="$2"
  local k
  local v
  while IFS='=' read -r k v; do
    if [[ "${k}" == "${key}" ]]; then
      printf '%s\n' "${v}"
      return 0
    fi
  done < "${file}"
  printf '\n'
}

format_failed_summary() {
  local max_rows="${1:-12}"
  local rows=0
  local total=0
  local out=""
  local file
  shift || true
  for file in "$@"; do
    total=$((total + 1))
    if (( rows >= max_rows )); then
      continue
    fi
    local task algo seed exit_code source_state log_file
    task="$(field_from_failed_file "${file}" task)"
    algo="$(field_from_failed_file "${file}" algo)"
    seed="$(field_from_failed_file "${file}" seed)"
    exit_code="$(field_from_failed_file "${file}" exit_code)"
    source_state="$(field_from_failed_file "${file}" source_wandb_state)"
    log_file="$(field_from_failed_file "${file}" log_file)"
    out="${out}
- ${task:-unknown}/${algo:-unknown}/seed${seed:-?}: exit=${exit_code:-?}, source=${source_state:-?}, marker=$(basename "${file}"), log=${log_file:-?}"
    rows=$((rows + 1))
  done
  if (( total > max_rows )); then
    out="${out}
- ... plus $((total - max_rows)) more new failed markers"
  fi
  printf '%s\n' "${out}"
}

list_failed_files() {
  if [[ ! -d "${ROOT}" ]]; then
    return
  fi
  if [[ ! -f "${ATTEMPT_FILE}" ]]; then
    find "${ROOT}" -type f -name '*.failed' | sort
    return
  fi

  local task algo seed state name marker
  while IFS=$'	' read -r task algo seed state name; do
    [[ "${task}" == "task" ]] && continue
    [[ -z "${name:-}" ]] && continue
    marker="${ROOT}/${name}.failed"
    if [[ -f "${marker}" ]]; then
      printf '%s
' "${marker}"
    fi
  done < "${ATTEMPT_FILE}" | sort
}

send_status_message() {
  local event="$1"
  local key_suffix="$2"
  local title="$3"
  local extra="$4"
  local done_count failed_count total_count expected active_python active_launcher pre_finished bj_time
  done_count="$(count_attempt_markers done)"
  failed_count="$(count_attempt_markers failed)"
  total_count=$((done_count + failed_count))
  expected="$(expected_runs)"
  active_python="$(count_active_final_python)"
  active_launcher="$(count_active_launcher)"
  pre_finished="$(plain_line_count "${FINISHED_FILE}")"
  bj_time="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"

  local msg
  msg="${title}
Beijing time: ${bj_time}
Run root: ${ROOT}
Local scheduled runs: ${expected}
Local done: ${done_count}
Local failed: ${failed_count}
Local marked: ${total_count}/${expected}
W&B finished before launch cache: ${pre_finished}
Active final python processes: ${active_python}
Active launcher processes: ${active_launcher}${extra}"

  send_feishu "${event}" "${key_suffix}" "${msg}"
}

is_pid_alive_for_this_script() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  ps -p "${pid}" -ww -o args= 2>/dev/null | grep -Fq 'isaaclab-baseline-final-feishu-monitor.sh'
}

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if is_pid_alive_for_this_script "${old_pid}"; then
    log "monitor already running pid=${old_pid}; exiting duplicate"
    exit 0
  fi
fi

printf '%s\n' "$$" > "${PID_FILE}"
trap 'log "monitor stopped"; rm -f "${PID_FILE}"' EXIT

touch "${SEEN_FAILED_FILE}"

initial_done="$(count_attempt_markers done)"
if [[ ! -f "${LAST_DONE_FILE}" ]]; then
  if [[ "${NOTIFY_EXISTING_PROGRESS}" == "1" ]]; then
    printf '0\n' > "${LAST_DONE_FILE}"
  else
    printf '%s\n' "$(((initial_done / NOTIFY_EVERY) * NOTIFY_EVERY))" > "${LAST_DONE_FILE}"
  fi
fi

if [[ "${NOTIFY_EXISTING_FAILURES}" != "1" ]]; then
  list_failed_files > "${SEEN_FAILED_FILE}"
fi

log "monitor started root=${ROOT} expected=$(expected_runs) notify_every=${NOTIFY_EVERY} poll_seconds=${POLL_SECONDS} pid=$$"

if [[ "${STARTUP_NOTIFY}" == "1" && ! -f "${STARTUP_SENT_FILE}" ]]; then
  if send_status_message "startup" "started" "FlowRL final baseline monitor started" ""; then
    date -u '+%Y-%m-%d %H:%M:%S UTC' > "${STARTUP_SENT_FILE}"
    log "startup notification sent"
  else
    log "startup notification failed; will retry on next poll"
  fi
fi

while true; do
  done_count="$(count_attempt_markers done)"
  failed_count="$(count_attempt_markers failed)"
  total_count=$((done_count + failed_count))
  expected="$(expected_runs)"
  active_python="$(count_active_final_python)"
  active_launcher="$(count_active_launcher)"

  log "poll done=${done_count} failed=${failed_count} total=${total_count}/${expected} active_python=${active_python} active_launcher=${active_launcher}"

  if [[ "${STARTUP_NOTIFY}" == "1" && ! -f "${STARTUP_SENT_FILE}" ]]; then
    if send_status_message "startup" "started" "FlowRL final baseline monitor started" ""; then
      date -u '+%Y-%m-%d %H:%M:%S UTC' > "${STARTUP_SENT_FILE}"
      log "startup notification sent"
    else
      log "startup notification failed; retrying after ${POLL_SECONDS}s"
    fi
  fi

  mapfile -t current_failed < <(list_failed_files)
  new_failed=()
  for failed_file in "${current_failed[@]}"; do
    if ! grep -Fxq "${failed_file}" "${SEEN_FAILED_FILE}" 2>/dev/null; then
      new_failed+=("${failed_file}")
    fi
  done
  if (( ${#new_failed[@]} > 0 )); then
    failed_hash="$(printf '%s\n' "${new_failed[@]}" | cksum | awk '{print $1}')"
    failed_extra="$(format_failed_summary 12 "${new_failed[@]}")"
    if send_status_message "failed" "${failed_hash}" "FlowRL final baseline failure detected" "${failed_extra}"; then
      printf '%s\n' "${new_failed[@]}" >> "${SEEN_FAILED_FILE}"
      sort -u "${SEEN_FAILED_FILE}" -o "${SEEN_FAILED_FILE}"
      log "failed notification sent new_failed=${#new_failed[@]}"
    else
      log "failed notification failed new_failed=${#new_failed[@]}; retrying after ${POLL_SECONDS}s"
    fi
  fi

  last_done="$(cat "${LAST_DONE_FILE}" 2>/dev/null || printf '0\n')"
  [[ "${last_done}" =~ ^[0-9]+$ ]] || last_done=0
  next_done=$((last_done + NOTIFY_EVERY))
  while (( done_count >= next_done && NOTIFY_EVERY > 0 )); do
    progress_extra="
Progress threshold reached: ${next_done} local done runs."
    if send_status_message "progress" "${next_done}" "FlowRL final baseline progress update" "${progress_extra}"; then
      printf '%s\n' "${next_done}" > "${LAST_DONE_FILE}"
      log "progress notification sent threshold=${next_done}"
      last_done="${next_done}"
      next_done=$((last_done + NOTIFY_EVERY))
    else
      log "progress notification failed threshold=${next_done}; retrying after ${POLL_SECONDS}s"
      break
    fi
  done

  if (( expected > 0 && total_count >= expected )) && [[ ! -f "${FINAL_SENT_FILE}" ]]; then
    if send_status_message "final" "all-marked" "FlowRL final baseline local queue fully marked" ""; then
      date -u '+%Y-%m-%d %H:%M:%S UTC' > "${FINAL_SENT_FILE}"
      log "final notification sent"
    else
      log "final notification failed; retrying after ${POLL_SECONDS}s"
    fi
  fi

  if (( expected > 0 && total_count < expected && active_python == 0 && active_launcher == 0 )) && [[ ! -f "${STOPPED_SENT_FILE}" ]]; then
    stopped_extra="
No final python or launcher process is currently visible while the local queue is incomplete."
    if send_status_message "stopped-incomplete" "no-active-process" "FlowRL final baseline monitor warning" "${stopped_extra}"; then
      date -u '+%Y-%m-%d %H:%M:%S UTC' > "${STOPPED_SENT_FILE}"
      log "stopped-incomplete notification sent"
    else
      log "stopped-incomplete notification failed; retrying after ${POLL_SECONDS}s"
    fi
  fi

  sleep "${POLL_SECONDS}"
done
