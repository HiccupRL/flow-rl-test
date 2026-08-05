#!/usr/bin/env bash

set -u

ROOT="${ROOT:-run_logs/isaaclab_policyflow_baseline_4gpu_10seed_100m}"
EXPECTED_RUNS="${EXPECTED_RUNS:-80}"
POLL_SECONDS="${POLL_SECONDS:-300}"
USER_ID="${USER_ID:-ou_a240f4006a7d88cf69c0c31b0e6cd776}"
NOTIFIED_FILE="${NOTIFIED_FILE:-/tmp/flowrl_policyflow_monitor.notified}"
LOG_FILE="${LOG_FILE:-$ROOT/feishu_monitor.log}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" >> "$LOG_FILE"
}

count_files() {
  local pattern="$1"
  if [[ ! -d "$ROOT" ]]; then
    printf '0\n'
    return
  fi
  find "$ROOT" -type f -name "$pattern" | wc -l | tr -d ' '
}

count_active_policyflow() {
  ps -eo args= | awk '
    /examples\/online\/main_isaaclab_onpolicy\.py/ &&
    /algo=policyflow/ &&
    !/policyflow_feishu_notify_monitor/ &&
    !/awk/ { count++ }
    END { print count + 0 }
  '
}

count_active_launcher() {
  ps -eo args= | awk '
    /isaaclab-policyflow-baseline-4gpu-10seed\.sh/ &&
    !/policyflow_feishu_notify_monitor/ &&
    !/awk/ { count++ }
    END { print count + 0 }
  '
}

send_feishu() {
  local state="$1"
  local done_count="$2"
  local failed_count="$3"
  local total_count="$4"
  local active_count="$5"
  local launcher_count="$6"
  local bj_time
  local msg
  bj_time="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"
  msg="PolicyFlow baseline finished notification
State: $state
Beijing time: $bj_time
Run root: $ROOT
Done: $done_count
Failed: $failed_count
Marked total: $total_count / $EXPECTED_RUNS
Active PolicyFlow processes: $active_count
Active launchers: $launcher_count"

  lark-cli im +messages-send \
    --as user \
    --user-id "$USER_ID" \
    --text "$msg" \
    --idempotency-key "policyflow-baseline-${state}-$(date +%s)"
}

if [[ -f "$NOTIFIED_FILE" ]]; then
  log "notification marker already exists: $NOTIFIED_FILE"
  exit 0
fi

log "policyflow monitor started root=$ROOT expected=$EXPECTED_RUNS notified_file=$NOTIFIED_FILE"

while true; do
  done_count="$(count_files '*.done')"
  failed_count="$(count_files '*.failed')"
  total_count=$((done_count + failed_count))
  active_count="$(count_active_policyflow)"
  launcher_count="$(count_active_launcher)"

  log "poll done=$done_count failed=$failed_count total=$total_count/$EXPECTED_RUNS active=$active_count launcher=$launcher_count"

  state=""
  if (( total_count >= EXPECTED_RUNS )); then
    state="completed_all_marked"
  elif (( active_count == 0 && launcher_count == 0 && total_count > 0 )); then
    state="stopped_incomplete"
  fi

  if [[ -n "$state" ]]; then
    if send_feishu "$state" "$done_count" "$failed_count" "$total_count" "$active_count" "$launcher_count" >> "$LOG_FILE" 2>&1; then
      printf '%s\n' "$state" > "$NOTIFIED_FILE"
      log "feishu notification sent state=$state"
      exit 0
    fi
    log "feishu notification failed state=$state; retrying after $POLL_SECONDS seconds"
  fi

  sleep "$POLL_SECONDS"
done
