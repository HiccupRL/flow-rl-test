#!/usr/bin/env bash
set -euo pipefail

# Reproduce one selected seed for each IsaacLab task/baseline cell, then render
# and validate its final inference video.  GPUS may contain any
# non-empty set of unique physical GPU indices.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS_SPEC="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARRAY <<<"${GPUS_SPEC}"
RUN_LABEL="${RUN_LABEL:-isaaclab_baseline_best_final_video72}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/isaaclab_baseline_final_videos}"
LOG_BASE="${LOG_BASE:-${OUTPUT_ROOT}/run_logs/${RUN_LABEL}}"
MANIFEST="${MANIFEST:-scripts/isaaclab/search_spaces/isaaclab_baseline_best_final_video72.tsv}"
JOB_HELPER="${JOB_HELPER:-scripts/isaaclab/reproduce_isaaclab_baseline_video_job.py}"
EXPORTER="${EXPORTER:-scripts/isaaclab/export_isaaclab_baseline_checkpoint_video.py}"
TRAIN_WRAPPER="${TRAIN_WRAPPER:-scripts/isaaclab/train_isaaclab_baseline_from_resolved_config.py}"
POLICYFLOW_COMPAT_PATCH="${POLICYFLOW_COMPAT_PATCH:-scripts/isaaclab/policyflow_recorded_commit_compat.patch}"
BASELINE_COMMIT="${BASELINE_COMMIT:-6ca3ecae50d3e0e52cff5fdcef57552c7402d927}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED="${EXPECTED:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS:-3}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-30}"
TRAIN_FRAMES_OVERRIDE="${TRAIN_FRAMES_OVERRIDE:-}"
EVAL_FRAMES_OVERRIDE="${EVAL_FRAMES_OVERRIDE:-}"
LOG_FRAMES_OVERRIDE="${LOG_FRAMES_OVERRIDE:-}"
VIDEO_STEPS="${VIDEO_STEPS:-1000}"
VIDEO_FPS="${VIDEO_FPS:-50}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
RENDER_TIMEOUT_SECONDS="${RENDER_TIMEOUT_SECONDS:-1800}"
VIDEO_PREREQ_CHECK="${VIDEO_PREREQ_CHECK:-1}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-45000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
DEFERRED_GPU="${DEFERRED_GPU:-}"
DEFERRED_GPU_READY_FILE="${DEFERRED_GPU_READY_FILE:-}"
SOURCE_DIR="${SOURCE_DIR:-}"
NVIDIA_VENDOR_ROOT="${NVIDIA_VENDOR_ROOT:-${REPO_ROOT}/../flow-rl-final/isaaclab_best_return_videos/nvidia_gl_550.163.01}"
NVIDIA_VENDOR_LIB="${NVIDIA_VENDOR_LIB:-${NVIDIA_VENDOR_ROOT}/usr/lib/x86_64-linux-gnu}"
NVIDIA_VULKAN_ICD="${NVIDIA_VULKAN_ICD:-${NVIDIA_VENDOR_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json}"
NVIDIA_EGL_VENDOR="${NVIDIA_EGL_VENDOR:-${NVIDIA_VENDOR_ROOT}/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
QUEUE_NEXT="${QUEUE_NEXT:-${LOG_BASE}/.queue_next}"
QUEUE_LOCK="${QUEUE_LOCK:-${LOG_BASE}/.queue_lock}"
RUNNER_LOCK="${RUNNER_LOCK:-${LOG_BASE}/.runner.lock}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

log() {
    echo "[$(date '+%F %T')] $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( 10#$1 > 0 ))
}

case "${OMNI_KIT_ACCEPT_EULA,,}" in
    yes|y|1) ;;
    *) die "OMNI_KIT_ACCEPT_EULA must be YES, Y, or 1" ;;
esac

(( ${#GPU_ARRAY[@]} > 0 )) || die "GPUS must contain at least one GPU index"
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU index: ${gpu}"
    [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "duplicate GPU index: ${gpu}"
    SEEN_GPUS["${gpu}"]=1
done
if [[ -n "${DEFERRED_GPU}" || -n "${DEFERRED_GPU_READY_FILE}" ]]; then
    [[ -n "${DEFERRED_GPU}" && -n "${DEFERRED_GPU_READY_FILE}" ]] || \
        die "DEFERRED_GPU and DEFERRED_GPU_READY_FILE must be set together"
    [[ "${DEFERRED_GPU}" =~ ^[0-9]+$ ]] || \
        die "DEFERRED_GPU must be a GPU index"
    [[ -n "${SEEN_GPUS[${DEFERRED_GPU}]:-}" ]] || \
        die "DEFERRED_GPU must be present in GPUS"
fi

for required_command in flock nvidia-smi patch sha256sum tar; do
    command -v "${required_command}" >/dev/null 2>&1 || \
        die "${required_command} is required"
done
for required_file in "${MANIFEST}" "${JOB_HELPER}" "${EXPORTER}" \
    "${TRAIN_WRAPPER}" "${POLICYFLOW_COMPAT_PATCH}"; do
    [[ -s "${required_file}" ]] || die "missing required file: ${required_file}"
done
for value_name in \
    MAX_JOB_ATTEMPTS RETRY_BACKOFF_SECONDS VIDEO_STEPS VIDEO_FPS VIDEO_WIDTH \
    VIDEO_HEIGHT RENDER_TIMEOUT_SECONDS MIN_FREE_MEM_MB GPU_POLL_SECONDS; do
    value="${!value_name}"
    if [[ "${value_name}" == "RETRY_BACKOFF_SECONDS" && "${value}" == "0" ]]; then
        continue
    fi
    positive_integer "${value}" || die "${value_name} must be a positive integer"
done
for value_name in TRAIN_FRAMES_OVERRIDE EVAL_FRAMES_OVERRIDE LOG_FRAMES_OVERRIDE; do
    value="${!value_name}"
    [[ -z "${value}" ]] || positive_integer "${value}" || \
        die "${value_name} must be a positive integer when set"
done
for boolean_name in \
    DRY_RUN PREPARE_ONLY SKIP_DONE VIDEO_PREREQ_CHECK WAIT_FOR_GPU_FREE; do
    value="${!boolean_name}"
    [[ "${value}" == "0" || "${value}" == "1" ]] || \
        die "${boolean_name} must be 0 or 1"
done
git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null || \
    die "recorded baseline commit is unavailable: ${BASELINE_COMMIT}"

mapfile -t ROWS < <(tail -n +2 "${MANIFEST}" | tr -d '\r')
[[ -n "${EXPECTED}" ]] || EXPECTED="${#ROWS[@]}"
positive_integer "${EXPECTED}" || die "EXPECTED must be a positive integer"
(( ${#ROWS[@]} == EXPECTED )) || \
    die "manifest count mismatch: expected ${EXPECTED}, got ${#ROWS[@]}"

"${PYTHON_BIN}" - "${MANIFEST}" "${EXPECTED}" "${REPO_ROOT}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path
import yaml

manifest = Path(sys.argv[1])
expected = int(sys.argv[2])
root = Path(sys.argv[3])
with manifest.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
required = {
    "slug", "task", "algo", "seed", "reference_return", "reference_frame",
    "selection_status", "run_id", "commit", "metadata_path", "config_path",
    "config_sha256", "source_status",
}
if len(rows) != expected:
    raise SystemExit(f"expected {expected} rows, found {len(rows)}")
if not rows or required - set(rows[0]):
    raise SystemExit(f"manifest missing columns: {sorted(required - set(rows[0] if rows else {}))}")
allowed_source_status = {
    "recorded_commit_agent_source",
    "recorded_commit_agent_plus_compat_symbol_overlay",
    "best_available_uncommitted_policyflow_overlay",
}
if len({row["slug"] for row in rows}) != len(rows):
    raise SystemExit("manifest slugs must be unique")
allowed_status = {"strict_200m_terminal_best", "historical_incomplete_fallback"}
for row in rows:
    metadata_path = Path(row["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not metadata_path.is_file():
        raise SystemExit(f"missing metadata for {row['slug']}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arguments = dict(
        item.split("=", 1) for item in metadata.get("args", []) if "=" in item
    )
    expected_identity = {
        "task": row["task"],
        "algo": row["algo"],
        "seed": row["seed"],
        "train_frames": "200000000",
    }
    config_path = Path(row["config_path"])
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise SystemExit(f"missing resolved config for {row['slug']}: {config_path}")
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_digest != row["config_sha256"]:
        raise SystemExit(f"resolved config hash mismatch for {row['slug']}")
    if row["source_status"] not in allowed_source_status:
        raise SystemExit(
            f"unknown source status for {row['slug']}: "
            f"{row['source_status']}"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_identity = {
        "task": str(config.get("task", "")),
        "algo": str(config.get("algo", {}).get("name", "")),
        "seed": str(config.get("seed", "")),
        "train_frames": str(config.get("train_frames", "")),
    }
    if config_identity != expected_identity:
        raise SystemExit(
            f"resolved config identity mismatch for {row['slug']}: "
            f"{config_identity} != {expected_identity}"
        )
    actual = {key: arguments.get(key) for key in expected_identity}
    if actual != expected_identity:
        raise SystemExit(
            f"metadata identity mismatch for {row['slug']}: "
            f"{actual} != {expected_identity}"
        )
    metadata_commit = str(metadata.get("git", {}).get("commit", "")).strip()
    if metadata_commit and row["commit"] != metadata_commit:
        raise SystemExit(f"commit mismatch for {row['slug']}")
    if row["selection_status"] not in allowed_status:
        raise SystemExit(
            f"unknown selection status for {row['slug']}: "
            f"{row['selection_status']}"
        )
    int(row["seed"])
    int(row["reference_frame"])
    float(row["reference_return"])
print(f"validated {len(rows)} manifest rows and provenance records")
PY

mkdir -p "${OUTPUT_ROOT}" "${LOG_BASE}"
exec 8>"${RUNNER_LOCK}"
flock -n 8 || die "another runner already owns ${RUNNER_LOCK}"

if [[ "${DRY_RUN}" != "1" && "${VIDEO_PREREQ_CHECK}" == "1" ]]; then
    [[ -d "${NVIDIA_VENDOR_LIB}" ]] || \
        die "missing NVIDIA vendor library directory: ${NVIDIA_VENDOR_LIB}"
    [[ -s "${NVIDIA_VULKAN_ICD}" && -s "${NVIDIA_EGL_VENDOR}" ]] || \
        die "missing private NVIDIA Vulkan/GLVND manifests"
    command -v vulkaninfo >/dev/null 2>&1 || \
        die "vulkaninfo is required for VIDEO_PREREQ_CHECK=1"
    runtime_dir="${XDG_RUNTIME_DIR:-${OUTPUT_ROOT}/.xdg_runtime}"
    mkdir -p "${runtime_dir}"
    chmod 700 "${runtime_dir}"
    export XDG_RUNTIME_DIR="${runtime_dir}"
    export LD_LIBRARY_PATH="${NVIDIA_VENDOR_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export VK_ICD_FILENAMES="${NVIDIA_VULKAN_ICD}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR}"
    vulkan_summary="$(vulkaninfo --summary 2>&1)"
    detected_gpus="$(grep -c 'deviceName.*NVIDIA GeForce RTX 4090' <<<"${vulkan_summary}" || true)"
    (( detected_gpus >= ${#GPU_ARRAY[@]} )) || {
        echo "${vulkan_summary}" >&2
        die "Vulkan found ${detected_gpus} RTX 4090 GPUs; need ${#GPU_ARRAY[@]}"
    }
    log "validated private NVIDIA Vulkan stack (${detected_gpus} RTX 4090 GPUs)"
fi

assigned_gpu_free() {
    local gpu="$1" free compute_pids
    free="$(
        nvidia-smi -i "${gpu}" --query-gpu=memory.free \
            --format=csv,noheader,nounits | tr -d '[:space:]'
    )"
    [[ "${free}" =~ ^[0-9]+$ ]] || return 1
    (( free >= MIN_FREE_MEM_MB )) || return 1
    compute_pids="$(
        nvidia-smi -i "${gpu}" --query-compute-apps=pid \
            --format=csv,noheader,nounits | tr -d '[:space:]'
    )"
    [[ -z "${compute_pids}" ]]
}

all_assigned_gpus_free() {
    local gpu
    for gpu in "${GPU_ARRAY[@]}"; do
        assigned_gpu_free "${gpu}" || return 1
    done
}

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${WAIT_FOR_GPU_FREE}" == "1" ]]; then
        log "elastic mode: each GPU worker will join as soon as its GPU is free"
    elif ! all_assigned_gpus_free; then
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits >&2
        die "one or more assigned GPUs are busy or below ${MIN_FREE_MEM_MB} MiB free"
    else
        log "GPUs ${GPUS_SPEC} passed the process and free-memory checks"
    fi
fi

prepare_source_snapshot() {
    local overlay_head short_commit stamp snapshot temporary marker
    local manifest_hash compat_patch_hash
    overlay_head="$(git rev-parse HEAD)"
    short_commit="${BASELINE_COMMIT:0:12}"
    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    snapshot="${OUTPUT_ROOT}/source_snapshot/baseline_${short_commit}_overlay_${stamp}"
    temporary="${snapshot}.preparing.$$"
    marker="${snapshot}/.baseline_video_source_ready"

    mkdir -p "${temporary}/scripts/isaaclab/search_spaces"
    git archive "${BASELINE_COMMIT}" | tar -x -C "${temporary}"
    cp examples/online/main_isaaclab_onpolicy.py \
        "${temporary}/examples/online/main_isaaclab_onpolicy.py"
    cp flowrl/agent/base.py "${temporary}/flowrl/agent/base.py"
    cp \
        flowrl/agent/online/__init__.py \
        flowrl/agent/online/dppo_scratch.py \
        flowrl/agent/online/policyflow.py \
        "${temporary}/flowrl/agent/online/"
    cp -a flowrl/config/online/. "${temporary}/flowrl/config/online/"
    cp flowrl/utils/action_smoothness.py \
        "${temporary}/flowrl/utils/action_smoothness.py"
    cp examples/online/config/isaaclab_onpolicy/algo/policyflow.yaml \
        "${temporary}/examples/online/config/isaaclab_onpolicy/algo/policyflow.yaml"
    cp "${JOB_HELPER}" \
        "${temporary}/scripts/isaaclab/reproduce_isaaclab_baseline_video_job.py"
    cp "${EXPORTER}" \
        "${temporary}/scripts/isaaclab/export_isaaclab_baseline_checkpoint_video.py"
    cp "${TRAIN_WRAPPER}" \
        "${temporary}/scripts/isaaclab/train_isaaclab_baseline_from_resolved_config.py"
    cp "${MANIFEST}" \
        "${temporary}/scripts/isaaclab/search_spaces/baseline_video_manifest.tsv"
    cp "${POLICYFLOW_COMPAT_PATCH}" \
        "${temporary}/scripts/isaaclab/policyflow_recorded_commit_compat.patch"
    (
        cd "${temporary}"
        patch -p1 < "${REPO_ROOT}/${POLICYFLOW_COMPAT_PATCH}"
    )
    manifest_hash="$(sha256sum "${MANIFEST}")"
    manifest_hash="${manifest_hash%% *}"
    compat_patch_hash="$(sha256sum "${POLICYFLOW_COMPAT_PATCH}")"
    compat_patch_hash="${compat_patch_hash%% *}"
    {
        printf 'recorded_baseline_commit=%s\n' "${BASELINE_COMMIT}"
        printf 'overlay_worktree_head=%s\n' "${overlay_head}"
        printf 'created_at=%s\n' "${stamp}"
        printf 'source_repo=%s\n' "${REPO_ROOT}"
        printf 'manifest_input=%s\n' "$(realpath "${MANIFEST}")"
        printf 'manifest_sha256=%s\n' "${manifest_hash}"
        printf 'policyflow_compat_patch_sha256=%s\n' "${compat_patch_hash}"
        printf 'fpo_source_semantics=recorded commit plus unused clamp_ste compatibility symbol\n'
        printf 'policyflow_source_semantics=best available compatible reconstruction from uncommitted overlay\n'
        printf '\n[overlay worktree git status --short]\n'
        git status --short
    } > "${temporary}/SOURCE_PROVENANCE.txt"
    (
        cd "${temporary}"
        find flowrl examples scripts/isaaclab -type f \
            -regextype posix-extended -regex '.*[.](py|yaml|yml|patch|tsv)' \
            -print0 | sort -z | xargs -0 sha256sum
    ) > "${temporary}/SOURCE_FILES.sha256"
    touch "${temporary}/.baseline_video_source_ready"
    mkdir -p "$(dirname "${snapshot}")"
    mv "${temporary}" "${snapshot}"
    SOURCE_DIR="${snapshot}"
    log "froze recorded baseline source plus PolicyFlow/checkpoint overlay at ${SOURCE_DIR}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ -z "${SOURCE_DIR}" ]]; then
        prepare_source_snapshot
    else
        SOURCE_DIR="$(realpath "${SOURCE_DIR}")"
        [[ -f "${SOURCE_DIR}/.baseline_video_source_ready" ]] || \
            die "SOURCE_DIR lacks .baseline_video_source_ready: ${SOURCE_DIR}"
    fi
    [[ -s "${SOURCE_DIR}/SOURCE_FILES.sha256" ]] || \
        die "snapshot lacks SOURCE_FILES.sha256: ${SOURCE_DIR}"
    (cd "${SOURCE_DIR}" && sha256sum -c SOURCE_FILES.sha256 >/dev/null) || \
        die "snapshot source checksum validation failed: ${SOURCE_DIR}"
    log "validated frozen source checksums"
    [[ -s "${SOURCE_DIR}/examples/online/main_isaaclab_onpolicy.py" ]] || \
        die "snapshot lacks the IsaacLab trainer"
    [[ -s "${SOURCE_DIR}/scripts/isaaclab/reproduce_isaaclab_baseline_video_job.py" ]] || \
        die "snapshot lacks the frozen job helper"
    [[ -s "${SOURCE_DIR}/scripts/isaaclab/export_isaaclab_baseline_checkpoint_video.py" ]] || \
        die "snapshot lacks the frozen exporter"
    [[ -s "${SOURCE_DIR}/scripts/isaaclab/train_isaaclab_baseline_from_resolved_config.py" ]] || \
        die "snapshot lacks the frozen resolved-config training wrapper"
    printf '%s\n' "${SOURCE_DIR}" > "${LOG_BASE}/source_dir.txt"
    if [[ "${PREPARE_ONLY}" == "1" ]]; then
        log "PREPARE_ONLY complete: ${SOURCE_DIR}"
        exit 0
    fi
fi

printf '0\n' > "${QUEUE_NEXT}"
worker_pids=()

cleanup() {
    local exit_code="$?" path pid command_line
    trap - EXIT INT TERM HUP
    for pid in "${worker_pids[@]:-}"; do
        [[ "${pid}" =~ ^[0-9]+$ ]] && kill -TERM "${pid}" 2>/dev/null || true
    done
    while IFS= read -r path; do
        [[ -n "${path}" ]] || continue
        pid="$(<"${path}")"
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
            if [[ "${command_line}" == *"reproduce_isaaclab_baseline_video_job.py"* ]]; then
                kill -TERM -- "-${pid}" 2>/dev/null || \
                    kill -TERM "${pid}" 2>/dev/null || true
            fi
        fi
    done < <(find "${LOG_BASE}" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null)
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM HUP

claim_next() {
    local next
    flock 9
    next="$(<"${QUEUE_NEXT}")"
    if (( next >= EXPECTED )); then
        flock -u 9
        return 1
    fi
    printf '%s\n' "$((next + 1))" > "${QUEUE_NEXT}"
    flock -u 9
    printf '%s\n' "${next}"
}

worker() {
    local worker_index="$1" gpu="$2" index row
    local slug task algo seed reference_return reference_frame
    local selection_status source_run_id commit metadata_path config_path
    local config_sha256 source_status
    local task_output marker attempt rc pid video
    local -a job_extra_args
    trap - EXIT INT TERM HUP
    if [[ "${DRY_RUN}" != "1" && "${gpu}" == "${DEFERRED_GPU}" ]]; then
        while [[ ! -f "${DEFERRED_GPU_READY_FILE}" ]]; do
            log "GPU ${gpu}: waiting for deferred ready gate ${DEFERRED_GPU_READY_FILE}"
            sleep "${GPU_POLL_SECONDS}"
        done
        log "GPU ${gpu}: deferred ready gate opened"
    fi
    if [[ "${DRY_RUN}" != "1" && "${WAIT_FOR_GPU_FREE}" == "1" ]]; then
        while ! assigned_gpu_free "${gpu}"; do
            log "GPU ${gpu}: waiting to be process-free with ${MIN_FREE_MEM_MB} MiB free"
            sleep "${GPU_POLL_SECONDS}"
        done
        log "GPU ${gpu}: passed the process and free-memory checks"
    fi
    exec 9>"${QUEUE_LOCK}"
    while index="$(claim_next)"; do
        row="${ROWS[${index}]}"
        IFS=$'\t' read -r \
            slug task algo seed reference_return reference_frame \
            selection_status source_run_id commit metadata_path config_path \
            config_sha256 source_status <<<"${row}"
        task_output="${OUTPUT_ROOT}/tasks/${slug}"
        marker="${LOG_BASE}/${slug}"
        video="${OUTPUT_ROOT}/${slug}_seed${seed}_best_final.mp4"
        mkdir -p "${task_output}"
        if [[ "${SKIP_DONE}" == "1" && -f "${marker}.done" && \
              -s "${video}" && -s "${video%.mp4}.json" ]]; then
            log "GPU ${gpu}: skip completed ${slug}"
            continue
        fi
        if [[ "${DRY_RUN}" == "1" ]]; then
            printf 'GPU=%s slug=%s task=%s algo=%s seed=%s status=%s\n' \
                "${gpu}" "${slug}" "${task}" "${algo}" "${seed}" \
                "${selection_status}"
            continue
        fi

        job_extra_args=(
            --steps "${VIDEO_STEPS}"
            --fps "${VIDEO_FPS}"
            --width "${VIDEO_WIDTH}"
            --height "${VIDEO_HEIGHT}"
            --render-timeout-seconds "${RENDER_TIMEOUT_SECONDS}"
        )
        [[ -z "${TRAIN_FRAMES_OVERRIDE}" ]] || \
            job_extra_args+=(--train-frames "${TRAIN_FRAMES_OVERRIDE}")
        [[ -z "${EVAL_FRAMES_OVERRIDE}" ]] || \
            job_extra_args+=(--eval-frames "${EVAL_FRAMES_OVERRIDE}")
        [[ -z "${LOG_FRAMES_OVERRIDE}" ]] || \
            job_extra_args+=(--log-frames "${LOG_FRAMES_OVERRIDE}")

        attempt=0
        while true; do
            attempt=$((attempt + 1))
            printf 'started_at=%s\ngpu=%s\nattempt=%s\n' \
                "$(date -Iseconds)" "${gpu}" "${attempt}" > "${marker}.running"
            log "GPU ${gpu}: ${slug} attempt ${attempt} (${task}, seed ${seed})"
            setsid "${PYTHON_BIN}" \
                "${SOURCE_DIR}/scripts/isaaclab/reproduce_isaaclab_baseline_video_job.py" \
                --source-dir "${SOURCE_DIR}" \
                --metadata "${metadata_path}" \
                --output-dir "${task_output}" \
                --historical-config "${config_path}" \
                --config-sha256 "${config_sha256}" \
                --source-status "${source_status}" \
                --gpu "${gpu}" \
                --slug "${slug}" \
                --task "${task}" \
                --algo "${algo}" \
                --seed "${seed}" \
                --commit "${commit}" \
                --reference-return "${reference_return}" \
                --reference-frame "${reference_frame}" \
                --selection-status "${selection_status}" \
                --source-run-id "${source_run_id}" \
                --run-label "${RUN_LABEL}" \
                --python "${PYTHON_BIN}" \
                "${job_extra_args[@]}" \
                > "${task_output}/job.attempt_${attempt}.log" 2>&1 &
            pid=$!
            echo "${pid}" > "${marker}.pid"
            if wait "${pid}"; then rc=0; else rc=$?; fi
            rm -f "${marker}.pid"
            if (( rc == 0 )); then
                mv "${marker}.running" "${marker}.done"
                printf 'finished_at=%s\ngpu=%s\nattempt=%s\nvideo=%s\n' \
                    "$(date -Iseconds)" "${gpu}" "${attempt}" "${video}" \
                    >> "${marker}.done"
                log "GPU ${gpu}: ${slug} succeeded"
                break
            fi
            mv "${marker}.running" "${marker}.attempt_${attempt}.failed"
            log "GPU ${gpu}: ${slug} failed rc=${rc}"
            if (( MAX_JOB_ATTEMPTS > 0 && attempt >= MAX_JOB_ATTEMPTS )); then
                return "${rc}"
            fi
            sleep "${RETRY_BACKOFF_SECONDS}"
        done
    done
}

log "matrix: ${EXPECTED} selected IsaacLab task/baseline reproductions"
log "GPUs: ${GPUS_SPEC}; dynamic queue with one experiment per GPU"
log "output: ${OUTPUT_ROOT}"
if [[ -n "${TRAIN_FRAMES_OVERRIDE}" ]]; then
    log "short-run override: train=${TRAIN_FRAMES_OVERRIDE}, eval=${EVAL_FRAMES_OVERRIDE:-original}, log=${LOG_FRAMES_OVERRIDE:-original}"
fi

for index in "${!GPU_ARRAY[@]}"; do
    worker "${index}" "${GPU_ARRAY[${index}]}" &
    worker_pids+=("$!")
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        worker_failure=1
    fi
done
worker_pids=()
if (( worker_failure != 0 )); then
    echo "At least one worker exhausted MAX_JOB_ATTEMPTS." >&2
    exit 1
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN complete: validated and assigned ${EXPECTED} manifest rows"
    trap - EXIT INT TERM HUP
    exit 0
fi

done_count="$(find "${LOG_BASE}" -maxdepth 1 -type f -name '*.done' | wc -l)"
mp4_count="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type f -name '*_best_final.mp4' | wc -l)"
if (( done_count != EXPECTED || mp4_count != EXPECTED )); then
    die "completion mismatch: done=${done_count}, mp4=${mp4_count}, expected=${EXPECTED}"
fi

"${PYTHON_BIN}" - \
    "${OUTPUT_ROOT}" \
    "${EXPECTED}" \
    "${MANIFEST}" \
    "${REPO_ROOT}" \
    "${TRAIN_FRAMES_OVERRIDE:-0}" <<'PY'
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

root = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = Path(sys.argv[3])
repo_root = Path(sys.argv[4]).resolve()
minimum_checkpoint_frame = int(sys.argv[5])
with manifest.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if len(rows) != expected:
    raise SystemExit(f"expected {expected} manifest rows, found {len(rows)}")
expected_names = {
    f"{row['slug']}_seed{row['seed']}_best_final.mp4" for row in rows
}
actual_names = {path.name for path in root.glob("*_best_final.mp4")}
if actual_names != expected_names:
    raise SystemExit(
        "video filename set differs from manifest: "
        f"missing={sorted(expected_names - actual_names)}, "
        f"unexpected={sorted(actual_names - expected_names)}"
    )
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for row in rows:
    video = root / f"{row['slug']}_seed{row['seed']}_best_final.mp4"
    metadata_path = video.with_suffix(".json")
    if not metadata_path.is_file():
        raise SystemExit(f"missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_identity = {
        "task": row["task"],
        "algorithm": row["algo"],
        "seed": int(row["seed"]),
        "selection_status": row["selection_status"],
        "source_run_id": row["run_id"],
        "historical_config_sha256": row["config_sha256"],
        "source_status": row["source_status"],
    }
    actual_identity = {key: metadata.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise SystemExit(
            f"metadata identity mismatch for {video}: "
            f"{actual_identity} != {expected_identity}"
        )
    if int(metadata.get("reference_frame", -1)) != int(row["reference_frame"]):
        raise SystemExit(f"reference frame mismatch: {video}")
    if not math.isclose(
        float(metadata.get("reference_eval_return", "nan")),
        float(row["reference_return"]),
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise SystemExit(f"reference return mismatch: {video}")
    checkpoint_frame = int(metadata.get("checkpoint_frame", 0))
    if minimum_checkpoint_frame and checkpoint_frame < minimum_checkpoint_frame:
        raise SystemExit(
            f"checkpoint {checkpoint_frame} is below target "
            f"{minimum_checkpoint_frame}: {video}"
        )
    checkpoint = Path(str(metadata.get("checkpoint", "")))
    required_checkpoint_files = (
        checkpoint / "checkpoint.msgpack",
        checkpoint / "trainer_state.npz",
        checkpoint / "_SUCCESS",
    )
    if not all(
        path.is_file() and path.stat().st_size > 0
        for path in required_checkpoint_files
    ):
        raise SystemExit(f"checkpoint is incomplete: {checkpoint}")
    if int(metadata.get("video_frames", 0)) < 2:
        raise SystemExit(f"too few frames: {video}")
    recorded_video = Path(str(metadata.get("video", "")))
    if recorded_video.resolve() != video.resolve():
        raise SystemExit(f"video path mismatch in metadata: {video}")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"ffmpeg validation failed: {video}")
print(
    f"validated {len(rows)} manifest-bound MP4/JSON/checkpoint records "
    f"at checkpoint >= {minimum_checkpoint_frame}"
)
PY

log "SUCCESS: ${EXPECTED}/${EXPECTED} validated videos are under ${OUTPUT_ROOT}"
trap - EXIT INT TERM HUP
