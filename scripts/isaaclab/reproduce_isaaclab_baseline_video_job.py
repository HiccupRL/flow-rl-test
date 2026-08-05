#!/usr/bin/env python3
"""Retrain one IsaacLab baseline seed and export a validated final MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import imageio_ffmpeg


ACTIVE_PROCESS: subprocess.Popen | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-status", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--reference-return", type=float, required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--selection-status", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-frames", type=int)
    parser.add_argument("--eval-frames", type=int)
    parser.add_argument("--log-frames", type=int)
    parser.add_argument("--render-timeout-seconds", type=int, default=1800)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def forward_signal(signum, _frame) -> None:
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.send_signal(signum)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int | None = None,
) -> None:
    global ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMMAND "
            + shlex.join(command)
            + "\n"
        )
        stream.flush()
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            return_code = ACTIVE_PROCESS.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            ACTIVE_PROCESS.terminate()
            try:
                ACTIVE_PROCESS.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ACTIVE_PROCESS.kill()
                ACTIVE_PROCESS.wait()
            raise RuntimeError(
                f"command exceeded timeout of {timeout_seconds}s: "
                + shlex.join(command)
            ) from error
        finally:
            ACTIVE_PROCESS = None
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def archive_incomplete(path: Path) -> None:
    if not path.exists():
        return
    archive = path.with_name(
        f"{path.name}.incomplete.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    suffix = 1
    while archive.exists():
        archive = path.with_name(f"{archive.name}.{suffix}")
        suffix += 1
    path.rename(archive)


def video_valid(video: Path) -> bool:
    metadata = video.with_suffix(".json")
    if not video.is_file() or video.stat().st_size == 0:
        return False
    if not metadata.is_file() or metadata.stat().st_size == 0:
        return False
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(video),
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


def validate_historical_config(
    path: Path,
    *,
    task: str,
    algo: str,
    seed: int,
    expected_sha256: str,
    train_frames: int | None,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"resolved config SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"task": task, "algo": algo, "seed": seed}
    actual = {
        "task": str(config.get("task", "")),
        "algo": str(config.get("algo", {}).get("name", "")),
        "seed": int(config.get("seed", -1)),
    }
    if actual != expected:
        raise ValueError(
            f"resolved config identity mismatch: {actual} != {expected}"
        )
    target_frames = (
        int(train_frames)
        if train_frames is not None
        else int(config.get("train_frames", 0))
    )
    if target_frames <= 0:
        raise ValueError(f"invalid train_frames={target_frames}")
    return target_frames


def latest_valid_checkpoint(
    checkpoint_root: Path,
    *,
    minimum_frame: int,
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            frame = int(path.name)
            if frame < minimum_frame:
                continue
            if not (path / "_SUCCESS").is_file():
                continue
            if not (path / "trainer_state.npz").is_file():
                continue
            candidates.append((frame, path))
    return max(candidates, default=(0, None))[1]


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    metadata_path = args.metadata.resolve()
    historical_config = args.historical_config.resolve()
    output = args.output_dir.resolve()
    training_root = output / "training_logs"
    run_tag = f"{args.run_label}-training"
    run_name = f"reproduce_{args.slug}_seed{args.seed}"
    run_dir = (
        training_root
        / args.algo
        / run_tag
        / args.task
        / run_name
    )
    checkpoint_root = run_dir / "ckpt"
    config = run_dir / "config.yaml"
    video = output.parent.parent / f"{args.slug}_seed{args.seed}_best_final.mp4"
    train_log = output / "train.log"
    video_log = output / "video_export.log"

    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_commit = str(metadata.get("git", {}).get("commit", "")).strip()
    if (
        args.commit not in {"", "worktree-uncommitted"}
        and metadata_commit
        and metadata_commit != args.commit
    ):
        raise ValueError(
            f"Git commit mismatch: manifest={args.commit}, "
            f"metadata={metadata_commit}"
        )
    target_frames = validate_historical_config(
        historical_config,
        task=args.task,
        algo=args.algo,
        seed=args.seed,
        expected_sha256=args.config_sha256,
        train_frames=args.train_frames,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "EGL_VISIBLE_DEVICES": str(args.gpu),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "FLOWRL_ISAACLAB_HEADLESS": "1",
            "FLOWRL_ISAACLAB_CLOSE_APP": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(source)
            + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
        }
    )
    train_command = [
        args.python,
        "scripts/isaaclab/train_isaaclab_baseline_from_resolved_config.py",
        "--config",
        str(historical_config),
        "--training-root",
        str(training_root),
        "--run-tag",
        run_tag,
        "--run-name",
        run_name,
        "--task",
        args.task,
        "--algo",
        args.algo,
        "--seed",
        str(args.seed),
    ]
    for flag, value in (
        ("--train-frames", args.train_frames),
        ("--eval-frames", args.eval_frames),
        ("--log-frames", args.log_frames),
    ):
        if value is not None:
            train_command.extend((flag, str(value)))


    if args.render_timeout_seconds <= 0:
        raise ValueError("--render-timeout-seconds must be positive")
    if video_valid(video):
        print(f"validated existing video: {video}")
        return

    checkpoint = latest_valid_checkpoint(
        checkpoint_root,
        minimum_frame=target_frames,
    )
    if checkpoint is None or not config.is_file():
        partial_checkpoint = latest_valid_checkpoint(
            checkpoint_root,
            minimum_frame=0,
        )
        if partial_checkpoint is None:
            archive_incomplete(training_root)
        run_logged(
            train_command,
            cwd=source,
            env=environment,
            log_path=train_log,
        )
        checkpoint = latest_valid_checkpoint(
            checkpoint_root,
            minimum_frame=target_frames,
        )
    if checkpoint is None or not config.is_file():
        raise RuntimeError(
            f"training did not create a final checkpoint under {checkpoint_root}"
        )

    if video.exists() or video.with_suffix(".json").exists():
        archive_root = output / "incomplete_video_exports"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for path in (video, video.with_suffix(".json")):
            if path.exists():
                shutil.move(str(path), archive_root / f"{stamp}_{path.name}")

    render_environment = environment.copy()
    render_environment.pop("CUDA_VISIBLE_DEVICES", None)
    render_environment.pop("EGL_VISIBLE_DEVICES", None)
    render_environment["ENABLE_CAMERAS"] = "1"
    run_logged(
        [
            args.python,
            "scripts/isaaclab/export_isaaclab_baseline_checkpoint_video.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(video),
            "--device",
            str(args.gpu),
            "--steps",
            str(args.steps),
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--reference-return",
            str(args.reference_return),
            "--reference-frame",
            str(args.reference_frame),
            "--selection-status",
            args.selection_status,
            "--source-run-id",
            args.source_run_id,
            "--historical-config",
            str(historical_config),
            "--config-sha256",
            args.config_sha256,
            "--source-status",
            args.source_status,
        ],
        cwd=source,
        env=render_environment,
        log_path=video_log,
        timeout_seconds=args.render_timeout_seconds,
    )
    if not video_valid(video):
        raise RuntimeError(f"ffmpeg validation failed: {video}")

    result = {
        "slug": args.slug,
        "task": args.task,
        "algorithm": args.algo,
        "seed": args.seed,
        "reference_return": args.reference_return,
        "reference_frame": args.reference_frame,
        "selection_status": args.selection_status,
        "source_status": args.source_status,
        "source_run_id": args.source_run_id,
        "metadata_commit": metadata_commit,
        "source_metadata": str(metadata_path),
        "historical_config": str(historical_config),
        "historical_config_sha256": args.config_sha256,
        "checkpoint": str(checkpoint),
        "video": str(video),
        "status": "success",
    }
    (output / "success.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(handled_signal, forward_signal)
    main()
