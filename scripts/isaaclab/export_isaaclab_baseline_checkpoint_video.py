#!/usr/bin/env python3
"""Render one deterministic IsaacLab baseline episode from a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ENABLE_CAMERAS", "1")
os.environ.setdefault("FLOWRL_ISAACLAB_HEADLESS", "1")
os.environ.setdefault("FLOWRL_ISAACLAB_CLOSE_APP", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from examples.online.main_isaaclab_onpolicy import IsaacLabOnPolicyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-eye", default="3.5,3.5,2.2")
    parser.add_argument("--camera-lookat", default="0.0,0.0,0.5")
    parser.add_argument("--reference-return", type=float)
    parser.add_argument("--reference-frame", type=int)
    parser.add_argument("--selection-status", default="")
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--historical-config", default="")
    parser.add_argument("--config-sha256", default="")
    parser.add_argument("--source-status", default="")
    return parser.parse_args()


def load_trainer_state(
    trainer: IsaacLabOnPolicyTrainer,
    checkpoint: Path,
) -> int:
    state_path = checkpoint / "trainer_state.npz"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    with np.load(state_path, allow_pickle=False) as state:
        checkpoint_frame = int(state["global_frame"])
        if bool(trainer.cfg.norm_obs):
            required = ("obs_mean", "obs_std", "obs_var", "obs_count")
            missing = [name for name in required if name not in state.files]
            if missing:
                raise ValueError(
                    f"{state_path} lacks normalizer fields: {missing}"
                )
            trainer.obs_normalizer.mean = np.asarray(
                state["obs_mean"], dtype=np.float32
            )
            trainer.obs_normalizer.std = np.asarray(
                state["obs_std"], dtype=np.float32
            )
            trainer.obs_normalizer.var = np.asarray(
                state["obs_var"], dtype=np.float32
            )
            trainer.obs_normalizer.count = float(state["obs_count"])
    return checkpoint_frame


def rgb_frame(raw_frame) -> np.ndarray:
    frame = np.asarray(raw_frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected rendered frame shape: {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if float(np.max(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0)
    return np.asarray(frame, dtype=np.uint8)


def enable_video_environment(
    *,
    width: int,
    height: int,
    camera_eye: str,
    camera_lookat: str,
) -> None:
    """Patch the old baseline wrapper to enable cameras without training changes."""
    from isaaclab.app import AppLauncher as original_launcher
    import isaaclab.app as isaaclab_app


    def camera_launcher(*args, **kwargs):
        kwargs["enable_cameras"] = True
        return original_launcher(*args, **kwargs)

    isaaclab_app.AppLauncher = camera_launcher

    import gymnasium as gym

    original_make = gym.make

    def tracking_make(env_id, *args, **kwargs):
        cfg = kwargs.get("cfg")
        viewer = getattr(cfg, "viewer", None)
        if viewer is not None:
            viewer.origin_type = "asset_root"
            viewer.asset_name = "robot"
            viewer.env_index = 0
            viewer.eye = tuple(float(value) for value in camera_eye.split(","))
            viewer.lookat = tuple(
                float(value) for value in camera_lookat.split(",")
            )
            viewer.resolution = (width, height)
        kwargs["render_mode"] = "rgb_array"
        return original_make(env_id, *args, **kwargs)

    gym.make = tracking_make


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    if not (checkpoint_path / "_SUCCESS").is_file():
        raise FileNotFoundError(checkpoint_path / "_SUCCESS")
    if args.steps <= 0 or args.fps <= 0:
        raise ValueError("--steps and --fps must be positive")

    enable_video_environment(
        width=args.width,
        height=args.height,
        camera_eye=args.camera_eye,
        camera_lookat=args.camera_lookat,
    )
    cfg = OmegaConf.load(config_path)
    cfg.device = str(args.device)
    cfg.num_envs = 1
    # Keep the algorithm's training-time num_envs.  Several agents use it to
    # build checkpointed state and to validate batch/minibatch divisibility.
    # Only the evaluation environment is scalar.

    cfg.eval.num_episodes = 1
    cfg.eval_frames = 0
    cfg.log.wandb = False
    cfg.log.save_ckpt = False
    cfg.log.save_video = False
    if "resume" in cfg.log:
        cfg.log.resume = False
    gpu_index = int(args.device)
    gpu_devices = jax.devices("gpu")
    if gpu_index < 0 or gpu_index >= len(gpu_devices):
        raise ValueError(
            f"physical GPU index {gpu_index} is unavailable to JAX "
            f"(found {len(gpu_devices)} devices)"
        )
    jax.config.update("jax_default_device", gpu_devices[gpu_index])

    cfg.log.dir = str(output_path.parent / "_video_eval_logs")
    cfg.log.tag = "checkpoint-video"
    cfg.log.name = "checkpoint-video"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = IsaacLabOnPolicyTrainer(cfg)
    writer = None
    try:
        trainer.agent.load(str(checkpoint_path))
        checkpoint_frame = load_trainer_state(trainer, checkpoint_path)

        env = trainer.env
        obs = np.asarray(env.reset(random_start_init=False), dtype=np.float32)
        episode_return = 0.0
        recorded_frames = 0

        writer = imageio.get_writer(
            output_path,
            fps=args.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        writer.append_data(rgb_frame(env.envs.render()))
        recorded_frames += 1

        for _ in range(min(args.steps, int(env.max_episode_steps))):
            normalized_obs = trainer._normalize_obs(obs)
            actions, _ = trainer.agent.sample_actions(
                jnp.asarray(normalized_obs),
                deterministic=True,
            )
            actions = np.asarray(jax.device_get(actions), dtype=np.float32)
            actions_to_env = np.clip(actions, -1.0, 1.0)
            obs, rewards, terminated, truncated, _ = env.step(actions_to_env)
            obs = np.asarray(obs, dtype=np.float32)
            episode_return += float(np.asarray(rewards)[0])
            writer.append_data(rgb_frame(env.envs.render()))
            recorded_frames += 1
            done = np.logical_or(terminated, truncated).astype(bool)
            if bool(done[0]):
                break

        writer.close()
        writer = None
        metadata = {
            "task": str(cfg.task),
            "algorithm": str(cfg.algo.name),
            "seed": int(cfg.seed),
            "checkpoint": str(checkpoint_path),
            "checkpoint_frame": checkpoint_frame,
            "config": str(config_path),
            "video": str(output_path),
            "video_frames": recorded_frames,
            "fps": args.fps,
            "resolution": [args.width, args.height],
            "rollout_return": episode_return,
            "reference_eval_return": args.reference_return,
            "historical_config": args.historical_config,
            "historical_config_sha256": args.config_sha256,
            "source_status": args.source_status,
            "reference_frame": args.reference_frame,
            "selection_status": args.selection_status,
            "source_run_id": args.source_run_id,
            "action_request_deterministic": True,
            "seeded_rollout": True,
            "stochastic_latent_possible": str(cfg.algo.name) in {
                "dppo",
                "policyflow",
            },
            "action_clip": [-1.0, 1.0],
            "headless": True,
            "camera_tracks": "robot",
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    finally:
        if writer is not None:
            writer.close()
        trainer.close()


if __name__ == "__main__":
    main()
