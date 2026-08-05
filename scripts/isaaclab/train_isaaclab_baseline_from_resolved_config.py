#!/usr/bin/env python3
"""Train one baseline from its saved resolved config with resumable checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("FLOWRL_ISAACLAB_HEADLESS", "1")
os.environ.setdefault("FLOWRL_ISAACLAB_CLOSE_APP", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from examples.online.main_isaaclab_onpolicy import IsaacLabOnPolicyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-frames", type=int)
    parser.add_argument("--eval-frames", type=int)
    parser.add_argument("--log-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    identity = {
        "task": str(cfg.task),
        "algo": str(cfg.algo.name),
        "seed": int(cfg.seed),
    }
    expected = {"task": args.task, "algo": args.algo, "seed": args.seed}
    if identity != expected:
        raise ValueError(
            f"resolved config identity mismatch: {identity} != {expected}"
        )

    if args.algo == "policyflow":
        # PolicyFlow was uncommitted in the recorded run, and only its later
        # compatible source remains.  Preserve every saved field and fill only
        # newly required fields from the frozen overlay's default config.
        default_path = (
            REPO_ROOT
            / "examples"
            / "online"
            / "config"
            / "isaaclab_onpolicy"
            / "algo"
            / "policyflow.yaml"
        )
        defaults = OmegaConf.load(default_path)
        cfg.algo = OmegaConf.merge(defaults.algo, cfg.algo)

    cfg.device = "0"
    cfg.log.dir = str(args.training_root.resolve())
    cfg.log.project = "isaaclab-baseline-final-video-reproduction"
    cfg.log.group = "isaaclab-baseline-final-video-reproduction"
    cfg.log.name = args.run_name
    cfg.log.tag = args.run_tag
    cfg.log.wandb = False
    cfg.log.wandb_mode = "disabled"
    cfg.log.save_ckpt = True
    cfg.log.resume = True
    cfg.log.max_checkpoints = 2
    cfg.log.save_video = False
    if args.train_frames is not None:
        cfg.train_frames = args.train_frames
    if args.eval_frames is not None:
        cfg.eval_frames = args.eval_frames
    if args.log_frames is not None:
        cfg.log_frames = args.log_frames
    for name in ("train_frames", "eval_frames", "log_frames"):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name} must be positive")

    trainer = IsaacLabOnPolicyTrainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
