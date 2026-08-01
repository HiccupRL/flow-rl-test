import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import hydra
import omegaconf

from examples.online.main_mujoco_onpolicy import (
    SUPPORTED_AGENTS,
    GymMujocoOnPolicyTrainer,
)
from flowrl.config.online.onpolicy_dmc_config import Config
from flowrl.env.online.dmc_env import DMControlEnv


DMC_RUN_SEED_STRIDE = 4096
DMC_EVAL_SEED_OFFSET = 1 << 31
DMC_MAX_RUN_SEED = DMC_EVAL_SEED_OFFSET // DMC_RUN_SEED_STRIDE


def _validate_dmc_seed_indices(run_seed: int, env_index: int) -> tuple[int, int]:
    run_seed = int(run_seed)
    env_index = int(env_index)
    if not 0 <= run_seed < DMC_MAX_RUN_SEED:
        raise ValueError(f"run_seed must be in [0, {DMC_MAX_RUN_SEED}), got {run_seed}")
    if not 0 <= env_index < DMC_RUN_SEED_STRIDE:
        raise ValueError(
            f"env_index must be in [0, {DMC_RUN_SEED_STRIDE}), got {env_index}"
        )
    return run_seed, env_index


def dmc_train_env_seed(run_seed: int, env_index: int) -> int:
    """Return a non-overlapping training-environment seed for a run."""
    run_seed, env_index = _validate_dmc_seed_indices(run_seed, env_index)
    return run_seed * DMC_RUN_SEED_STRIDE + env_index


def dmc_eval_env_seed(run_seed: int, env_index: int) -> int:
    """Return a non-overlapping evaluation seed, disjoint from training."""
    run_seed, env_index = _validate_dmc_seed_indices(run_seed, env_index)
    return DMC_EVAL_SEED_OFFSET + run_seed * DMC_RUN_SEED_STRIDE + env_index


def _make_env(
    task: str,
    seed: int,
    frame_skip: int,
    frame_stack: int,
    horizon: int,
):
    def thunk():
        return DMControlEnv(
            task=task,
            seed=seed,
            visual=False,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            horizon=horizon,
        )

    return thunk


def _make_vector_env(
    env_fns,
    mode: str,
    shared_memory: bool,
    context: str | None,
):
    autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
    mode = mode.lower()
    if mode == "sync":
        return gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=autoreset_mode,
        )
    if mode == "async":
        if context == "":
            context = None
        return gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=shared_memory,
            context=context,
            autoreset_mode=autoreset_mode,
        )
    raise ValueError(f"Unsupported env_mode={mode!r}; expected 'sync' or 'async'.")


class DMControlOnPolicyTrainer(GymMujocoOnPolicyTrainer):
    benchmark = "dmc"

    @property
    def global_frame(self) -> int:
        return self.cfg.frame_skip * self.global_step

    def _make_train_env(self):
        cfg = self.cfg
        return _make_vector_env(
            env_fns=[
                _make_env(
                    task=cfg.task,
                    seed=dmc_train_env_seed(cfg.seed, env_index),
                    frame_skip=cfg.frame_skip,
                    frame_stack=cfg.frame_stack,
                    horizon=cfg.horizon,
                )
                for env_index in range(cfg.num_envs)
            ],
            mode=cfg.env_mode,
            shared_memory=cfg.async_shared_memory,
            context=cfg.async_context,
        )

    def _make_eval_env(self):
        cfg = self.cfg
        return gym.vector.SyncVectorEnv(
            [
                _make_env(
                    task=cfg.task,
                    seed=dmc_eval_env_seed(cfg.seed, env_index),
                    frame_skip=cfg.frame_skip,
                    frame_stack=cfg.frame_stack,
                    horizon=cfg.horizon,
                )
                for env_index in range(cfg.eval.num_episodes)
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    def _reset_train_env(self):
        seeds = [
            dmc_train_env_seed(self.cfg.seed, env_index)
            for env_index in range(self.num_envs)
        ]
        obs, _ = self.train_env.reset(seed=seeds)
        return obs

    def _reset_eval_env(self):
        seeds = [
            dmc_eval_env_seed(self.cfg.seed, env_index)
            for env_index in range(self.cfg.eval.num_episodes)
        ]
        obs, _ = self.eval_env.reset(seed=seeds)
        return obs

    def _resolve_max_episode_steps(self) -> int:
        if self.cfg.max_episode_steps is not None:
            return int(self.cfg.max_episode_steps)
        return int(self.eval_env.envs[0].max_ep_timesteps)


@hydra.main(
    config_path="./config/dmc_onpolicy",
    config_name="config",
    version_base=None,
)
def main(cfg: Config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        _ = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = (
            "Algorithm is not specified. Please specify it via "
            "`algo=<algo_name>` in command.\nAvailable algorithms are:\n  "
        )
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        raise SystemExit(1)

    trainer = DMControlOnPolicyTrainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
