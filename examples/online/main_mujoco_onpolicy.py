import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from flowrl.agent.online import (
    DPPOAgent,
    DPPOFromScratchAgent,
    FPOAgent,
    FPOPPAgent,
    GenPOAgent,
    PPOAgent,
    PolicyFlowAgent,
)
from flowrl.config.online.onpolicy_mujoco_config import Config
from flowrl.dataset.buffer.state import EmpiricalNormalizer
from flowrl.types import Dict, RolloutBatch
from flowrl.utils.action_smoothness import compute_action_smoothness_metrics
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

EVAL_ACTION_SEED_OFFSET = 1_000_000

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, type] = {
    "ppo": PPOAgent,
    "dppo": DPPOAgent,
    "dppo_scratch": DPPOFromScratchAgent,
    "fpo": FPOAgent,
    "fpopp": FPOPPAgent,
    "genpo": GenPOAgent,
    "policyflow": PolicyFlowAgent,
}


def _cfg_value(node, key, default=None):
    try:
        if OmegaConf.is_missing(node, key):
            return default
        value = getattr(node, key)
    except Exception:
        return default
    if value == "???":
        return default
    return value


def _make_env(task: str):
    def thunk():
        env = gym.make(task)
        return gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    return thunk


def _make_vector_env(
    task: str,
    num_envs: int,
    mode: str,
    shared_memory: bool,
    context: str | None,
):
    env_fns = [_make_env(task) for _ in range(num_envs)]
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


class GymMujocoOnPolicyTrainer:
    benchmark = "mujoco"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._closed = False
        self.debug_finite_checks = bool(cfg.debug_finite_checks)

        set_seed_everywhere(cfg.seed)
        wandb_project = _cfg_value(cfg.log, "project")
        wandb_entity = _cfg_value(cfg.log, "entity")
        if wandb_entity == "":
            wandb_entity = None
        run_name = f"{cfg.task}-{cfg.algo.name}-seed{cfg.seed}"
        wandb_name = _cfg_value(cfg.log, "name", run_name)
        wandb_group = _cfg_value(cfg.log, "group", cfg.log.tag)
        wandb_mode = _cfg_value(
            cfg.log, "wandb_mode", os.environ.get("WANDB_MODE", "online")
        )
        wandb_tags = _cfg_value(
            cfg.log,
            "tags",
            [cfg.task, cfg.algo.name, self.benchmark, "onpolicy"],
        )
        if wandb_tags is not None:
            wandb_tags = list(wandb_tags)
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            name=run_name,
            unique_name=wandb_name,
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {
                    "activate": bool(wandb_project)
                    and bool(_cfg_value(cfg.log, "wandb", True)),
                    "config": OmegaConf.to_container(cfg),
                    "settings": wandb.Settings(_disable_stats=True),
                    "project": wandb_project,
                    "entity": wandb_entity,
                    "group": wandb_group,
                    "mode": wandb_mode,
                    "tags": wandb_tags,
                    "unique_name": wandb_name,
                },
            },
        )
        self.ckpt_save_dir = os.path.join(self.logger.log_dir, "ckpt")
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("=" * 35 + " Config " + "=" * 35)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        self.num_envs = cfg.num_envs
        self.rollout_length = cfg.rollout_length
        self.rollout_frames = self.num_envs * self.rollout_length
        total_minibatch = cfg.algo.num_minibatches * cfg.algo.batch_size
        minibatch_population = self.rollout_frames
        population_name = "num_envs * rollout_length"
        assert total_minibatch <= minibatch_population, (
            f"num_minibatches * batch_size ({total_minibatch}) must be <= "
            f"{population_name} ({minibatch_population})"
        )
        self.train_env = self._make_train_env()
        self.eval_env = self._make_eval_env()

        self.obs_dim = self.train_env.single_observation_space.shape[-1]
        self.action_dim = self.train_env.single_action_space.shape[-1]
        self.max_episode_steps = self._resolve_max_episode_steps()

        if cfg.norm_obs:
            self.obs_normalizer = EmpiricalNormalizer(shape=(self.obs_dim,))

        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.obs_dim,
            act_dim=self.action_dim,
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        self.global_step = 0

    def _make_train_env(self):
        return _make_vector_env(
            task=self.cfg.task,
            num_envs=self.cfg.num_envs,
            mode=self.cfg.env_mode,
            shared_memory=self.cfg.async_shared_memory,
            context=self.cfg.async_context,
        )

    def _make_eval_env(self):
        return gym.vector.SyncVectorEnv(
            [_make_env(self.cfg.task) for _ in range(self.cfg.eval.num_episodes)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    @property
    def global_frame(self) -> int:
        return self.global_step

    def _resolve_max_episode_steps(self) -> int:
        if self.cfg.max_episode_steps is not None:
            return int(self.cfg.max_episode_steps)
        env = self.eval_env.envs[0]
        spec = getattr(env, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
            return int(spec.max_episode_steps)
        unwrapped_spec = getattr(getattr(env, "unwrapped", None), "spec", None)
        if (
            unwrapped_spec is not None
            and getattr(unwrapped_spec, "max_episode_steps", None) is not None
        ):
            return int(unwrapped_spec.max_episode_steps)
        return 1000

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.norm_obs:
            return self.obs_normalizer.normalize(obs)
        return obs

    def _reset_train_env(self) -> np.ndarray:
        seeds = [self.cfg.seed + i for i in range(self.num_envs)]
        obs, _ = self.train_env.reset(seed=seeds)
        return np.asarray(obs, dtype=np.float32)

    def _reset_eval_env(self) -> np.ndarray:
        seeds = [self.cfg.seed + 100_000 + i for i in range(self.cfg.eval.num_episodes)]
        obs, _ = self.eval_env.reset(seed=seeds)
        return np.asarray(obs, dtype=np.float32)

    def _assert_finite(self, name: str, value) -> None:
        if not self.debug_finite_checks:
            return
        try:
            arr = np.asarray(jax.device_get(value))
        except Exception as exc:
            raise FloatingPointError(
                f"Could not materialize {name} for finite check at frame {self.global_frame}: {exc}"
            ) from exc

        if arr.dtype.kind not in "buifc":
            return

        finite = np.isfinite(arr)
        if np.all(finite):
            return

        bad_count = int(arr.size - np.count_nonzero(finite))
        finite_values = arr[finite]
        value_summary = ""
        if finite_values.size > 0:
            summary_values = (
                np.abs(finite_values)
                if np.iscomplexobj(finite_values)
                else finite_values
            )
            value_summary = (
                f", finite_min={float(np.min(summary_values)):.6g},"
                f" finite_max={float(np.max(summary_values)):.6g}"
            )
        raise FloatingPointError(
            f"Non-finite values in {name} at frame {self.global_frame}: "
            f"shape={arr.shape}, bad={bad_count}/{arr.size}{value_summary}"
        )

    def _assert_finite_metrics(self, prefix: str, metrics) -> None:
        if not self.debug_finite_checks:
            return
        if metrics is None:
            return
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                if isinstance(value, dict):
                    self._assert_finite_metrics(name, value)
                else:
                    self._assert_finite(name, value)
            return
        self._assert_finite(prefix, metrics)

    def _actual_next_obs(
        self,
        next_obs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: dict,
    ) -> np.ndarray:
        actual_next_obs = np.asarray(next_obs, dtype=np.float32).copy()
        done_indices = np.where(np.logical_or(terminated, truncated))[0]
        final_obs = infos.get("final_obs") if isinstance(infos, dict) else None
        if final_obs is None:
            return actual_next_obs
        for i in done_indices:
            value = final_obs[i]
            if value is not None:
                actual_next_obs[i] = np.asarray(value, dtype=np.float32)
        return actual_next_obs

    def collect_rollouts(self) -> RolloutBatch:
        T = self.rollout_length
        B = self.num_envs

        all_raw_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_actions = np.zeros((T, B, self.action_dim), dtype=np.float32)
        all_next_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_rewards = np.zeros((T, B, 1), dtype=np.float32)
        all_terminated = np.zeros((T, B, 1), dtype=np.float32)
        all_truncated = np.zeros((T, B, 1), dtype=np.float32)
        all_extras = None
        action_clip_count = 0
        action_count = 0
        action_clip_delta_abs_max = 0.0
        raw_action_abs_max = 0.0
        executed_action_abs_max = 0.0

        for t in range(T):
            self._assert_finite("train/obs", self.obs)
            obs_norm = np.asarray(self._normalize_obs(self.obs), dtype=np.float32)
            self._assert_finite("train/obs_norm", obs_norm)
            all_raw_obs[t] = self.obs.copy()
            all_obs[t] = obs_norm

            actions, info = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=False
            )
            actions_np = np.asarray(actions, dtype=np.float32)
            self._assert_finite("train/actions", actions_np)
            self._assert_finite_metrics("train/action_info", info)
            actions_clipped = np.clip(actions_np, -1.0, 1.0).astype(np.float32)
            self._assert_finite("train/actions_clipped", actions_clipped)
            clip_delta = np.abs(actions_np - actions_clipped)
            action_clip_count += int(np.count_nonzero(clip_delta > 1e-6))
            action_count += int(clip_delta.size)
            action_clip_delta_abs_max = max(
                action_clip_delta_abs_max, float(np.max(clip_delta))
            )
            raw_action_abs_max = max(
                raw_action_abs_max, float(np.max(np.abs(actions_np)))
            )
            executed_action_abs_max = max(
                executed_action_abs_max, float(np.max(np.abs(actions_clipped)))
            )
            all_actions[t] = actions_clipped

            if all_extras is None:
                all_extras = {
                    k: np.zeros((T, *np.asarray(v).shape), dtype=np.float32)
                    for k, v in info.items()
                }
            for k, v in info.items():
                all_extras[k][t] = np.asarray(v, dtype=np.float32)

            next_obs, rewards, terminated, truncated, infos = self.train_env.step(
                actions_clipped
            )
            next_obs = np.asarray(next_obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=np.float32)
            truncated = np.asarray(truncated, dtype=np.float32)
            actual_next_obs = self._actual_next_obs(
                next_obs, terminated.astype(bool), truncated.astype(bool), infos
            )

            self._assert_finite("train/next_obs", next_obs)
            self._assert_finite("train/actual_next_obs", actual_next_obs)
            self._assert_finite("train/rewards", rewards)
            self._assert_finite("train/terminated", terminated)
            self._assert_finite("train/truncated", truncated)

            all_rewards[t] = rewards[..., np.newaxis]
            all_terminated[t] = terminated[..., np.newaxis]
            all_truncated[t] = truncated[..., np.newaxis]

            next_obs_norm = np.asarray(
                self._normalize_obs(actual_next_obs), dtype=np.float32
            )
            self._assert_finite("train/next_obs_norm", next_obs_norm)
            all_next_obs[t] = next_obs_norm

            self.ep_returns += rewards
            self.ep_lengths += 1
            done = (terminated + truncated) > 0.5
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                episode_metrics = {
                    "rollout/episode_return": np.mean(self.ep_returns[done_indices]),
                    "rollout/episode_length": np.mean(self.ep_lengths[done_indices]),
                    "rollout/num_completed": len(done_indices),
                }
                self._assert_finite_metrics("", episode_metrics)
                self.logger.log_scalars("", episode_metrics, step=self.global_frame)
                self.ep_returns[done_indices] = 0.0
                self.ep_lengths[done_indices] = 0

            self.obs = next_obs

        if self.cfg.norm_obs:
            self.obs_normalizer.update(all_raw_obs.reshape(-1, self.obs_dim))

        self._assert_finite("rollout/obs_batch", all_obs)
        self._assert_finite("rollout/actions_batch", all_actions)
        self._assert_finite("rollout/next_obs_batch", all_next_obs)
        self._assert_finite("rollout/rewards_batch", all_rewards)
        self.rollout_metrics = {
            "rollout/action_clip_fraction": action_clip_count / max(action_count, 1),
            "rollout/action_clip_delta_abs_max": action_clip_delta_abs_max,
            "rollout/raw_action_abs_max": raw_action_abs_max,
            "rollout/executed_action_abs_max": executed_action_abs_max,
        }

        return RolloutBatch(
            obs=jnp.array(all_obs),
            actions=jnp.array(all_actions),
            next_obs=jnp.array(all_next_obs),
            rewards=jnp.array(all_rewards),
            terminated=jnp.array(all_terminated),
            truncated=jnp.array(all_truncated),
            extras={k: jnp.array(v) for k, v in all_extras.items()},
        )

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            train_env = getattr(self, "train_env", None)
            if train_env is not None:
                train_env.close()
            eval_env = getattr(self, "eval_env", None)
            if eval_env is not None:
                eval_env.close()
        finally:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.close()

    def train(self):
        cfg = self.cfg
        last_log_frame = 0
        last_eval_frame = 0
        self.obs = self._reset_train_env()
        self._assert_finite("train/reset_obs", self.obs)
        self.ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.num_envs, dtype=np.float32)

        self.eval_and_save()
        with tqdm(total=cfg.train_frames, desc="training") as pbar:
            while self.global_frame < cfg.train_frames:
                prev_frame = self.global_frame
                rollout_data = self.collect_rollouts()
                self.global_step += self.rollout_frames
                metrics = dict(
                    self.agent.train_step(rollout_data, step=self.global_frame)
                )
                metrics.update(self.rollout_metrics)
                self._assert_finite_metrics("agent_metrics", metrics)

                if self.global_frame - last_log_frame >= cfg.log_frames:
                    self.logger.log_scalars("", metrics, step=self.global_frame)
                    last_log_frame = self.global_frame

                if self.global_frame - last_eval_frame >= cfg.eval_frames:
                    self.eval_and_save()
                    last_eval_frame = self.global_frame

                pbar.update(self.global_frame - prev_frame)
            if last_eval_frame != self.global_frame:
                self.eval_and_save()
        self.close()

    def eval_and_save(self):
        training_rng = self.agent.rng
        self.agent.rng = jax.random.PRNGKey(
            EVAL_ACTION_SEED_OFFSET + int(self.cfg.seed)
        )
        try:
            self._eval_and_save_impl()
        finally:
            self.agent.rng = training_rng

    def _eval_and_save_impl(self):
        obs = self._reset_eval_env()
        self._assert_finite("eval/reset_obs", obs)
        eval_returns = np.zeros(self.cfg.eval.num_episodes, dtype=np.float32)
        eval_lengths = np.zeros(self.cfg.eval.num_episodes, dtype=np.float32)
        eval_dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)
        eval_actions = np.empty(
            (
                self.max_episode_steps,
                self.cfg.eval.num_episodes,
                self.action_dim,
            ),
            dtype=np.float32,
        )
        num_eval_steps = 0

        for eval_step in range(self.max_episode_steps):
            obs_norm = np.asarray(self._normalize_obs(obs), dtype=np.float32)
            self._assert_finite("eval/obs_norm", obs_norm)
            actions, _ = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=True
            )
            actions_np = np.asarray(actions, dtype=np.float32)
            self._assert_finite("eval/actions", actions_np)
            actions_clipped = np.clip(actions_np, -1.0, 1.0).astype(np.float32)
            self._assert_finite("eval/actions_clipped", actions_clipped)
            eval_actions[eval_step] = actions_clipped
            num_eval_steps = eval_step + 1
            obs, rewards, terminated, truncated, _ = self.eval_env.step(actions_clipped)
            obs = np.asarray(obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=np.float32)
            truncated = np.asarray(truncated, dtype=np.float32)
            self._assert_finite("eval/next_obs", obs)
            self._assert_finite("eval/rewards", rewards)
            self._assert_finite("eval/terminated", terminated)
            self._assert_finite("eval/truncated", truncated)

            eval_returns += rewards * (1 - eval_dones)
            eval_lengths += 1 * (1 - eval_dones)
            eval_dones = eval_dones | ((terminated + truncated) > 0.5)
            if np.all(eval_dones):
                break

        eval_metrics = {
            "mean": np.mean(eval_returns),
            "median": np.median(eval_returns),
            "std": np.std(eval_returns),
            "min": np.min(eval_returns),
            "max": np.max(eval_returns),
            "length": np.mean(eval_lengths),
        }
        eval_metrics.update(
            compute_action_smoothness_metrics(
                eval_actions[:num_eval_steps],
                eval_lengths,
            )
        )
        self._assert_finite_metrics("eval", eval_metrics)
        self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, f"{self.global_frame}"))


@hydra.main(
    config_path="./config/mujoco_onpolicy",
    config_name="config",
    version_base=None,
)
def main(cfg: Config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        _ = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = "Algorithm is not specified. Please specify the algorithm via `algo=<algo_name>` in command."
        err_string += "\nAvailable algorithms are:\n  "
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        exit(1)

    trainer = GymMujocoOnPolicyTrainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
