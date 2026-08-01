import os
import re
import shutil
import signal
import uuid

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
from flowrl.config.online.onpolicy_isaaclab_config import Config
from flowrl.dataset.buffer.state import EmpiricalNormalizer
from flowrl.env.online.isaaclab_env import IsaacLabEnv
from flowrl.types import Dict, RolloutBatch
from flowrl.utils.action_smoothness import compute_action_smoothness_metrics
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

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


class IsaacLabOnPolicyTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._closed = False
        self._stop_signal = None

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._request_stop)

        set_seed_everywhere(cfg.seed)
        wandb_project = _cfg_value(cfg.log, "project")
        wandb_entity = _cfg_value(cfg.log, "entity")
        if wandb_entity == "":
            wandb_entity = None
        run_name = f"{cfg.task}-{cfg.algo.name}-seed{cfg.seed}"
        wandb_name = _cfg_value(cfg.log, "name", run_name) or run_name
        wandb_group = _cfg_value(cfg.log, "group", cfg.log.tag) or cfg.log.tag
        wandb_mode = _cfg_value(cfg.log, "wandb_mode", os.environ.get("WANDB_MODE", "online"))
        wandb_tags = _cfg_value(cfg.log, "tags", [cfg.task, cfg.algo.name])
        if wandb_tags is not None:
            wandb_tags = list(wandb_tags)
        run_log_dir = "/".join(
            [cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task, wandb_name]
        )
        self.ckpt_save_dir = os.path.join(run_log_dir, "ckpt")
        self._resume_checkpoint = self._find_latest_checkpoint()
        wandb_run_id = self._find_wandb_run_id() if self._resume_checkpoint else None
        wandb_init_kwargs = {}
        if wandb_run_id is not None:
            wandb_init_kwargs = {"id": wandb_run_id, "resume": "allow"}
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            name=run_name,
            unique_name=wandb_name,
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {
                    "activate": bool(wandb_project) and bool(_cfg_value(cfg.log, "wandb", True)),
                    "config": OmegaConf.to_container(cfg),
                    "settings": wandb.Settings(_disable_stats=True),
                    "project": wandb_project,
                    "entity": wandb_entity,
                    "group": wandb_group,
                    "mode": wandb_mode,
                    "tags": wandb_tags,
                    "unique_name": wandb_name,
                    **wandb_init_kwargs,
                },
            },
        )
        if os.path.realpath(self.logger.log_dir) != os.path.realpath(run_log_dir):
            raise RuntimeError(
                f"Unexpected logger directory: {self.logger.log_dir} != {run_log_dir}"
            )
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("=" * 35 + " Config " + "=" * 35)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        # Create IsaacLab vectorized environment (returns numpy arrays)
        self.num_envs = cfg.num_envs
        self.rollout_length = cfg.rollout_length
        self.env = IsaacLabEnv(
            task=cfg.task,
            device="cuda:"+str(cfg.device),
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            action_bound=cfg.action_bound,
            disable_bootstrap=cfg.disable_bootstrap,
        )

        self.obs_dim = self.env.num_obs
        self.action_dim = self.env.num_actions
        self.max_episode_steps = self.env.max_episode_steps

        if cfg.norm_obs:
            self.obs_normalizer = EmpiricalNormalizer(shape=(self.obs_dim,))

        # Create agent
        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.obs_dim,
            act_dim=self.action_dim,
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        self.global_step = 0
        self.global_episode = 0
        self._restored = False
        if self._resume_checkpoint is not None:
            self._restore_checkpoint(self._resume_checkpoint)

    @property
    def global_frame(self) -> int:
        return self.global_step

    def _request_stop(self, signum, _frame) -> None:
        if self._stop_signal is None:
            self._stop_signal = signum
            print(
                f"Received signal {signum}; checkpointing at the next training boundary.",
                flush=True,
            )

    def _find_latest_checkpoint(self):
        if not bool(_cfg_value(self.cfg.log, "resume", False)):
            return None
        if not os.path.isdir(self.ckpt_save_dir):
            return None

        candidates = []
        for entry in os.scandir(self.ckpt_save_dir):
            if not entry.is_dir() or not entry.name.isdigit():
                continue
            if not os.path.isfile(os.path.join(entry.path, "_SUCCESS")):
                continue
            if not os.path.isfile(os.path.join(entry.path, "trainer_state.npz")):
                continue
            candidates.append((int(entry.name), entry.path))
        return max(candidates, default=(None, None))[1]

    def _find_wandb_run_id(self):
        latest_run = os.path.join(os.path.dirname(self.ckpt_save_dir), "wandb", "latest-run")
        if not os.path.exists(latest_run):
            return None
        match = re.fullmatch(
            r"run-\d{8}_\d{6}-(.+)",
            os.path.basename(os.path.realpath(latest_run)),
        )
        return match.group(1) if match else None

    def _save_checkpoint(self) -> None:
        if not self.cfg.log.save_ckpt:
            return

        frame = self.global_frame
        checkpoint_path = os.path.join(self.ckpt_save_dir, str(frame))
        success_file = os.path.join(checkpoint_path, "_SUCCESS")
        if os.path.isfile(success_file):
            return
        if os.path.exists(checkpoint_path):
            raise RuntimeError(f"Incomplete checkpoint already exists: {checkpoint_path}")

        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        temporary_path = os.path.join(
            self.ckpt_save_dir,
            f".{frame}.tmp-{os.getpid()}-{uuid.uuid4().hex}",
        )
        self.agent.save(temporary_path)

        state = {
            "checkpoint_version": np.asarray(1, dtype=np.int64),
            "global_frame": np.asarray(frame, dtype=np.int64),
            "task": np.asarray(self.cfg.task),
            "algo": np.asarray(self.cfg.algo.name),
            "seed": np.asarray(self.cfg.seed, dtype=np.int64),
            "agent_rng": np.asarray(jax.device_get(self.agent.rng)),
        }
        for model_name in self.agent.saved_model_names:
            model = getattr(self.agent, model_name)
            state[f"model_rng_{model_name}"] = np.asarray(
                jax.device_get(model.dropout_rng)
            )
        if self.cfg.norm_obs:
            state.update(
                {
                    "obs_mean": self.obs_normalizer.mean,
                    "obs_std": self.obs_normalizer.std,
                    "obs_var": self.obs_normalizer.var,
                    "obs_count": np.asarray(self.obs_normalizer.count),
                }
            )
        with open(os.path.join(temporary_path, "trainer_state.npz"), "wb") as state_file:
            np.savez(state_file, **state)
        with open(os.path.join(temporary_path, "_SUCCESS"), "w", encoding="utf-8") as marker:
            marker.write(f"frame={frame}\n")
        os.rename(temporary_path, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        max_checkpoints = int(_cfg_value(self.cfg.log, "max_checkpoints", 2))
        if max_checkpoints < 1:
            raise ValueError(f"log.max_checkpoints must be positive, got {max_checkpoints}")
        checkpoints = []
        for entry in os.scandir(self.ckpt_save_dir):
            if entry.is_dir() and entry.name.isdigit():
                checkpoints.append((int(entry.name), entry.path))
        for _, stale_path in sorted(checkpoints, reverse=True)[max_checkpoints:]:
            shutil.rmtree(stale_path)

    def _restore_checkpoint(self, checkpoint_path: str) -> None:
        with np.load(os.path.join(checkpoint_path, "trainer_state.npz")) as state:
            checkpoint_version = int(state["checkpoint_version"])
            if checkpoint_version != 1:
                raise ValueError(
                    f"Unsupported checkpoint version: {checkpoint_version}"
                )
            expected = {
                "task": self.cfg.task,
                "algo": self.cfg.algo.name,
                "seed": self.cfg.seed,
            }
            actual = {
                "task": str(state["task"]),
                "algo": str(state["algo"]),
                "seed": int(state["seed"]),
            }
            if actual != expected:
                raise ValueError(
                    f"Checkpoint identity mismatch: expected={expected}, actual={actual}"
                )
            checkpoint_frame = int(state["global_frame"])
            if checkpoint_frame != int(os.path.basename(checkpoint_path)):
                raise ValueError(
                    f"Checkpoint frame mismatch: path={checkpoint_path}, state={checkpoint_frame}"
                )

            self.agent.load(checkpoint_path)
            self.agent.rng = jnp.asarray(state["agent_rng"])
            for model_name in self.agent.saved_model_names:
                rng_key = f"model_rng_{model_name}"
                model = getattr(self.agent, model_name)
                setattr(
                    self.agent,
                    model_name,
                    model.replace(dropout_rng=jnp.asarray(state[rng_key])),
                )
            if self.cfg.norm_obs:
                self.obs_normalizer.mean = state["obs_mean"].copy()
                self.obs_normalizer.std = state["obs_std"].copy()
                self.obs_normalizer.var = state["obs_var"].copy()
                self.obs_normalizer.count = float(state["obs_count"])
            self.global_step = checkpoint_frame
        self._restored = True
        print(f"Restored checkpoint: {checkpoint_path}", flush=True)

    def _exit_if_stop_requested(self) -> None:
        if self._stop_signal is None:
            return
        self._save_checkpoint()
        raise SystemExit(128 + self._stop_signal)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.norm_obs:
            return self.obs_normalizer.normalize(obs)
        return obs

    def _assert_finite(self, name: str, value) -> None:
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
            summary_values = np.abs(finite_values) if np.iscomplexobj(finite_values) else finite_values
            value_summary = (
                f", finite_min={float(np.min(summary_values)):.6g},"
                f" finite_max={float(np.max(summary_values)):.6g}"
            )
        raise FloatingPointError(
            f"Non-finite values in {name} at frame {self.global_frame}: "
            f"shape={arr.shape}, bad={bad_count}/{arr.size}{value_summary}"
        )

    def _assert_finite_metrics(self, prefix: str, metrics) -> None:
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

    def collect_rollouts(self) -> RolloutBatch:
        """Collect rollouts from IsaacLab vectorized env."""
        T = self.rollout_length
        B = self.num_envs

        all_raw_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_actions = np.zeros((T, B, self.action_dim), dtype=np.float32)
        all_next_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_rewards = np.zeros((T, B, 1), dtype=np.float32)
        all_terminated = np.zeros((T, B, 1), dtype=np.float32)
        all_truncated = np.zeros((T, B, 1), dtype=np.float32)
        all_extras = None  # Lazily initialized from first sample_actions info

        for t in range(T):
            self._assert_finite("train/obs", self.obs)
            obs_norm = self._normalize_obs(self.obs)
            self._assert_finite("train/obs_norm", obs_norm)
            all_raw_obs[t] = self.obs.copy()
            all_obs[t] = obs_norm

            actions, info = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=False
            )
            actions_np = np.array(actions)
            self._assert_finite("train/actions", actions_np)
            self._assert_finite_metrics("train/action_info", info)
            actions_to_env = np.clip(actions_np, -1.0, 1.0)
            self._assert_finite("train/actions_to_env", actions_to_env)
            all_actions[t] = actions_np

            # Generically collect algorithm-specific info
            if all_extras is None:
                all_extras = {
                    k: np.zeros((T, *np.array(v).shape), dtype=np.float32)
                    for k, v in info.items()
                }
            for k, v in info.items():
                all_extras[k][t] = np.array(v)

            next_obs, rewards, terminated, truncated, infos = self.env.step(
                actions_to_env
            )
            self._assert_finite("train/next_obs", next_obs)
            self._assert_finite("train/rewards", rewards)
            self._assert_finite("train/terminated", terminated)
            self._assert_finite("train/truncated", truncated)

            all_rewards[t] = rewards[..., np.newaxis]
            all_terminated[t] = terminated[..., np.newaxis]
            all_truncated[t] = truncated[..., np.newaxis]

            next_obs_norm = self._normalize_obs(next_obs)
            self._assert_finite("train/next_obs_norm", next_obs_norm)
            all_next_obs[t] = next_obs_norm

            # Track episode stats
            self.ep_returns += rewards
            self.ep_lengths += 1

            done = (terminated + truncated) > 0.5
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                mean_return = np.mean(self.ep_returns[done_indices])
                mean_length = np.mean(self.ep_lengths[done_indices])
                episode_metrics = {
                    "rollout/episode_return": mean_return,
                    "rollout/episode_length": mean_length,
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
            env = getattr(self, "env", None)
            if env is not None:
                env.close()
        finally:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.close()

    def train(self):
        cfg = self.cfg
        last_log_frame = self.global_frame
        last_eval_frame = self.global_frame
        self.obs = self.env.reset(random_start_init=True)
        self._assert_finite("train/reset_obs", self.obs)
        self.ep_returns = np.zeros(self.num_envs)
        self.ep_lengths = np.zeros(self.num_envs)

        if not self._restored:
            self.eval_and_save()
        self._exit_if_stop_requested()
        with tqdm(
            total=cfg.train_frames,
            initial=self.global_frame,
            desc="training",
        ) as pbar:
            while self.global_frame < cfg.train_frames:
                prev_frame = self.global_frame
                rollout_data = self.collect_rollouts()
                self.global_step += self.rollout_length * self.num_envs
                metrics = self.agent.train_step(rollout_data, step=self.global_frame)
                self._assert_finite_metrics("agent_metrics", metrics)

                if self.global_frame - last_log_frame >= cfg.log_frames:
                    self.logger.log_scalars("", metrics, step=self.global_frame)
                    last_log_frame = self.global_frame

                if self.global_frame - last_eval_frame >= cfg.eval_frames:
                    self.eval_and_save()
                    last_eval_frame = self.global_frame

                pbar.update(self.global_frame - prev_frame)
                self._exit_if_stop_requested()
            self.eval_and_save()
        self.close()

    def eval_and_save(self):
        """Evaluate by running the policy for max_episode_steps in the same env."""
        obs = self.env.reset(random_start_init=False)
        self._assert_finite("eval/reset_obs", obs)
        eval_returns = np.zeros(self.num_envs)
        eval_lengths = np.zeros(self.num_envs)
        eval_dones = np.zeros(self.num_envs, dtype=bool)
        eval_actions = np.empty(
            (self.max_episode_steps, self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        num_eval_steps = 0

        for eval_step in range(self.max_episode_steps):
            obs_norm = self._normalize_obs(obs)
            self._assert_finite("eval/obs_norm", obs_norm)
            actions, _ = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=True
            )
            actions_np = np.array(actions)
            self._assert_finite("eval/actions", actions_np)
            actions_to_env = np.clip(actions_np, -1.0, 1.0)
            self._assert_finite("eval/actions_to_env", actions_to_env)
            eval_actions[eval_step] = actions_to_env
            num_eval_steps = eval_step + 1
            obs, rewards, terminated, truncated, _ = self.env.step(actions_to_env)
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
            self._save_checkpoint()

        # # Restore training state: reset with decorrelated horizons
        self.obs = self.env.reset(random_start_init=True)
        self._assert_finite("train/reset_obs_after_eval", self.obs)
        self.ep_returns = np.zeros(self.num_envs)
        self.ep_lengths = np.zeros(self.num_envs)


@hydra.main(
    config_path="./config/isaaclab_onpolicy",
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

    trainer = IsaacLabOnPolicyTrainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
