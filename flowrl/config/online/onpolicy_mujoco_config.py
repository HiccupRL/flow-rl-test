from dataclasses import dataclass
from typing import Any, Optional

from .hb_config import EvalConfig, LogConfig


@dataclass
class Config:
    seed: int
    device: str
    task: str
    algo: Any
    norm_obs: bool
    train_frames: int
    eval_frames: int
    log_frames: int
    num_envs: int
    rollout_length: int
    env_mode: str
    async_shared_memory: bool
    async_context: Optional[str]
    debug_finite_checks: bool
    max_episode_steps: Optional[int]
    log: LogConfig
    eval: EvalConfig
