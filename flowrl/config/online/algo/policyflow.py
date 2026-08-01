from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class PolicyFlowFlowConfig:
    activation: str
    hidden_dims: List[int]
    time_dim: int
    steps: int
    clip_sampler: bool
    lr: float
    x_min: float
    x_max: float
    output_scale: float
    init_logstd: float
    logstd_min: float
    logstd_max: float
    interpolation_type: str
    time_sampling: str
    architecture: str
    zero_init_output: bool
    output_init_scale: float
    fourier_scale: float
    eval_zero_x0: bool
    eval_sample_delta: bool


@dataclass
class PolicyFlowConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    gamma: float
    gae_lambda: float
    ratio_clip: float
    reward_scaling: float
    normalize_advantage: bool
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
    gaussian_entropy_loss_scale: float
    brownian_reg_loss_scale: float
    brownian_reduction: str
    value_loss_scale: float
    clip_predicted_values: bool
    value_clip: float
    value_clip_mode: str
    log_ratio_clip: float
    weight_decay: float
    time_limit_bootstrap: bool
    critic_output_init_scale: float
    adaptive_learning_rate: bool
    desired_kl: float
    learning_rate_min: float
    learning_rate_max: float
    learning_rate_factor: float
    flow: PolicyFlowFlowConfig
