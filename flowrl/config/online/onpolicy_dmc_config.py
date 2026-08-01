from dataclasses import dataclass

from .onpolicy_mujoco_config import Config as OnPolicyConfig


@dataclass
class Config(OnPolicyConfig):
    frame_skip: int
    frame_stack: int
    horizon: int
