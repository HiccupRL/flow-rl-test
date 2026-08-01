from dataclasses import dataclass

from .dppo import DPPOConfig


@dataclass
class DPPOFromScratchConfig(DPPOConfig):
    """DPPO adaptation for training without a pretrained diffusion policy."""

    critic_warmup_rollouts: int = 2
