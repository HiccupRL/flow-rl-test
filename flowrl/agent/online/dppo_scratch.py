from functools import partial
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.online.dppo import DPPOAgent
from flowrl.config.online.algo.dppo_scratch import DPPOFromScratchConfig
from flowrl.flow.ddpm import DDPM
from flowrl.functional.activation import get_activation, mish
from flowrl.types import Metric, RolloutBatch


class ScratchDPPOResidualMLP(nn.Module):
    """Official residual layout with a zero-initialized output projection."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        del training
        if len(self.hidden_dims) % 2 == 0:
            raise ValueError(
                "Scratch DPPO needs an odd number of hidden layers "
                "(one input layer followed by two-layer residual blocks)."
            )
        if not self.hidden_dims or len(set(self.hidden_dims)) != 1:
            raise ValueError(
                "Scratch DPPO hidden dimensions must be non-empty and equal."
            )

        hidden_dim = self.hidden_dims[0]
        x = nn.Dense(hidden_dim)(x)
        for _ in range((len(self.hidden_dims) - 1) // 2):
            residual = x
            x = nn.Dense(hidden_dim)(self.activation(x))
            x = nn.Dense(hidden_dim)(self.activation(x))
            x = x + residual
        return nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(x)


class ScratchDPPODiffusionBackbone(nn.Module):
    """Noise predictor initialized as epsilon_theta(x_t, t, s) = x_t.

    A randomly initialized epsilon predictor produces almost entirely
    boundary-saturated actions with a short cosine DDPM.  The identity skip
    is the score of a standard-normal reference distribution.  The
    zero-initialized residual keeps the initial policy centered and remains
    fully trainable.
    """

    residual_predictor: nn.Module
    time_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        condition: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        half_dim = self.time_dim // 2
        frequency = jnp.exp(
            -jnp.log(10000.0)
            * jnp.arange(half_dim, dtype=jnp.float32)
            / (half_dim - 1)
        )
        time_embedding = time * frequency
        time_embedding = jnp.concatenate(
            [jnp.sin(time_embedding), jnp.cos(time_embedding)], axis=-1
        )
        time_embedding = nn.Dense(self.time_dim * 2)(time_embedding)
        time_embedding = mish(time_embedding)
        time_embedding = nn.Dense(self.time_dim)(time_embedding)
        inputs = jnp.concatenate([x, time_embedding, condition], axis=-1)
        residual = self.residual_predictor(inputs, training=training)
        return x + residual


@partial(jax.jit, static_argnames=("steps",))
def _scratch_action_diagnostics(
    action_chains: jnp.ndarray,
    steps: int,
) -> Metric:
    raw_actions = action_chains[..., steps, :]
    return {
        "misc/action_raw_abs_mean": jnp.abs(raw_actions).mean(),
        "misc/action_raw_std": raw_actions.std(),
        "misc/action_env_clip_fraction": (
            jnp.abs(raw_actions) > 1.0
        ).mean(),
        "misc/action_vector_clip_fraction": (
            jnp.abs(raw_actions).max(axis=-1) > 1.0
        ).mean(),
    }


class DPPOFromScratchAgent(DPPOAgent):
    """DPPO with stable scratch initialization and critic warmup.

    The official manipulation experiments fine-tune pretrained policies.
    This class is an explicit adaptation for the project's from-scratch
    on-policy protocol; ``DPPOAgent`` remains the official-style baseline.
    """

    name = "DPPOFromScratchAgent"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: DPPOFromScratchConfig,
        seed: int,
    ):
        if cfg.backbone_cls != "residual_mlp":
            raise ValueError(
                "dppo_scratch requires backbone_cls='residual_mlp' for its "
                "zero-initialized residual score network."
            )
        if cfg.critic_warmup_rollouts < 0:
            raise ValueError("critic_warmup_rollouts must be non-negative.")

        super().__init__(obs_dim, act_dim, cfg, seed)
        self.rng, actor_rng = jax.random.split(self.rng)
        actor_activation = get_activation(cfg.diffusion.activation)
        actor_network = ScratchDPPODiffusionBackbone(
            residual_predictor=ScratchDPPOResidualMLP(
                hidden_dims=cfg.diffusion.hidden_dims,
                output_dim=act_dim,
                activation=actor_activation,
            ),
            time_dim=cfg.diffusion.time_dim,
        )
        self.actor = DDPM.create(
            network=actor_network,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, act_dim)),
                jnp.zeros((1, 1), dtype=jnp.int32),
                jnp.ones((1, obs_dim)),
            ),
            x_dim=act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule,
            noise_schedule_params={},
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(
        self,
        rollout: RolloutBatch,
        step: int,
    ) -> Metric:
        diagnostics = _scratch_action_diagnostics(
            rollout.extras["action_chains"],
            self.cfg.diffusion.steps,
        )
        rollout_frames = self.cfg.num_envs * self.cfg.rollout_length
        warmup_frames = self.cfg.critic_warmup_rollouts * rollout_frames
        update_actor = step > warmup_frames

        actor_before_update = self.actor
        metrics = super().train_step(rollout, step)
        if not update_actor:
            # Keep critic and reward-scaler updates while discarding the actor
            # update and its optimizer state during the warmup rollouts.
            self.actor = actor_before_update

        metrics.update(diagnostics)
        metrics["misc/actor_update_enabled"] = jnp.asarray(
            update_actor, dtype=jnp.float32
        )
        return metrics
