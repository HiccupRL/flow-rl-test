from functools import partial
from typing import Callable, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.ppo import compute_gae
from flowrl.config.online.algo.dppo import DPPOConfig
from flowrl.flow.ddpm import DDPM
from flowrl.functional.activation import get_activation, mish
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.types import Metric, PRNGKey, RolloutBatch


class DPPOResidualMLP(nn.Module):
    """Residual MLP used by the official DPPO scratch configuration."""

    hidden_dims: Sequence[int]
    output_dim: int = 0
    activation: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        del training
        if not self.hidden_dims:
            if self.output_dim > 0:
                return nn.Dense(self.output_dim)(x)
            return x
        if len(self.hidden_dims) % 2 == 0:
            raise ValueError(
                "DPPO residual MLP needs an odd number of hidden layers "
                "(one input layer followed by two-layer residual blocks)."
            )
        if len(set(self.hidden_dims)) != 1:
            raise ValueError("DPPO residual MLP hidden dimensions must be equal.")

        hidden_dim = self.hidden_dims[0]
        x = nn.Dense(hidden_dim)(x)
        for _ in range((len(self.hidden_dims) - 1) // 2):
            residual = x
            x = nn.Dense(hidden_dim)(self.activation(x))
            x = nn.Dense(hidden_dim)(self.activation(x))
            x = x + residual
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim)(x)
        return x


class DPPODiffusionBackbone(nn.Module):
    """Official DPPO MLP time-conditioning layout."""

    noise_predictor: nn.Module
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
        return self.noise_predictor(inputs, training=training)


class RunningRewardScalerNetwork(nn.Module):
    """Checkpointable state for DPPO's discounted-return reward scaler."""

    num_envs: int

    @nn.compact
    def __call__(self):
        mean = self.param("mean", nn.initializers.zeros, ())
        var = self.param("var", nn.initializers.ones, ())
        count = self.param(
            "count", lambda _: jnp.asarray(1e-4, dtype=jnp.float32)
        )
        return_accumulator = self.param(
            "return_accumulator",
            nn.initializers.zeros,
            (self.num_envs, 1),
        )
        return mean, var, count, return_accumulator


def clipped_gaussian_mean_log_prob(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """DPPO log-probability averaged over action dimensions."""
    per_dim_log_prob = -0.5 * jnp.square((value - mean) / std)
    per_dim_log_prob -= 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)
    return jnp.clip(per_dim_log_prob, -5.0, 2.0).mean(axis=-1)


def _ddpm_transition_mean(
    actor: DDPM,
    obs: jnp.ndarray,
    xt: jnp.ndarray,
    timestep: jnp.ndarray,
) -> jnp.ndarray:
    # DDPM stores a leading identity schedule entry, while the denoiser uses
    # the official zero-based timestep range K-1, ..., 0.
    eps_theta = actor(
        xt, timestep - 1, condition=obs, training=False
    )
    alpha_hat = actor.alpha_hats[timestep]
    alpha_hat_prev = actor.alpha_hats[timestep - 1]
    alpha = actor.alphas[timestep]
    beta = actor.betas[timestep]
    x0_hat = (
        xt - jnp.sqrt(1.0 - alpha_hat) * eps_theta
    ) / jnp.sqrt(alpha_hat)
    if actor.clip_sampler:
        x0_hat = jnp.clip(x0_hat, actor.x_min, actor.x_max)
    mean_coef1 = jnp.sqrt(alpha_hat_prev) * beta / (1.0 - alpha_hat)
    mean_coef2 = jnp.sqrt(alpha) * (1.0 - alpha_hat_prev) / (
        1.0 - alpha_hat
    )
    return mean_coef1 * x0_hat + mean_coef2 * xt


@partial(jax.jit, static_argnames=("steps", "min_logprob_std"))
def jit_compute_transition_log_probs(
    actor: DDPM,
    obs: jnp.ndarray,
    xt: jnp.ndarray,
    xt_1: jnp.ndarray,
    denoising_indices: jnp.ndarray,
    steps: int,
    min_logprob_std: float,
) -> jnp.ndarray:
    """Log-probabilities for a batch of individually sampled denoising steps."""
    timestep = (steps - denoising_indices).reshape(-1, 1)
    mean = _ddpm_transition_mean(actor, obs, xt, timestep)
    posterior_std = jnp.sqrt(
        jnp.maximum(actor.postvars[timestep], 0.0)
    )
    std = jnp.maximum(posterior_std, min_logprob_std)
    return clipped_gaussian_mean_log_prob(xt_1, mean, std)


@partial(jax.jit, static_argnames=("steps", "min_logprob_std"))
def jit_compute_chain_log_probs(
    actor: DDPM,
    obs: jnp.ndarray,
    chain: jnp.ndarray,
    steps: int,
    min_logprob_std: float,
) -> jnp.ndarray:
    """Return one averaged action log-probability per denoising transition."""
    batch_size = obs.shape[0]

    def step_fn(_, denoising_index):
        timestep = jnp.full(
            (batch_size, 1),
            steps - denoising_index,
            dtype=jnp.int32,
        )
        xt = chain[:, denoising_index]
        xt_1 = chain[:, denoising_index + 1]
        mean = _ddpm_transition_mean(actor, obs, xt, timestep)
        posterior_std = jnp.sqrt(
            jnp.maximum(actor.postvars[timestep], 0.0)
        )
        std = jnp.maximum(posterior_std, min_logprob_std)
        log_prob = clipped_gaussian_mean_log_prob(xt_1, mean, std)
        return None, log_prob

    _, step_log_probs = jax.lax.scan(
        step_fn, None, jnp.arange(steps, dtype=jnp.int32)
    )
    return jnp.transpose(step_log_probs, (1, 0))


@partial(
    jax.jit,
    static_argnames=(
        "deterministic",
        "steps",
        "min_sampling_std",
        "randn_clip_value",
        "eval_zero_xT",
    ),
)
def jit_sample_actions(
    rng: PRNGKey,
    actor: DDPM,
    obs: jnp.ndarray,
    deterministic: bool,
    steps: int,
    min_sampling_std: float,
    randn_clip_value: float,
    eval_zero_xT: bool,
) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    """Sample the DDPM chain with the exploration schedule used by DPPO."""
    batch_size = obs.shape[0]
    rng, prior_rng = jax.random.split(rng)
    xT = jax.random.normal(prior_rng, (batch_size, actor.x_dim))
    if deterministic and eval_zero_xT:
        xT = jnp.zeros_like(xT)

    def sample_step(carry, timestep):
        rng, xt = carry
        rng, noise_rng = jax.random.split(rng)
        timestep_batch = jnp.full(
            (batch_size, 1), timestep, dtype=jnp.int32
        )
        mean = _ddpm_transition_mean(actor, obs, xt, timestep_batch)
        posterior_std = jnp.sqrt(
            jnp.maximum(actor.postvars[timestep], 0.0)
        )
        if deterministic:
            std = jnp.where(
                timestep == 1,
                0.0,
                jnp.maximum(posterior_std, 1e-3),
            )
        else:
            std = jnp.maximum(posterior_std, min_sampling_std)
        noise = jnp.clip(
            jax.random.normal(noise_rng, xt.shape),
            -randn_clip_value,
            randn_clip_value,
        )
        xt_1 = mean + std * noise
        return (rng, xt_1), xt

    (rng, action), chain_prefix = jax.lax.scan(
        sample_step,
        (rng, xT),
        jnp.arange(steps, 0, -1, dtype=jnp.int32),
        unroll=True,
    )
    chain = jnp.transpose(
        jnp.concatenate([chain_prefix, action[jnp.newaxis]], axis=0),
        (1, 0, 2),
    )
    return rng, action, chain


def _scale_rewards(
    reward_scaler: Model,
    rewards: jnp.ndarray,
    done: jnp.ndarray,
    gamma: float,
    clip_value: float,
) -> Tuple[Model, jnp.ndarray, jnp.ndarray]:
    """Scale rewards by running variance of discounted backward returns."""
    params = reward_scaler.params

    def return_step(previous_return, inputs):
        reward, episode_done = inputs
        discounted_return = reward + gamma * previous_return
        next_return = discounted_return * (1.0 - episode_done)
        return next_return, discounted_return

    return_accumulator, discounted_returns = jax.lax.scan(
        return_step,
        params["return_accumulator"],
        (rewards, done),
    )
    flattened_returns = discounted_returns.reshape(-1)
    batch_mean = flattened_returns.mean()
    batch_var = flattened_returns.var()
    batch_count = flattened_returns.size

    delta = batch_mean - params["mean"]
    total_count = params["count"] + batch_count
    mean = params["mean"] + delta * batch_count / total_count
    m_a = params["var"] * params["count"]
    m_b = batch_var * batch_count
    m2 = (
        m_a
        + m_b
        + jnp.square(delta) * params["count"] * batch_count / total_count
    )
    var = m2 / (total_count - 1.0)
    scaled_rewards = jnp.clip(
        rewards / jnp.sqrt(var + 1e-8),
        -clip_value,
        clip_value,
    )
    new_params = {
        "mean": mean,
        "var": var,
        "count": total_count,
        "return_accumulator": return_accumulator,
    }
    reward_scaler = reward_scaler.replace(
        state=reward_scaler.state.replace(params=new_params)
    )
    return reward_scaler, scaled_rewards, jnp.sqrt(var + 1e-8)


@partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "gae_lambda",
        "gamma_denoising",
        "clip_epsilon",
        "clip_epsilon_base",
        "clip_epsilon_rate",
        "critic_loss_coeff",
        "reward_scaling",
        "reward_scale_running",
        "reward_scale_clip",
        "normalize_advantage",
        "num_epochs",
        "num_minibatches",
        "batch_size",
        "denoising_steps",
        "min_logprob_std",
    ),
)
def jit_update_dppo(
    rng: PRNGKey,
    actor: DDPM,
    critic: Model,
    reward_scaler: Model,
    rollout: RolloutBatch,
    gamma: float,
    gae_lambda: float,
    gamma_denoising: float,
    clip_epsilon: float,
    clip_epsilon_base: float,
    clip_epsilon_rate: float,
    critic_loss_coeff: float,
    reward_scaling: float,
    reward_scale_running: bool,
    reward_scale_clip: float,
    normalize_advantage: bool,
    num_epochs: int,
    num_minibatches: int,
    batch_size: int,
    denoising_steps: int,
    min_logprob_std: float,
):
    rollout_length, env_count = rollout.rewards.shape[:2]
    transition_count = rollout_length * env_count
    denoising_steps = int(denoising_steps)

    done = jnp.maximum(rollout.terminated, rollout.truncated)
    if reward_scale_running:
        reward_scaler, rewards, return_std = _scale_rewards(
            reward_scaler,
            rollout.rewards,
            done,
            gamma,
            reward_scale_clip,
        )
        rewards = rewards * reward_scaling
    else:
        rewards = rollout.rewards * reward_scaling
        return_std = jnp.asarray(1.0, dtype=rewards.dtype)

    value_pred = critic(rollout.obs)
    next_value_pred = critic(rollout.next_obs)
    gae_vs, gae_advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            rewards=rewards,
            values=value_pred,
            next_values=next_value_pred,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )
    )

    flat_obs = rollout.obs.reshape(transition_count, -1)
    flat_chains = rollout.extras["action_chains"].reshape(
        transition_count, denoising_steps + 1, -1
    )
    flat_advantages = gae_advantages.reshape(transition_count)
    flat_returns = gae_vs.reshape(transition_count, 1)
    flat_old_step_log_probs = jax.lax.stop_gradient(
        jit_compute_chain_log_probs(
            actor,
            flat_obs,
            flat_chains,
            denoising_steps,
            min_logprob_std,
        )
    )

    denoising_discounts = jnp.power(
        gamma_denoising,
        jnp.arange(denoising_steps - 1, -1, -1),
    )
    schedule_fraction = jnp.linspace(0.0, 1.0, denoising_steps)
    exponential_denominator = jnp.expm1(clip_epsilon_rate)
    schedule_progress = jnp.where(
        jnp.abs(exponential_denominator) > 1e-8,
        jnp.expm1(clip_epsilon_rate * schedule_fraction)
        / exponential_denominator,
        schedule_fraction,
    )
    clip_schedule = clip_epsilon_base + (
        clip_epsilon - clip_epsilon_base
    ) * schedule_progress

    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, permutation_rng = jax.random.split(rng)
        # Keep the same minibatch contract as PPO/FPO: batch_size counts
        # environment transitions. Every selected transition carries its full
        # K-step denoising chain, which is evaluated as an inner loss axis.
        permutation = jax.random.permutation(permutation_rng, transition_count)
        minibatch_indices = permutation[
            : num_minibatches * batch_size
        ].reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry
            minibatch_obs = flat_obs[indices]
            minibatch_chains = flat_chains[indices]
            old_log_probs = flat_old_step_log_probs[indices]
            advantages = flat_advantages[indices]
            returns = flat_returns[indices]

            if normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            weighted_advantages = (
                advantages[:, None] * denoising_discounts[None, :]
            )
            clip_epsilon_batch = clip_schedule[None, :]

            def actor_loss_fn(actor_params, dropout_rng):
                del dropout_rng
                current_actor = actor.replace(
                    state=actor.state.replace(params=actor_params)
                )
                new_log_probs = jit_compute_chain_log_probs(
                    current_actor,
                    minibatch_obs,
                    minibatch_chains,
                    denoising_steps,
                    min_logprob_std,
                )
                log_ratio = new_log_probs - old_log_probs
                ratios = jnp.exp(log_ratio)
                unclipped_objective = ratios * weighted_advantages
                clipped_objective = jnp.clip(
                    ratios,
                    1.0 - clip_epsilon_batch,
                    1.0 + clip_epsilon_batch,
                ) * weighted_advantages
                loss = -jnp.minimum(
                    unclipped_objective, clipped_objective
                ).mean()
                return loss, {
                    "loss/policy_loss": loss,
                    "misc/policy_ratio": ratios.mean(),
                    "misc/clipped_ratio": (
                        jnp.abs(ratios - 1.0) > clip_epsilon_batch
                    ).mean(),
                    "misc/approx_kl": (
                        (ratios - 1.0) - log_ratio
                    ).mean(),
                    "misc/log_ratio_abs_max": jnp.abs(log_ratio).max(),
                }

            new_actor, actor_info = actor.apply_gradient(actor_loss_fn)

            def critic_loss_fn(critic_params, dropout_rng):
                values = critic.apply(
                    {"params": critic_params},
                    minibatch_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                value_loss = 0.5 * jnp.square(returns - values).mean()
                objective = critic_loss_coeff * value_loss
                return objective, {
                    "loss/value_loss": value_loss,
                    "loss/value_objective": objective,
                    "misc/value_mean": values.mean(),
                }

            new_critic, critic_info = critic.apply_gradient(critic_loss_fn)
            return (rng, new_actor, new_critic), {
                **actor_info,
                **critic_info,
            }

        (rng, actor, critic), minibatch_metrics = jax.lax.scan(
            minibatch_step,
            init=(rng, actor, critic),
            xs=minibatch_indices,
        )
        return (rng, actor, critic), minibatch_metrics

    (rng, actor, critic), all_metrics = jax.lax.scan(
        epoch_step,
        init=(rng, actor, critic),
        length=num_epochs,
    )
    metrics = jax.tree.map(lambda value: value.mean(), all_metrics)
    metrics.update(
        {
            "misc/reward_mean": rollout.rewards.mean(),
            "misc/scaled_reward_mean": rewards.mean(),
            "misc/reward_return_std": return_std,
            "misc/advantages_mean": flat_advantages.mean(),
            "misc/advantages_std": flat_advantages.std(),
        }
    )
    return rng, actor, critic, reward_scaler, metrics


class DPPOAgent(BaseAgent):
    """Diffusion Policy Policy Optimization (DPPO)."""

    name = "DPPOAgent"
    model_names = ["actor", "critic", "reward_scaler"]

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: DPPOConfig,
        seed: int,
    ):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        if cfg.diffusion.solver != "ddpm":
            raise ValueError(
                "DPPO currently supports solver='ddpm' only; the likelihood "
                "for stochastic DDIM transitions is not implemented."
            )

        transition_count = cfg.num_envs * cfg.rollout_length
        if transition_count % cfg.batch_size != 0:
            raise ValueError(
                "DPPO batch_size must evenly divide the rollout's environment "
                f"transitions: {transition_count} % {cfg.batch_size} != 0."
            )
        expected_minibatches = transition_count // cfg.batch_size
        if cfg.num_minibatches != expected_minibatches:
            raise ValueError(
                "DPPO minibatches must cover all environment-transition "
                "batches: expected num_minibatches="
                f"{expected_minibatches}, got {cfg.num_minibatches}. "
                "DPPO uses the same batch_size contract as PPO/FPO; the full "
                "K-step denoising chain is an inner loss dimension."
            )

        self.rng, actor_rng, critic_rng, scaler_rng = jax.random.split(
            self.rng, 4
        )
        backbone_cls = {
            "mlp": MLP,
            "residual_mlp": DPPOResidualMLP,
            "simba": Simba,
        }[cfg.backbone_cls]
        actor_activation = get_activation(cfg.diffusion.activation)
        critic_activation = get_activation(cfg.critic_activation)

        actor_network = DPPODiffusionBackbone(
            noise_predictor=backbone_cls(
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

        critic_network = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=critic_activation,
            )
        )
        self.critic = Model.create(
            critic_network,
            critic_rng,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.reward_scaler = Model.create(
            RunningRewardScalerNetwork(num_envs=cfg.num_envs),
            scaler_rng,
            inputs=(),
        )

    def train_step(
        self,
        rollout: RolloutBatch,
        step: int,
    ) -> Metric:
        del step
        (
            self.rng,
            self.actor,
            self.critic,
            self.reward_scaler,
            metrics,
        ) = jit_update_dppo(
            self.rng,
            self.actor,
            self.critic,
            self.reward_scaler,
            rollout,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            gamma_denoising=self.cfg.gamma_denoising,
            clip_epsilon=self.cfg.clip_epsilon,
            clip_epsilon_base=self.cfg.clip_epsilon_base,
            clip_epsilon_rate=self.cfg.clip_epsilon_rate,
            critic_loss_coeff=self.cfg.critic_loss_coeff,
            reward_scaling=self.cfg.reward_scaling,
            reward_scale_running=self.cfg.reward_scale_running,
            reward_scale_clip=self.cfg.reward_scale_clip,
            normalize_advantage=self.cfg.normalize_advantage,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
            denoising_steps=self.cfg.diffusion.steps,
            min_logprob_std=(
                self.cfg.diffusion.min_logprob_denoising_std
            ),
        )
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        assert num_samples == 1, "DPPO only supports num_samples=1"
        self.rng, action, chain = jit_sample_actions(
            self.rng,
            self.actor,
            obs,
            deterministic,
            self.cfg.diffusion.steps,
            self.cfg.diffusion.min_sampling_denoising_std,
            self.cfg.diffusion.randn_clip_value,
            self.cfg.diffusion.eval_zero_xT,
        )
        step_log_probs = jit_compute_chain_log_probs(
            self.actor,
            obs,
            chain,
            self.cfg.diffusion.steps,
            self.cfg.diffusion.min_logprob_denoising_std,
        )
        log_prob = step_log_probs.mean(axis=-1, keepdims=True)
        return action, {
            "log_prob": log_prob,
            "action_chains": chain,
        }
