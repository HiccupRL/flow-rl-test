from functools import partial
from typing import Callable, Sequence

import flax.linen as nn
from flax import traverse_util
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.fpo import OutputScale, clamp_ste
from flowrl.agent.online.ppo import compute_gae
from flowrl.config.online.algo.policyflow import PolicyFlowConfig
from flowrl.flow.cnf import ContinuousNormalizingFlow, FlowBackbone
from flowrl.functional.activation import get_activation
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Metric, Param, PRNGKey, RolloutBatch


def _pytorch_linear_kernel_init(fan_in: int):
    return nn.initializers.uniform(scale=fan_in ** -0.5)


def _pytorch_linear_bias_init(fan_in: int):
    return nn.initializers.uniform(scale=fan_in ** -0.5)


def _scaled_xavier_normal(scale: float):
    xavier = nn.initializers.xavier_normal()

    def init(key, shape, dtype=jnp.float32):
        return scale * xavier(key, shape, dtype)

    return init


class PolicyFlowVelocityNetwork(nn.Module):
    """JAX port of the official PolicyFlow ``FlowMlp``.

    The official implementation uses frozen random Fourier frequencies, a
    two-layer Mish time MLP, a linear observation embedding, and a Glorot actor
    MLP whose last kernel is initialized at 1/100 scale.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    embedding_dim: int
    activation: Callable
    zero_init_output: bool
    output_init_scale: float
    fourier_scale: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        condition: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        del training
        frequency_count = self.embedding_dim // 8
        frequencies = self.param(
            "fourier_frequencies",
            nn.initializers.normal(stddev=self.fourier_scale),
            (frequency_count,),
        )
        frequencies = jax.lax.stop_gradient(frequencies)
        angles = 2.0 * jnp.pi * time * frequencies
        time_features = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        time_fan_in = 2 * frequency_count
        time_embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=_pytorch_linear_kernel_init(time_fan_in),
            bias_init=_pytorch_linear_bias_init(time_fan_in),
        )(time_features)
        time_embedding = jax.nn.mish(time_embedding)
        time_embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=_pytorch_linear_kernel_init(self.embedding_dim),
            bias_init=_pytorch_linear_bias_init(self.embedding_dim),
        )(time_embedding)
        observation_fan_in = condition.shape[-1]
        observation_embedding = nn.Dense(
            self.embedding_dim,
            kernel_init=_pytorch_linear_kernel_init(observation_fan_in),
            bias_init=_pytorch_linear_bias_init(observation_fan_in),
        )(condition)

        features = jnp.concatenate(
            [x, time_embedding + observation_embedding],
            axis=-1,
        )
        fan_in = self.action_dim + self.embedding_dim
        for hidden_dim in self.hidden_dims:
            features = nn.Dense(
                hidden_dim,
                kernel_init=_scaled_xavier_normal(1.0),
                bias_init=_pytorch_linear_bias_init(fan_in),
            )(features)
            features = self.activation(features)
            fan_in = hidden_dim

        output_scale = 0.0 if self.zero_init_output else self.output_init_scale
        output_bias_init = (
            nn.initializers.zeros
            if self.zero_init_output
            else _pytorch_linear_bias_init(fan_in)
        )
        return nn.Dense(
            self.action_dim,
            kernel_init=_scaled_xavier_normal(output_scale),
            bias_init=output_bias_init,
        )(features)


class PolicyFlowCriticNetwork(nn.Module):
    """JAX port of the official PolicyFlow critic ``Network``."""

    hidden_dims: Sequence[int]
    activation: Callable
    output_init_scale: float

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        del training
        features = observations
        fan_in = observations.shape[-1]
        for hidden_dim in self.hidden_dims:
            features = nn.Dense(
                hidden_dim,
                kernel_init=_scaled_xavier_normal(1.0),
                bias_init=_pytorch_linear_bias_init(fan_in),
            )(features)
            features = self.activation(features)
            fan_in = hidden_dim
        return nn.Dense(
            1,
            kernel_init=_scaled_xavier_normal(self.output_init_scale),
            bias_init=_pytorch_linear_bias_init(fan_in),
        )(features)


class PolicyFlowBackbone(nn.Module):
    backbone: nn.Module
    action_dim: int
    init_logstd: float

    @nn.compact
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        self.param(
            "logstd",
            nn.initializers.constant(self.init_logstd),
            (self.action_dim,),
        )
        return self.backbone(*args, **kwargs)


def _policyflow_std(params: Param, logstd_min: float, logstd_max: float) -> jnp.ndarray:
    logstd = jnp.clip(params["logstd"], logstd_min, logstd_max)
    return jnp.exp(logstd)


def _gaussian_log_prob(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    var = jnp.square(std)
    logstd = jnp.log(std)
    log_prob = -0.5 * (jnp.square(x - mean) / var + 2.0 * logstd + jnp.log(2.0 * jnp.pi))
    return jnp.sum(log_prob, axis=-1, keepdims=True)


def _gaussian_entropy(std: jnp.ndarray) -> jnp.ndarray:
    entropy = jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return jnp.sum(entropy, axis=-1, keepdims=True)


def _policyflow_time_grid(steps: int, interpolation_type: str) -> jnp.ndarray:
    if interpolation_type == "trigflow":
        final_t = 0.5 * jnp.pi
    else:
        final_t = 1.0
    endpoints = jnp.linspace(0.0, final_t, steps + 1)
    mids = 0.5 * (endpoints[:-1] + endpoints[1:])
    # The official implementation samples starts, midpoints, and the terminal
    # endpoint (2 * steps + 1 values).
    return jnp.sort(jnp.concatenate([endpoints, mids], axis=0))


def _policyflow_final_time(interpolation_type: str) -> float:
    if interpolation_type == "trigflow":
        return 0.5 * jnp.pi
    return 1.0


def _interpolate(
    rng: PRNGKey,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    interpolation_type: str,
) -> jnp.ndarray:
    if interpolation_type == "rectified_flow":
        return (1.0 - t) * x0 + t * x1
    if interpolation_type == "stochastic_interpolant":
        noise = jax.random.normal(rng, x1.shape)
        noise_scale = jnp.sqrt(2.0 * t * jnp.clip(1.0 - t, a_min=1e-6))
        return (1.0 - t) * x0 + t * x1 + noise_scale * noise
    if interpolation_type == "trigflow":
        return jnp.cos(t) * x0 + jnp.sin(t) * x1
    raise ValueError(f"Unsupported PolicyFlow interpolation_type={interpolation_type}")


def _brownian_regularizer(
    xt: jnp.ndarray,
    t: jnp.ndarray,
    vel: jnp.ndarray,
    vel_last: jnp.ndarray,
    interpolation_type: str,
    reduction: str,
) -> jnp.ndarray:
    if interpolation_type == "rectified_flow":
        lhs = (1.0 - t) * vel
        rhs = xt - t * vel_last
    elif interpolation_type == "stochastic_interpolant":
        lhs = (2.0 * jnp.square(t - 0.5) + 0.5) * vel
        rhs = xt - t * vel_last
    elif interpolation_type == "trigflow":
        lhs = jnp.cos(t) * vel
        rhs = jnp.cos(t) * xt - jnp.sin(t) * vel_last
    else:
        raise ValueError(f"Unsupported PolicyFlow interpolation_type={interpolation_type}")
    squared_error = jnp.square(lhs - rhs)
    if reduction == "official_mean":
        # Match torch.nn.functional.mse_loss in the released implementation.
        return jnp.mean(squared_error)
    if reduction == "paper_l2":
        # The paper writes an action-space squared L2 norm inside the batch
        # expectation, without the extra division by action dimension.
        return jnp.mean(jnp.sum(squared_error, axis=-1))
    raise ValueError(f"Unsupported PolicyFlow brownian_reduction={reduction}")


def _compute_flow_variation(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    actor_params: Param,
    old_actor_params: Param,
    obs: jnp.ndarray,
    action_prior: jnp.ndarray,
    x0: jnp.ndarray,
    logstd_min: float,
    logstd_max: float,
    interpolation_type: str,
    time_sampling: str,
    brownian_reduction: str,
    compute_brownian_reg_loss: bool,
):
    rng, t_rng, noise_rng = jax.random.split(rng, 3)
    if time_sampling == "discrete":
        time_grid = _policyflow_time_grid(actor.steps, interpolation_type)
        idx = jax.random.randint(t_rng, (action_prior.shape[0],), 0, time_grid.shape[0])
        t = time_grid[idx][..., None]
    elif time_sampling == "continuous":
        t = jax.random.uniform(
            t_rng,
            (action_prior.shape[0], 1),
            minval=0.0,
            maxval=_policyflow_final_time(interpolation_type),
        )
    else:
        raise ValueError(f"Unsupported PolicyFlow time_sampling={time_sampling}")
    xt = _interpolate(noise_rng, x0, action_prior, t, interpolation_type)

    vel_last = jax.lax.stop_gradient(
        actor.apply(
            {"params": old_actor_params},
            xt,
            t,
            condition=obs,
            training=False,
        )
    )
    vel = actor.apply(
        {"params": actor_params},
        xt,
        t,
        condition=obs,
        training=True,
    )
    delta_vel = vel - vel_last
    std = _policyflow_std(actor_params, logstd_min, logstd_max)
    std = jnp.broadcast_to(std, action_prior.shape)

    if compute_brownian_reg_loss:
        brownian_reg_loss = _brownian_regularizer(
            xt,
            t,
            vel,
            vel_last,
            interpolation_type,
            brownian_reduction,
        )
    else:
        brownian_reg_loss = jnp.array(0.0, dtype=action_prior.dtype)
    return delta_vel, std, brownian_reg_loss


def _compute_kl_divergence(
    actions_mean: jnp.ndarray,
    actions_std: jnp.ndarray,
    last_action_mean: jnp.ndarray,
    last_action_std: jnp.ndarray,
) -> jnp.ndarray:
    # Keep the released implementation's placement and magnitude of epsilon.
    std_ratio = actions_std / last_action_std + 1.0e-5
    std_drifted = jnp.square(last_action_std) + jnp.square(last_action_mean - actions_mean)
    kl = jnp.sum(
        jnp.log(std_ratio)
        + std_drifted / (2.0 * jnp.square(actions_std))
        - 0.5,
        axis=-1,
    )
    return jnp.mean(kl)


def _clipped_value_loss(
    predicted_values: jnp.ndarray,
    target_values: jnp.ndarray,
    old_values: jnp.ndarray,
    value_clip: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """PPO clipped value loss used by RSL-RL.

    Taking the maximum of the unclipped and clipped losses preserves the
    gradient when the new prediction leaves the clipping interval.
    """
    loss_unclipped = jnp.square(target_values - predicted_values)
    predicted_values_clipped = old_values + jnp.clip(
        predicted_values - old_values,
        -value_clip,
        value_clip,
    )
    loss_clipped = jnp.square(target_values - predicted_values_clipped)
    loss_elements = jnp.maximum(loss_unclipped, loss_clipped)
    clip_fraction = jnp.mean(loss_clipped > loss_unclipped)
    return jnp.mean(loss_elements), clip_fraction


def _sample_policyflow_prior(
    actor: ContinuousNormalizingFlow,
    actor_params: Param,
    x0: jnp.ndarray,
    obs: jnp.ndarray,
    interpolation_type: str,
) -> jnp.ndarray:
    step_grid = jnp.linspace(
        0.0,
        _policyflow_final_time(interpolation_type),
        actor.steps + 1,
    )

    def step_fn(xt, i):
        t0 = step_grid[i]
        t1 = step_grid[i + 1]
        dt = t1 - t0
        t = jnp.full((*xt.shape[:-1], 1), t0, dtype=xt.dtype)
        vel_t = actor.apply(
            {"params": actor_params},
            xt,
            t,
            condition=obs,
            training=False,
        )
        xt_mid = xt + 0.5 * dt * vel_t
        t_mid = jnp.full((*xt.shape[:-1], 1), t0 + 0.5 * dt, dtype=xt.dtype)
        vel_mid = actor.apply(
            {"params": actor_params},
            xt_mid,
            t_mid,
            condition=obs,
            training=False,
        )
        x_next = xt + dt * vel_mid
        if actor.clip_sampler:
            x_next = jnp.clip(x_next, actor.x_min, actor.x_max)
        return x_next, None

    action_prior, _ = jax.lax.scan(step_fn, x0, jnp.arange(actor.steps))
    return action_prior


def _adaptive_optimizer(
    learning_rate: float,
    weight_decay: float,
) -> optax.GradientTransformation:
    def weight_decay_mask(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: path[-1] != "fourier_frequencies"
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # Gradient clipping is applied jointly over actor and critic gradients in
    # the update, matching the single optimizer in the released PyTorch code.
    return optax.chain(
        optax.inject_hyperparams(
            optax.adamw,
            static_args=("mask",),
        )(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask=weight_decay_mask,
        )
    )


def _optimizer_learning_rate(model: Model) -> jnp.ndarray:
    injected_state = model.state.opt_state[-1]
    return injected_state.hyperparams["learning_rate"]


def _set_optimizer_learning_rate(model: Model, learning_rate: jnp.ndarray) -> Model:
    opt_state = list(model.state.opt_state)
    injected_state = opt_state[-1]
    hyperparams = {
        **injected_state.hyperparams,
        "learning_rate": learning_rate,
    }
    opt_state[-1] = injected_state._replace(hyperparams=hyperparams)
    return model.replace(
        state=model.state.replace(opt_state=tuple(opt_state)),
    )


@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "ratio_clip",
    "reward_scaling", "normalize_advantage",
    "num_epochs", "num_minibatches", "batch_size",
    "gaussian_entropy_loss_scale", "brownian_reg_loss_scale",
    "value_loss_scale", "clip_predicted_values", "value_clip",
    "value_clip_mode", "time_limit_bootstrap",
    "clip_grad_norm",
    "log_ratio_clip", "logstd_min", "logstd_max",
    "interpolation_type", "time_sampling", "brownian_reduction",
    "adaptive_learning_rate",
))
def jit_update_policyflow(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    critic: Model,
    old_actor_params: Param,
    rollout: RolloutBatch,
    gamma: float,
    gae_lambda: float,
    ratio_clip: float,
    reward_scaling: float,
    normalize_advantage: bool,
    num_epochs: int,
    num_minibatches: int,
    batch_size: int,
    gaussian_entropy_loss_scale: float,
    brownian_reg_loss_scale: float,
    value_loss_scale: float,
    clip_predicted_values: bool,
    value_clip: float,
    value_clip_mode: str,
    time_limit_bootstrap: bool,
    clip_grad_norm: float | None,
    log_ratio_clip: float,
    logstd_min: float,
    logstd_max: float,
    interpolation_type: str,
    time_sampling: str,
    brownian_reduction: str,
    adaptive_learning_rate: bool,
    desired_kl: float,
    learning_rate_min: float,
    learning_rate_max: float,
    learning_rate_factor: float,
):
    T, B = rollout.rewards.shape[:2]

    critic_obs = rollout.extras.get("critic_obs", rollout.obs)
    next_critic_obs = rollout.extras.get("next_critic_obs", rollout.next_obs)
    value_pred = critic(critic_obs)
    next_value_pred = critic(next_critic_obs)
    rewards = rollout.rewards * reward_scaling
    if time_limit_bootstrap:
        # Match the official implementation: add a timeout bootstrap before
        # treating truncations as episode boundaries.
        rewards = rewards + gamma * value_pred * rollout.truncated
    episode_terminated = jnp.maximum(
        rollout.terminated,
        rollout.truncated,
    )
    gae_vs, gae_advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=episode_terminated,
            truncated=rollout.truncated,
            rewards=rewards,
            values=value_pred,
            next_values=next_value_pred,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )
    )

    if normalize_advantage:
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_advantages = gae_advantages.reshape(T * B, 1)
    flat_gae_vs = gae_vs.reshape(T * B, 1)
    flat_old_values = value_pred.reshape(T * B, 1)
    flat_critic_obs = critic_obs.reshape(T * B, -1)

    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_action_prior = rollout.extras["actions_prior"].reshape(T * B, -1)
    flat_flow_x0 = rollout.extras["flow_x0"].reshape(T * B, -1)
    flat_delta_actions = rollout.extras["delta_actions"].reshape(T * B, -1)
    flat_delta_std = rollout.extras["delta_actions_std"].reshape(T * B, -1)
    flat_delta_log_prob = rollout.extras["delta_actions_log_prob"].reshape(T * B, 1)

    compute_brownian_reg_loss = brownian_reg_loss_scale > 0.0

    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, perm_rng = jax.random.split(rng)

        perm = jax.random.permutation(perm_rng, T * B)
        total = num_minibatches * batch_size
        perm = perm[:total]
        mb_indices = perm.reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry
            rng, actor_rng = jax.random.split(rng)

            mb_obs = flat_obs[indices]
            mb_critic_obs = flat_critic_obs[indices]
            mb_advantages = flat_advantages[indices]
            mb_gae_vs = flat_gae_vs[indices]
            mb_old_values = flat_old_values[indices]

            mb_action_prior = flat_action_prior[indices]
            mb_flow_x0 = flat_flow_x0[indices]
            mb_delta_actions = flat_delta_actions[indices]
            mb_delta_std = flat_delta_std[indices]
            mb_delta_log_prob = flat_delta_log_prob[indices]

            def actor_loss_fn(actor_params, dropout_rng):
                delta_vel, delta_std_new, brownian_reg_loss = _compute_flow_variation(
                    dropout_rng,
                    actor,
                    actor_params,
                    old_actor_params,
                    mb_obs,
                    mb_action_prior,
                    mb_flow_x0,
                    logstd_min,
                    logstd_max,
                    interpolation_type,
                    time_sampling,
                    brownian_reduction,
                    compute_brownian_reg_loss,
                )
                actions_log_prob_new = _gaussian_log_prob(
                    mb_delta_actions,
                    delta_vel,
                    delta_std_new,
                )
                log_ratio = actions_log_prob_new - mb_delta_log_prob
                if log_ratio_clip > 0.0:
                    log_ratio = clamp_ste(log_ratio, -log_ratio_clip, log_ratio_clip)
                ratio = jnp.exp(log_ratio)

                surrogate = mb_advantages * ratio
                surrogate_clipped = mb_advantages * jnp.clip(
                    ratio,
                    1.0 - ratio_clip,
                    1.0 + ratio_clip,
                )
                policy_loss = -jnp.mean(jnp.minimum(surrogate, surrogate_clipped))

                entropy = jnp.mean(_gaussian_entropy(delta_std_new))
                gaussian_entropy_loss = -gaussian_entropy_loss_scale * entropy
                brownian_loss = brownian_reg_loss_scale * brownian_reg_loss
                actor_loss = policy_loss + gaussian_entropy_loss + brownian_loss

                kl = _compute_kl_divergence(
                    delta_vel,
                    delta_std_new,
                    jnp.zeros_like(delta_vel),
                    mb_delta_std,
                )
                return actor_loss, {
                    "loss/policy_loss": policy_loss,
                    "loss/gaussian_entropy_loss": gaussian_entropy_loss,
                    "loss/brownian_reg_loss": brownian_loss,
                    "misc/policy_ratio": jnp.mean(ratio),
                    "misc/clipped_ratio": jnp.mean(jnp.abs(ratio - 1.0) > ratio_clip),
                    "misc/kl": kl,
                    "misc/delta_vel_max": jnp.max(delta_vel),
                    "misc/delta_vel_min": jnp.min(delta_vel),
                    "misc/delta_std_mean": jnp.mean(delta_std_new),
                }

            actor_dropout_rng, next_actor_dropout_rng = jax.random.split(
                actor.dropout_rng
            )
            actor_grads, actor_metrics = jax.grad(
                actor_loss_fn,
                has_aux=True,
            )(actor.state.params, actor_dropout_rng)

            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params},
                    mb_critic_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                if clip_predicted_values:
                    if value_clip_mode == "official":
                        v_clipped = mb_old_values + jnp.clip(
                            v - mb_old_values,
                            -value_clip,
                            value_clip,
                        )
                        value_mse = jnp.mean(jnp.square(mb_gae_vs - v_clipped))
                        value_clip_fraction = jnp.mean(v != v_clipped)
                    elif value_clip_mode == "ppo":
                        value_mse, value_clip_fraction = _clipped_value_loss(
                            predicted_values=v,
                            target_values=mb_gae_vs,
                            old_values=mb_old_values,
                            value_clip=value_clip,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported PolicyFlow value_clip_mode={value_clip_mode}"
                        )
                else:
                    value_mse = jnp.mean(jnp.square(mb_gae_vs - v))
                    value_clip_fraction = jnp.array(0.0, dtype=v.dtype)
                v_loss = value_loss_scale * value_mse
                return v_loss, {
                    "loss/value_loss": v_loss,
                    "misc/value_mean": jnp.mean(v),
                    "misc/value_clip_fraction": value_clip_fraction,
                }

            critic_dropout_rng, next_critic_dropout_rng = jax.random.split(
                critic.dropout_rng
            )
            critic_grads, critic_metrics = jax.grad(
                critic_loss_fn,
                has_aux=True,
            )(critic.state.params, critic_dropout_rng)
            joint_grad_norm = optax.tree.norm(
                {
                    "actor": actor_grads,
                    "critic": critic_grads,
                }
            )
            if clip_grad_norm:
                grad_scale = jnp.minimum(
                    1.0,
                    clip_grad_norm / (joint_grad_norm + 1.0e-6),
                )
                actor_grads = jax.tree.map(lambda g: g * grad_scale, actor_grads)
                critic_grads = jax.tree.map(lambda g: g * grad_scale, critic_grads)

            new_actor = actor.replace(
                state=actor.state.apply_gradients(grads=actor_grads),
                dropout_rng=next_actor_dropout_rng,
            )
            new_critic = critic.replace(
                state=critic.state.apply_gradients(grads=critic_grads),
                dropout_rng=next_critic_dropout_rng,
            )
            learning_rate = _optimizer_learning_rate(new_actor)
            metrics = {**actor_metrics, **critic_metrics}
            metrics["misc/grad_norm"] = joint_grad_norm
            metrics["misc/learning_rate"] = learning_rate
            return (actor_rng, new_actor, new_critic), metrics

        (rng, actor, critic), mb_metrics = jax.lax.scan(
            minibatch_step,
            init=(rng, actor, critic),
            xs=mb_indices,
        )
        return (rng, actor, critic), mb_metrics

    (rng, actor, critic), all_metrics = jax.lax.scan(
        epoch_step,
        init=(rng, actor, critic),
        length=num_epochs,
    )

    metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    learning_rate = _optimizer_learning_rate(actor)
    if adaptive_learning_rate:
        kl = metrics["misc/kl"]
        learning_rate = jnp.where(
            kl > 2.0 * desired_kl,
            jnp.maximum(
                learning_rate_min,
                learning_rate / learning_rate_factor,
            ),
            learning_rate,
        )
        learning_rate = jnp.where(
            kl < 0.5 * desired_kl,
            jnp.minimum(
                learning_rate_max,
                learning_rate * learning_rate_factor,
            ),
            learning_rate,
        )
        actor = _set_optimizer_learning_rate(actor, learning_rate)
        critic = _set_optimizer_learning_rate(critic, learning_rate)
    metrics.update({
        "misc/reward_mean": rollout.rewards.mean(),
        "misc/obs_mean": flat_obs.mean(),
        "misc/obs_std": flat_obs.std(axis=0).mean(),
        "misc/action_l1_mean": jnp.abs(flat_actions).mean(),
        "misc/action_prior_l1_mean": jnp.abs(flat_action_prior).mean(),
        "misc/delta_action_l1_mean": jnp.abs(flat_delta_actions).mean(),
        "misc/advantages_mean": flat_advantages.mean(),
        "misc/advantages_std": flat_advantages.std(axis=0).mean(),
        "misc/learning_rate": learning_rate,
    })

    return rng, actor, critic, metrics


@partial(jax.jit, static_argnames=(
    "deterministic", "eval_zero_x0", "eval_sample_delta",
    "logstd_min", "logstd_max", "interpolation_type",
))
def jit_sample_action_policyflow(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    obs: jnp.ndarray,
    deterministic: bool,
    eval_zero_x0: bool,
    eval_sample_delta: bool,
    logstd_min: float,
    logstd_max: float,
    interpolation_type: str,
):
    rng, x0_rng, delta_rng = jax.random.split(rng, 3)
    B = obs.shape[0]
    x0_random = jax.random.normal(x0_rng, (B, actor.x_dim))
    x0_zero = jnp.zeros((B, actor.x_dim), dtype=obs.dtype)
    use_zero = deterministic and eval_zero_x0
    x0 = jax.lax.cond(use_zero, lambda: x0_zero, lambda: x0_random)

    action_prior = _sample_policyflow_prior(
        actor,
        actor.state.params,
        x0,
        obs,
        interpolation_type,
    )
    std = _policyflow_std(actor.state.params, logstd_min, logstd_max)
    std = jnp.broadcast_to(std, action_prior.shape)

    delta_random = std * jax.random.normal(delta_rng, action_prior.shape)
    sample_delta = (not deterministic) or eval_sample_delta
    delta_actions = jax.lax.cond(
        sample_delta,
        lambda: delta_random,
        lambda: jnp.zeros_like(delta_random),
    )
    actions = action_prior + delta_actions
    # Keep the sampled Gaussian residual unchanged. The trainer/environment
    # may clip the executed action, but PPO must store the pre-clipped sample
    # whose Gaussian density is evaluated below.
    delta_log_prob = _gaussian_log_prob(delta_actions, jnp.zeros_like(delta_actions), std)

    return actions, action_prior, x0, delta_actions, std, delta_log_prob


class PolicyFlowAgent(BaseAgent):
    """
    PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning.
    """
    name = "PolicyFlowAgent"
    model_names = ["actor", "critic"]

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: PolicyFlowConfig,
        seed: int,
        critic_obs_dim: int | None = None,
    ):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.critic_obs_dim = critic_obs_dim or obs_dim
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        critic_activation = get_activation(cfg.critic_activation)
        actor_activation = get_activation(cfg.flow.activation)
        backbone_cls = {
            "mlp": MLP,
            "simba": Simba,
        }[cfg.backbone_cls]

        if cfg.num_minibatches * cfg.batch_size != cfg.num_envs * cfg.rollout_length:
            raise ValueError(
                "PolicyFlow expects every rollout sample to be used exactly once per epoch: "
                f"num_minibatches*batch_size={cfg.num_minibatches * cfg.batch_size}, "
                f"num_envs*rollout_length={cfg.num_envs * cfg.rollout_length}."
            )
        if cfg.flow.time_dim % 8 != 0:
            raise ValueError(
                "The official PolicyFlow Fourier embedding requires flow.time_dim "
                f"to be divisible by 8, got {cfg.flow.time_dim}."
            )
        if cfg.flow.time_sampling not in {"discrete", "continuous"}:
            raise ValueError(
                "PolicyFlow flow.time_sampling must be 'discrete' or 'continuous', "
                f"got {cfg.flow.time_sampling}."
            )
        if cfg.value_clip_mode not in {"official", "ppo"}:
            raise ValueError(
                "PolicyFlow value_clip_mode must be 'official' or 'ppo', "
                f"got {cfg.value_clip_mode}."
            )
        if cfg.brownian_reduction not in {"official_mean", "paper_l2"}:
            raise ValueError(
                "PolicyFlow brownian_reduction must be 'official_mean' or "
                f"'paper_l2', got {cfg.brownian_reduction}."
            )

        critic_def = PolicyFlowCriticNetwork(
            hidden_dims=cfg.critic_hidden_dims,
            activation=critic_activation,
            output_init_scale=cfg.critic_output_init_scale,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.critic_obs_dim)),),
            optimizer=_adaptive_optimizer(
                learning_rate=cfg.critic_lr,
                weight_decay=cfg.weight_decay,
            ),
            clip_grad_norm=None,
        )

        if cfg.flow.architecture == "official":
            velocity_network = PolicyFlowVelocityNetwork(
                hidden_dims=cfg.flow.hidden_dims,
                action_dim=self.act_dim,
                embedding_dim=cfg.flow.time_dim,
                activation=actor_activation,
                zero_init_output=cfg.flow.zero_init_output,
                output_init_scale=cfg.flow.output_init_scale,
                fourier_scale=cfg.flow.fourier_scale,
            )
        elif cfg.flow.architecture == "legacy":
            velocity_network = FlowBackbone(
                vel_predictor=backbone_cls(
                    hidden_dims=cfg.flow.hidden_dims,
                    activation=actor_activation,
                    output_dim=self.act_dim,
                ),
                time_embedding=LearnableFourierEmbedding(
                    output_dim=cfg.flow.time_dim,
                ),
            )
        else:
            raise ValueError(
                "PolicyFlow flow.architecture must be 'official' or 'legacy', "
                f"got {cfg.flow.architecture}."
            )

        flow_backbone = PolicyFlowBackbone(
            backbone=OutputScale(
                backbone=velocity_network,
                output_scale=cfg.flow.output_scale,
            ),
            action_dim=self.act_dim,
            init_logstd=cfg.flow.init_logstd,
        )
        self.actor = ContinuousNormalizingFlow.create(
            network=flow_backbone,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, 1)),
                jnp.ones((1, self.obs_dim)),
            ),
            x_dim=self.act_dim,
            steps=cfg.flow.steps,
            clip_sampler=cfg.flow.clip_sampler,
            x_min=cfg.flow.x_min,
            x_max=cfg.flow.x_max,
            optimizer=_adaptive_optimizer(
                learning_rate=cfg.flow.lr,
                weight_decay=cfg.weight_decay,
            ),
            clip_grad_norm=None,
        )

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        old_actor_params = self.actor.state.params
        self.rng, self.actor, self.critic, metrics = jit_update_policyflow(
            self.rng,
            self.actor,
            self.critic,
            old_actor_params,
            rollout,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            ratio_clip=self.cfg.ratio_clip,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
            gaussian_entropy_loss_scale=self.cfg.gaussian_entropy_loss_scale,
            brownian_reg_loss_scale=self.cfg.brownian_reg_loss_scale,
            value_loss_scale=self.cfg.value_loss_scale,
            clip_predicted_values=self.cfg.clip_predicted_values,
            value_clip=self.cfg.value_clip,
            value_clip_mode=self.cfg.value_clip_mode,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
            clip_grad_norm=self.cfg.clip_grad_norm,
            log_ratio_clip=self.cfg.log_ratio_clip,
            logstd_min=self.cfg.flow.logstd_min,
            logstd_max=self.cfg.flow.logstd_max,
            interpolation_type=self.cfg.flow.interpolation_type,
            time_sampling=self.cfg.flow.time_sampling,
            brownian_reduction=self.cfg.brownian_reduction,
            adaptive_learning_rate=self.cfg.adaptive_learning_rate,
            desired_kl=self.cfg.desired_kl,
            learning_rate_min=self.cfg.learning_rate_min,
            learning_rate_max=self.cfg.learning_rate_max,
            learning_rate_factor=self.cfg.learning_rate_factor,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        self.rng, sample_rng = jax.random.split(self.rng)
        actions, action_prior, x0, delta_actions, std, delta_log_prob = jit_sample_action_policyflow(
            sample_rng,
            self.actor,
            obs,
            deterministic,
            eval_zero_x0=self.cfg.flow.eval_zero_x0,
            eval_sample_delta=self.cfg.flow.eval_sample_delta,
            logstd_min=self.cfg.flow.logstd_min,
            logstd_max=self.cfg.flow.logstd_max,
            interpolation_type=self.cfg.flow.interpolation_type,
        )

        return actions, {
            "actions_prior": action_prior,
            "flow_x0": x0,
            "delta_actions": delta_actions,
            "delta_actions_std": std,
            "delta_actions_log_prob": delta_log_prob,
        }
