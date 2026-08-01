"""Action smoothness metrics for continuous-control evaluation."""

from typing import Dict

import numpy as np


HIGH_FREQUENCY_CUTOFF = 0.25


def compute_action_smoothness_metrics(
    actions: np.ndarray,
    episode_lengths: np.ndarray,
    high_frequency_cutoff: float = HIGH_FREQUENCY_CUTOFF,
) -> Dict[str, float]:
    """Compute episode-aware temporal smoothness metrics.

    Args:
        actions: Executed normalized actions with shape ``(time, episode, action)``.
            Values after an episode's valid prefix are ignored.
        episode_lengths: Valid action count for each episode, shape ``(episode,)``.
        high_frequency_cutoff: Frequencies at or above this value, in cycles per
            control step, are treated as high frequency. The default 0.25 is the
            upper half of the frequency range up to the Nyquist frequency (0.5).

    Returns:
        Five scalars whose keys can be logged under the ``eval`` namespace.

    The frequency-domain metric removes each action channel's temporal mean, then
    pools spectral energy across episodes and channels. Constant action signals
    therefore have a high-frequency ratio of zero. Episodes are never concatenated,
    so resets do not create artificial differences or high-frequency energy.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3:
        raise ValueError(
            "actions must have shape (time, episode, action), "
            f"got {actions.shape}"
        )

    time_steps, num_episodes, action_dim = actions.shape
    if num_episodes == 0 or action_dim == 0:
        raise ValueError("actions must contain at least one episode and action channel")

    lengths_array = np.asarray(episode_lengths)
    if lengths_array.shape != (num_episodes,):
        raise ValueError(
            f"episode_lengths must have shape ({num_episodes},), "
            f"got {lengths_array.shape}"
        )
    if not np.all(np.isfinite(lengths_array)):
        raise ValueError("episode_lengths must be finite")

    rounded_lengths = np.rint(lengths_array)
    if not np.array_equal(lengths_array, rounded_lengths):
        raise ValueError("episode_lengths must contain integers")
    lengths = rounded_lengths.astype(np.int64)
    if np.any(lengths < 0) or np.any(lengths > time_steps):
        raise ValueError(
            f"episode_lengths must be between 0 and the time dimension ({time_steps})"
        )
    if not 0.0 < high_frequency_cutoff <= 0.5:
        raise ValueError("high_frequency_cutoff must be in (0, 0.5]")

    delta_norms = []
    delta_sum = 0.0
    delta_count = 0
    accel_sum = 0.0
    accel_count = 0
    jerk_sum = 0.0
    jerk_count = 0
    high_frequency_power = 0.0
    total_dynamic_power = 0.0

    for episode_idx, episode_length in enumerate(lengths):
        episode_actions = np.asarray(
            actions[:episode_length, episode_idx], dtype=np.float64
        )
        if not np.all(np.isfinite(episode_actions)):
            raise ValueError(f"actions for episode {episode_idx} must be finite")

        if episode_length >= 2:
            delta = np.diff(episode_actions, axis=0)
            episode_delta_norms = np.linalg.norm(delta, axis=-1)
            delta_norms.append(episode_delta_norms)
            delta_sum += float(np.sum(episode_delta_norms))
            delta_count += episode_delta_norms.size

            if episode_length >= 3:
                acceleration = np.diff(delta, axis=0)
                acceleration_norms = np.linalg.norm(acceleration, axis=-1)
                accel_sum += float(np.sum(acceleration_norms))
                accel_count += acceleration_norms.size

                if episode_length >= 4:
                    jerk = np.diff(acceleration, axis=0)
                    jerk_norms = np.linalg.norm(jerk, axis=-1)
                    jerk_sum += float(np.sum(jerk_norms))
                    jerk_count += jerk_norms.size

        if episode_length >= 2:
            centered_actions = episode_actions - np.mean(
                episode_actions, axis=0, keepdims=True
            )
            spectrum = np.fft.rfft(centered_actions, axis=0)
            power = np.abs(spectrum) ** 2 / episode_length

            # Recover full-spectrum energy from the one-sided real FFT.
            if episode_length % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0

            frequencies = np.fft.rfftfreq(episode_length, d=1.0)
            dynamic_mask = frequencies > 0.0
            high_frequency_mask = frequencies >= high_frequency_cutoff
            total_dynamic_power += float(np.sum(power[dynamic_mask]))
            high_frequency_power += float(np.sum(power[high_frequency_mask]))

    if delta_norms:
        pooled_delta_norms = np.concatenate(delta_norms)
        delta_p95 = float(np.percentile(pooled_delta_norms, 95))
    else:
        delta_p95 = 0.0

    high_frequency_ratio = (
        high_frequency_power / total_dynamic_power
        if total_dynamic_power > 0.0
        else 0.0
    )

    return {
        "action_delta_l2_mean": delta_sum / delta_count if delta_count else 0.0,
        "action_delta_l2_p95": delta_p95,
        "action_accel_l2_mean": accel_sum / accel_count if accel_count else 0.0,
        "action_jerk_l2_mean": jerk_sum / jerk_count if jerk_count else 0.0,
        "action_hf_power_ratio": float(np.clip(high_frequency_ratio, 0.0, 1.0)),
    }
