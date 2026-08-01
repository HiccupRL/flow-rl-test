import copy
from collections import deque
from functools import lru_cache

import gymnasium as gym
import numpy as np
from dm_control import suite

# define pendulum swingup dense task
from dm_control.suite.pendulum import (
    _COSINE_BOUND,
    _DEFAULT_TIME_LIMIT,
    Physics,
    base,
    collections,
    control,
    get_model_and_assets,
    rewards,
)
from dm_control.suite.wrappers import action_scale


class SwingUpDense(base.Task):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Pendulum`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of the episode."""
        physics.named.data.qpos["hinge"] = self.random.uniform(-np.pi, np.pi)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Return state observations for the pendulum."""
        obs = collections.OrderedDict()
        obs["orientation"] = physics.pole_orientation()
        obs["velocity"] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(
            physics.pole_vertical(),
            (_COSINE_BOUND, 1),
            margin=1,
        )


def make_pendulum_swingup_dense(
    time_limit=_DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs=None,
):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = SwingUpDense(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs,
    )


# These tasks only mutate per-environment MjData during reset, so their clones
# can safely share the immutable compiled MjModel. Other tasks get a model copy.
_SHARED_MODEL_TASKS = {
    ("ball_in_cup", "catch"),
    ("cartpole", "balance"),
    ("cheetah", "run"),
    ("point_mass", "easy"),
}


@lru_cache(maxsize=None)
def _load_template(domain: str, task: str):
    if domain == "pendulum" and task == "swingup_dense":
        env = make_pendulum_swingup_dense(random=0)
        env.task.visualize_reward = False
        return env
    return suite.load(
        domain,
        task,
        task_kwargs={"random": 0},
        visualize_reward=False,
    )


def _clone_control_env(domain: str, task: str, seed: int):
    template = _load_template(domain, task)
    physics = template.physics.copy(
        share_model=(domain, task) in _SHARED_MODEL_TASKS
    )
    task_instance = copy.deepcopy(template.task)
    task_instance._random = np.random.RandomState(seed)
    return control.Environment(
        physics=physics,
        task=task_instance,
        time_limit=template._step_limit * template.control_timestep(),
        n_sub_steps=template._n_sub_steps,
        flat_observation=template._flat_observation,
    )


class DMControlEnv(gym.Env):
    """Gymnasium-compatible state or pixel wrapper for DMControl tasks."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str,
        seed: int,
        visual: bool,
        frame_skip: int,
        frame_stack: int,
        horizon: int = 1000,
        image_size: int = 84,
        camera: int = 0,
    ) -> None:
        super().__init__()

        self.task_name = task
        self.seed = int(seed)
        self.visual = visual
        self.frame_skip = int(frame_skip)
        self.frame_stack = int(frame_stack)
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        self.camera = camera
        self.render_mode = "rgb_array"

        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be positive, got {self.frame_skip}")
        if self.frame_stack < 1:
            raise ValueError(f"frame_stack must be positive, got {self.frame_stack}")

        self.domain, self.task = task.split("-", 1)
        self.env = self._make_env(self.seed)

        if self.visual:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.frame_stack * 3, self.image_size, self.image_size),
                dtype=np.uint8,
            )
        else:
            obs_shape = sum(
                np.prod(value.shape) for value in self.env.observation_spec().values()
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(obs_shape) * self.frame_stack,),
                dtype=np.float32,
            )
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.full(action_spec.shape, -1.0, dtype=action_spec.dtype),
            high=np.full(action_spec.shape, 1.0, dtype=action_spec.dtype),
            dtype=action_spec.dtype,
        )
        self.action_space.seed(self.seed)
        self.observation_space.seed(self.seed)

        self.max_ep_timesteps = (
            self.horizon + self.frame_skip - 1
        ) // self.frame_skip
        self.queue = deque(maxlen=self.frame_stack)
        self.t = 0

    def _make_env(self, seed: int):
        env = _clone_control_env(self.domain, self.task, seed)
        return action_scale.Wrapper(
            env,
            minimum=-1.0,
            maximum=1.0,
        )

    def get_obs(self, time_step: object):
        if self.visual:
            return self.render().transpose(2, 0, 1)
        return np.concatenate(
            [value.reshape(-1) for value in time_step.observation.values()]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        if seed is not None:
            self.seed = int(seed)
            self.env.task._random = np.random.RandomState(self.seed)
            self.action_space.seed(self.seed)
            self.observation_space.seed(self.seed)

        self.t = 0
        self.queue.clear()
        time_step = self.env.reset()
        obs = self.get_obs(time_step)
        for _ in range(self.frame_stack):
            self.queue.append(obs)
        return np.concatenate(self.queue).astype(
            self.observation_space.dtype,
            copy=False,
        ), {}

    def step(self, action: np.ndarray):
        self.t += 1
        action = np.asarray(action, dtype=self.action_space.dtype)

        reward = 0.0
        time_step = None
        for _ in range(self.frame_skip):
            time_step = self.env.step(action)
            reward += float(time_step.reward or 0.0)
            if time_step.last():
                break

        obs = self.get_obs(time_step)
        self.queue.append(obs)
        discount = 1.0 if time_step.discount is None else float(time_step.discount)
        terminated = bool(time_step.last() and discount == 0.0)
        truncated = bool(
            self.t >= self.max_ep_timesteps
            or (time_step.last() and not terminated)
        )
        stacked_obs = np.concatenate(self.queue).astype(
            self.observation_space.dtype,
            copy=False,
        )
        return stacked_obs, reward, terminated, truncated, {}

    def render(self, size: int | None = None):
        size = self.image_size if size is None else int(size)
        camera = dict(quadruped=2).get(self.domain, self.camera)
        return self.env.physics.render(size, size, camera_id=camera)

    def close(self):
        env = getattr(self, "env", None)
        if env is not None:
            close = getattr(env, "close", None)
            if close is not None:
                close()
