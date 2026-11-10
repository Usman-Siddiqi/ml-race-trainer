"""Gymnasium environments over the shared simulator, independent of display and learner."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import OBSERVATION_SIZE, EnvConfig
from .simulation import SENSOR_ANGLES, RaceSim
from .tracks import make_track


def _scalar(name, cast=float):
    def read(self):
        return cast(getattr(self.sim, name)[0])

    def write(self, value):
        getattr(self.sim, name)[0] = value

    return property(read, write)


class RaceCarEnv(gym.Env):
    """One car. Rendering, manual driving, and replay read its state directly."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    sensor_angles = SENSOR_ANGLES

    def __init__(self, config=None, render_mode=None, *, track_type=None):
        self.config = config or EnvConfig(track=track_type or "oval")
        if track_type is not None and config is not None and track_type != config.track:
            raise ValueError("track_type and config.track disagree")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unknown render mode: {render_mode}")
        self.render_mode = render_mode
        self.sim = RaceSim(self.config, 1)
        self.set_track(self.config.track)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.array([0.0] * (OBSERVATION_SIZE - 2) + [-1.0] * 2, dtype=np.float32)
        self.observation_space = spaces.Box(low, np.ones(OBSERVATION_SIZE, np.float32))
        self._renderer = None

    angle = _scalar("angle")
    speed = _scalar("speed")
    steps = _scalar("steps", int)
    current_checkpoint = _scalar("current_checkpoint", int)
    checkpoints_passed = _scalar("checkpoints_passed", int)
    crashed = _scalar("crashed", bool)
    completed = _scalar("completed", bool)
    episode_reward = _scalar("episode_reward")
    _finished = _scalar("finished", bool)
    _last_distance = _scalar("last_distance")

    @property
    def position(self):
        return self.sim.position[0]

    @position.setter
    def position(self, value):
        self.sim.position[0] = value

    @property
    def sensor_distances(self):
        return self.sim.sensor_distances[0]

    @property
    def sensor_ends(self):
        return self.sim.sensor_ends[0]

    @property
    def total_checkpoints(self):
        return len(self.track.gates)

    def set_track(self, name):
        """Switch layout; takes effect at the next reset."""
        self.track = make_track(name)
        if not self.track.closed and self.config.laps != 1:
            raise ValueError("A sprint supports one finish, not repeated laps")
        self.sim.finished[0] = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset_env(0, self.track, self.np_random)
        return self._observation(), self._info()

    def corners(self):
        return self.sim.corners()[0]

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite action values: throttle/brake and steering")
        reward, terminated, truncated = self.sim.step(action[None])
        return (
            self._observation(),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            self._info(),
        )

    def _observation(self):
        return self.sim.observe()[0]

    def _info(self):
        return self.sim.info(0)

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .rendering import RaceRenderer

            self._renderer = RaceRenderer(self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class VectorRaceEnv:
    """Many cars for headless training. A finished car restarts on a freshly drawn track."""

    def __init__(self, config, tracks, count, seed=0, *, random_position=True):
        self.config, self.count = config, count
        self.track_names = tuple(tracks)
        if not self.track_names:
            raise ValueError("Training needs at least one track")
        self.random_position = random_position
        self.rng = np.random.default_rng(seed)
        self.sim = RaceSim(config, count)

    def _restart(self, index):
        name = self.track_names[int(self.rng.integers(len(self.track_names)))]
        self.sim.reset_env(index, make_track(name), self.rng, random_position=self.random_position)

    def reset(self):
        for index in range(self.count):
            self._restart(index)
        return self.sim.observe()

    def step(self, actions):
        """Returns (observations, rewards, terminated, truncated, next_observations, episodes).

        ``next_observations`` are the true successors, including final ones, for replay.
        ``observations`` are what to act on next: restarted cars see their new track.
        ``episodes`` summarizes the cars that just ended, or is None.
        """
        sim = self.sim
        rewards, terminated, truncated = sim.step(actions)
        successors = sim.observe()
        ended = np.flatnonzero(terminated | truncated)
        if not len(ended):
            return successors, rewards, terminated, truncated, successors, None
        episodes = {
            "progress": sim.progress()[ended],
            "completed": sim.completed[ended].copy(),
            "crashed": sim.crashed[ended].copy(),
            "reward": sim.episode_reward[ended].copy(),
            "steps": sim.steps[ended].copy(),
        }
        for index in ended:
            self._restart(index)
        observations = successors.copy()
        observations[ended] = sim.observe(ended)
        return observations, rewards, terminated, truncated, successors, episodes
