"""A deterministic Gymnasium simulator, independent of the display and learner."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import OBSERVATION_SIZE, EnvConfig
from .tracks import make_track


class RaceCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    sensor_angles = np.deg2rad([-90, -65, -40, -20, 0, 20, 40, 65, 90])

    def __init__(self, config=None, render_mode=None, *, track_type=None):
        self.config = config or EnvConfig(track=track_type or "oval")
        if track_type is not None and config is not None and track_type != config.track:
            raise ValueError("track_type and config.track disagree")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unknown render mode: {render_mode}")
        self.render_mode = render_mode
        self.track = make_track(self.config.track)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.array([0.0] * 10 + [-1.0] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(low, np.ones(OBSERVATION_SIZE, np.float32))
        self._renderer = None
        self._finished = True
        self.position = np.zeros(2)
        self.angle = self.speed = 0.0

    @property
    def total_checkpoints(self):
        return len(self.track.gates)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        position, tangent = self.track.sample(self.track.start_distance)
        self.position = position.copy()
        self.angle = math.atan2(tangent[1], tangent[0])
        if self.config.random_start:
            lateral = self.np_random.uniform(-6, 6)
            self.position += np.array([-tangent[1], tangent[0]]) * lateral
            self.angle += self.np_random.uniform(-0.06, 0.06)
        self.speed = 0.0
        self.steps = self.current_checkpoint = self.checkpoints_passed = self.laps_completed = 0
        self.crashed = self.completed = False
        self._finished = False
        self._previous_steering = 0.0
        self._last_distance, _ = self.track.project(self.position)
        self.route_distance = self.best_distance = 0.0
        self._last_progress_step = 0
        self.episode_reward = 0.0
        return self._observation(), self._info()

    def corners(self):
        local = np.array([[-12, -6], [12, -6], [12, 6], [-12, 6]])
        c, s = math.cos(self.angle), math.sin(self.angle)
        return local @ np.array([[c, s], [-s, c]]) + self.position

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite action values: throttle/brake and steering")
        throttle, steering = np.clip(action, -1, 1)
        previous = self.position.copy()
        cfg = self.config
        force = throttle * (cfg.acceleration if throttle >= 0 else cfg.braking) - cfg.drag
        self.speed = float(np.clip(self.speed + force * cfg.dt, 0, cfg.max_speed))
        # Two short physics steps preserve corner collision checks at maximum speed.
        for _ in range(2):
            self.angle += (
                self.speed / cfg.wheelbase * math.tan(steering * cfg.max_steering) * cfg.dt / 2
            )
            self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi
            self.position += (
                self.speed * cfg.dt / 2 * np.array([math.cos(self.angle), math.sin(self.angle)])
            )
            if not self.track.contains(self.corners()).all():
                self.crashed = True
                break
        self.steps += 1
        distance, _ = self.track.project(self.position)
        advance = distance - self._last_distance
        if self.track.closed:
            advance = (advance + self.track.length / 2) % self.track.length - self.track.length / 2
        # Near sharp bends projection can jump; never award more than actual plausible motion.
        advance = float(np.clip(advance, -cfg.max_speed * cfg.dt, cfg.max_speed * cfg.dt))
        self._last_distance = distance
        self.route_distance += advance
        if self.route_distance > self.best_distance + 1:
            self.best_distance = self.route_distance
            self._last_progress_step = self.steps
        reward = advance / 12 - 0.02 - 0.01 * abs(steering - self._previous_steering)
        self._previous_steering = float(steering)
        gate = self.track.gates[self.current_checkpoint]
        if not self.crashed and gate.crossed(previous, self.position, self.track.half_width):
            self.checkpoints_passed += 1
            self.current_checkpoint += 1
            reward += 2.0
            if self.current_checkpoint == self.total_checkpoints:
                self.laps_completed += 1
                self.current_checkpoint = 0
                self.completed = self.laps_completed >= cfg.laps
                if self.completed:
                    reward += 20.0
        if self.crashed:
            reward -= 10.0
        terminated = self.crashed or self.completed
        stalled = self.steps - self._last_progress_step >= cfg.stall_steps
        truncated = not terminated and (self.steps >= cfg.max_steps or stalled)
        self._finished = terminated or truncated
        reason = (
            "crash"
            if self.crashed
            else "finish"
            if self.completed
            else "stall"
            if stalled
            else "time_limit"
            if truncated
            else "running"
        )
        self.episode_reward += float(reward)
        return (
            self._observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(reason),
        )

    def _observation(self):
        angles = self.angle + self.sensor_angles
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self.sensor_distances = self.track.raycast(
            self.position, directions, self.config.sensor_range
        )
        self.sensor_ends = self.position + directions * self.sensor_distances[:, None]
        distance, lateral = self.track.project(self.position)
        lookahead, _ = self.track.sample(distance + max(45.0, self.speed * 0.4))
        relative_heading = math.atan2(*(lookahead - self.position)[::-1]) - self.angle
        return np.array(
            [
                *(self.sensor_distances / self.config.sensor_range),
                self.speed / self.config.max_speed,
                math.sin(relative_heading),
                math.cos(relative_heading),
                np.clip(lateral / self.track.half_width, -1, 1),
            ],
            dtype=np.float32,
        )

    def _info(self, reason="running"):
        route_length = (self.track.finish_distance - self.track.start_distance) * self.config.laps
        return {
            "track": self.track.name,
            "steps": self.steps,
            "checkpoints_passed": self.checkpoints_passed,
            "total_checkpoints": self.total_checkpoints * self.config.laps,
            "current_checkpoint": self.current_checkpoint,
            "laps_completed": self.laps_completed,
            "progress": 1.0
            if self.completed
            else float(np.clip(self.route_distance / route_length, 0, 1)),
            "crashed": self.crashed,
            "completed": self.completed,
            "speed": self.speed,
            "elapsed_seconds": self.steps * self.config.dt,
            "termination_reason": reason,
        }

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
