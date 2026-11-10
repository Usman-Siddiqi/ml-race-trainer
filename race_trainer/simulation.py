"""Batched car physics, sensors, rewards, and episode rules for N cars at once.

This is the only implementation of the driving rules. The Gymnasium environment is one
car in this simulator; headless training and evaluation run many.
"""

import numpy as np

from .config import OBSERVATION_SIZE, SENSOR_ANGLES_DEGREES, EnvConfig
from .tracks import TrackBatch, gate_crossed

SENSOR_ANGLES = np.deg2rad(SENSOR_ANGLES_DEGREES)
SENSOR_ANGLES.setflags(write=False)
REASONS = ("running", "crash", "finish", "stall", "time_limit")
_CAR_CORNERS = np.array([[-12.0, -6.0], [12.0, -6.0], [12.0, 6.0], [-12.0, 6.0]])


class RaceSim:
    def __init__(self, config=None, count=1):
        self.config = config or EnvConfig()
        self.count = count
        self.geometry = TrackBatch(count)
        self.position = np.zeros((count, 2))
        self.angle = np.zeros(count)
        self.speed = np.zeros(count)
        self.previous_action = np.zeros((count, 2))
        self.steps = np.zeros(count, dtype=np.int64)
        self.current_checkpoint = np.zeros(count, dtype=np.int64)
        self.checkpoints_passed = np.zeros(count, dtype=np.int64)
        self.gates_to_finish = np.ones(count, dtype=np.int64)
        self.route_length = np.ones(count)
        self.last_distance = np.zeros(count)
        self.route_distance = np.zeros(count)
        self.best_distance = np.zeros(count)
        self.last_progress_step = np.zeros(count, dtype=np.int64)
        self.episode_reward = np.zeros(count)
        self.crashed = np.zeros(count, dtype=bool)
        self.completed = np.zeros(count, dtype=bool)
        self.finished = np.ones(count, dtype=bool)
        self.reason = np.zeros(count, dtype=np.int64)
        self.sensor_distances = np.zeros((count, len(SENSOR_ANGLES)))
        self.sensor_ends = np.zeros((count, len(SENSOR_ANGLES), 2))

    def reset_env(self, index, track, rng=None, *, random_position=False):
        """Place one car on a track. A generator adds start jitter; without one it is exact."""
        cfg = self.config
        self.geometry.assign(index, track)
        start = track.start_distance
        if random_position and rng is not None:
            # Training starts anywhere so every corner is met early, not just the first.
            last = track.length if track.closed else start + 0.6 * (track.finish_distance - start)
            start = float(rng.uniform(start, last))
        position, tangent = track.sample(start)
        angle = np.arctan2(tangent[1], tangent[0])
        if cfg.random_start and rng is not None:
            position = position + np.array([-tangent[1], tangent[0]]) * rng.uniform(-6, 6)
            angle += rng.uniform(-0.06, 0.06)
        gates = len(track.gates)
        first = int(np.searchsorted(track.gate_distances, start, side="right"))
        if track.closed:
            first %= gates
            final = track.gate_distances[first - 1]
            lap = (final - start) % track.length or track.length
            self.gates_to_finish[index] = gates * cfg.laps
            self.route_length[index] = lap + (cfg.laps - 1) * track.length
        else:
            self.gates_to_finish[index] = gates - first
            self.route_length[index] = track.finish_distance - start
        self.position[index], self.angle[index], self.speed[index] = position, angle, 0.0
        self.previous_action[index] = 0.0
        self.steps[index] = self.checkpoints_passed[index] = self.last_progress_step[index] = 0
        self.current_checkpoint[index] = first
        self.last_distance[index] = track.project(position)[0]
        self.route_distance[index] = self.best_distance[index] = self.episode_reward[index] = 0.0
        self.crashed[index] = self.completed[index] = self.finished[index] = False
        self.reason[index] = 0

    def corners(self, position=None, angle=None):
        position = self.position if position is None else position
        angle = self.angle if angle is None else angle
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        x, y = _CAR_CORNERS[:, 0], _CAR_CORNERS[:, 1]
        return np.stack((x * c - y * s, x * s + y * c), axis=2) + position[:, None, :]

    def step(self, actions):
        """Advance every unfinished car one step. Returns rewards, terminated, truncated."""
        cfg = self.config
        geometry = self.geometry
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1, 1)
        throttle, steering = actions[:, 0], actions[:, 1]
        live = ~self.finished
        previous = self.position.copy()
        force = np.where(throttle >= 0, throttle * cfg.acceleration, throttle * cfg.braking)
        speed = np.clip(self.speed + (force - cfg.drag) * cfg.dt, 0, cfg.max_speed)
        self.speed = np.where(live, speed, self.speed)
        # Tyres limit cornering: above grip / speed the car runs wide instead of turning.
        yaw = self.speed / cfg.wheelbase * np.tan(steering * cfg.max_steering)
        limit = cfg.grip / np.maximum(self.speed, 1e-6)
        yaw = np.clip(yaw, -limit, limit)
        moving = live.copy()
        # Two short physics steps preserve corner collision checks at maximum speed.
        for _ in range(2):
            angle = (self.angle + yaw * cfg.dt / 2 + np.pi) % (2 * np.pi) - np.pi
            self.angle = np.where(moving, angle, self.angle)
            heading = np.column_stack((np.cos(self.angle), np.sin(self.angle)))
            travel = (self.speed * cfg.dt / 2)[:, None] * heading
            self.position = np.where(moving[:, None], self.position + travel, self.position)
            hit = moving & ~geometry.on_road(self.corners()).all(axis=1)
            self.crashed |= hit
            moving &= ~hit
        self.steps += live
        distance, _ = geometry.project(self.position)
        advance = distance - self.last_distance
        wrapped = (advance + geometry.length / 2) % geometry.length - geometry.length / 2
        advance = np.where(geometry.closed, wrapped, advance)
        # Near sharp bends projection can jump; never award more than actual plausible motion.
        reach = cfg.max_speed * cfg.dt
        advance = np.where(live, np.clip(advance, -reach, reach), 0.0)
        self.last_distance = np.where(live, distance, self.last_distance)
        self.route_distance += advance
        improved = self.route_distance > self.best_distance + 1
        self.best_distance = np.where(improved, self.route_distance, self.best_distance)
        self.last_progress_step = np.where(improved, self.steps, self.last_progress_step)
        reward = cfg.progress_reward * advance - cfg.steering_penalty * np.abs(
            steering - self.previous_action[:, 1]
        )
        self.previous_action = np.where(live[:, None], actions, self.previous_action)
        rows = np.arange(self.count)
        passed = (
            live
            & ~self.crashed
            & gate_crossed(
                previous,
                self.position,
                geometry.gate_positions[rows, self.current_checkpoint],
                geometry.gate_tangents[rows, self.current_checkpoint],
                geometry.half_width,
            )
        )
        self.checkpoints_passed += passed
        self.current_checkpoint = (self.current_checkpoint + passed) % geometry.gate_count
        completed = passed & (self.checkpoints_passed >= self.gates_to_finish)
        self.completed |= completed
        crashed = live & self.crashed
        reward = reward + cfg.finish_bonus * completed - cfg.crash_penalty * crashed
        reward = np.where(live, reward, 0.0)
        terminated = crashed | completed
        stalled = self.steps - self.last_progress_step >= cfg.stall_steps
        truncated = live & ~terminated & (stalled | (self.steps >= cfg.max_steps))
        self.reason = np.where(
            crashed,
            1,
            np.where(completed, 2, np.where(truncated, np.where(stalled, 3, 4), self.reason)),
        )
        self.finished |= terminated | truncated
        self.episode_reward += reward
        return reward, terminated, truncated

    def observe(self, rows=None):
        """Car-mounted inputs only: wall distances, speed, and the previous controls."""
        cfg = self.config
        select = slice(None) if rows is None else rows
        position, angle = self.position[select], self.angle[select]
        angles = angle[:, None] + SENSOR_ANGLES
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=2)
        distances = self.geometry.raycast(position, directions, cfg.sensor_range, rows)
        self.sensor_distances[select] = distances
        self.sensor_ends[select] = position[:, None, :] + directions * distances[:, :, None]
        observation = np.empty((len(position), OBSERVATION_SIZE), dtype=np.float32)
        rays = len(SENSOR_ANGLES)
        observation[:, :rays] = distances / cfg.sensor_range
        observation[:, rays] = self.speed[select] / cfg.max_speed
        observation[:, rays + 1] = self.previous_action[select, 1]
        observation[:, rays + 2] = self.previous_action[select, 0]
        return observation

    def progress(self):
        fraction = np.clip(self.route_distance / self.route_length, 0, 1)
        return np.where(self.completed, 1.0, fraction)

    def info(self, index):
        track = self.geometry.tracks[index]
        gates = len(track.gates)
        return {
            "track": track.name,
            "steps": int(self.steps[index]),
            "checkpoints_passed": int(self.checkpoints_passed[index]),
            "total_checkpoints": int(self.gates_to_finish[index]),
            "current_checkpoint": int(self.current_checkpoint[index]),
            "laps_completed": int(self.checkpoints_passed[index] // gates) if track.closed else 0,
            "progress": float(self.progress()[index]),
            "crashed": bool(self.crashed[index]),
            "completed": bool(self.completed[index]),
            "speed": float(self.speed[index]),
            "elapsed_seconds": float(self.steps[index] * self.config.dt),
            "termination_reason": REASONS[self.reason[index]],
        }
