"""Serializable contracts shared by simulation, checkpoints, and every entry point."""

import math
from dataclasses import dataclass

# Hand-built layouts. Training uses generated tracks by default, so these are held out.
TRACK_NAMES = ("oval", "s_track", "square")
RANDOM_PREFIX = "random:"
# "random" (training only) means: draw each episode from a pool of generated tracks.
RANDOM_POOL = "random"
# Generated-track seeds are split so evaluation layouts are never trained on.
VALIDATION_SEED_BASE = 1_000_000_000
TEST_SEED_BASE = 2_000_000_000
MAX_TRACK_SEED = 2**31 - 1

OBSERVATION_VERSION = 3
ACTION_VERSION = 3
# Car-mounted inputs only: wall-distance rays, speed, and the previous controls.
SENSOR_ANGLES_DEGREES = (-90, -70, -50, -35, -22, -13, -6, 0, 6, 13, 22, 35, 50, 70, 90)
OBSERVATION_SIZE = len(SENSOR_ANGLES_DEGREES) + 3


def track_seed(spec):
    """Return the generator seed of a ``random:<seed>`` spec, or None for other specs."""
    if not isinstance(spec, str) or not spec.startswith(RANDOM_PREFIX):
        return None
    digits = spec[len(RANDOM_PREFIX) :]
    if not digits.isdecimal() or int(digits) > MAX_TRACK_SEED:
        raise ValueError(f"Invalid generated track {spec!r}; use random:<seed> up to 2^31-1")
    return int(digits)


def check_track(spec, *, allow_pool=False):
    if spec in TRACK_NAMES or (allow_pool and spec == RANDOM_POOL):
        return spec
    if track_seed(spec) is None:
        choices = ", ".join(TRACK_NAMES)
        raise ValueError(f"Unknown track {spec!r}; choose {choices}, or random:<seed>")
    return spec


def generated_tracks(base, count):
    return tuple(f"{RANDOM_PREFIX}{base + index}" for index in range(count))


def validation_tracks(count=16):
    """Unseen layouts used to pick the best checkpoint during training."""
    return generated_tracks(VALIDATION_SEED_BASE, count)


def heldout_tracks(count=20):
    """Unseen layouts reserved for the final generalization report."""
    return TRACK_NAMES + generated_tracks(TEST_SEED_BASE, count)


@dataclass(frozen=True)
class EnvConfig:
    track: str = "oval"
    laps: int = 1
    max_steps: int = 2000
    stall_steps: int = 120
    random_start: bool = True
    dt: float = 1 / 30
    max_speed: float = 180.0
    acceleration: float = 120.0
    braking: float = 220.0
    drag: float = 15.0
    wheelbase: float = 26.0
    max_steering: float = 0.55
    grip: float = 260.0
    sensor_range: float = 220.0
    progress_reward: float = 0.05
    steering_penalty: float = 0.01
    crash_penalty: float = 5.0
    finish_bonus: float = 5.0

    def __post_init__(self):
        numeric = (
            "dt",
            "max_speed",
            "acceleration",
            "braking",
            "drag",
            "wheelbase",
            "max_steering",
            "grip",
            "sensor_range",
            "progress_reward",
            "steering_penalty",
            "crash_penalty",
            "finish_bonus",
        )
        for key in numeric:
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        check_track(self.track)
        if self.laps < 1 or self.max_steps < 1 or self.stall_steps < 1:
            raise ValueError("laps, max_steps, and stall_steps must be positive")
        if self.track == "s_track" and self.laps != 1:
            raise ValueError("The S-track is a sprint; it supports one finish, not repeated laps")
        for key in (
            "dt",
            "max_speed",
            "acceleration",
            "braking",
            "wheelbase",
            "grip",
            "sensor_range",
            "progress_reward",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.max_steering < 1.2 or self.drag < 0:
            raise ValueError("Invalid steering angle or drag")
        if min(self.steering_penalty, self.crash_penalty, self.finish_bonus) < 0:
            raise ValueError("Reward penalties and bonuses must be nonnegative")
        if self.max_speed * self.dt > 10:
            raise ValueError("max_speed * dt must be <= 10 pixels to preserve collision resolution")


@dataclass(frozen=True)
class AgentConfig:
    hidden_size: int = 256
    learning_rate: float = 0.0003
    final_learning_rate: float = 0.00003
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 2048
    replay_capacity: int = 400_000
    learning_starts: int = 20_000
    train_every: int = 64
    target_update_every: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.004
    epsilon_spread: float = 15.0
    exploration_steps: int = 300_000
    exploration_hold: int = 4
    actor_sync_every: int = 4
    gradient_clip: float = 10.0

    def __post_init__(self):
        for key in (
            "learning_rate",
            "final_learning_rate",
            "gamma",
            "epsilon_start",
            "epsilon_end",
            "epsilon_spread",
            "gradient_clip",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        for key in (
            "hidden_size",
            "n_step",
            "batch_size",
            "replay_capacity",
            "learning_starts",
            "train_every",
            "target_update_every",
            "exploration_steps",
            "exploration_hold",
            "actor_sync_every",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must hold at least one batch")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Expected 0 <= epsilon_end <= epsilon_start <= 1")
        if self.epsilon_spread < 1 or self.epsilon_end * self.epsilon_spread > 1:
            raise ValueError("epsilon_spread must be >= 1 and keep every epsilon within 0..1")
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer parameters")
        if not 0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError("final_learning_rate must be positive and at most learning_rate")


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 1_000_000
    num_envs: int = 64
    tracks: tuple[str, ...] = (RANDOM_POOL,)
    track_pool: int = 512
    eval_tracks: tuple[str, ...] = validation_tracks()
    seed: int = 42
    eval_every: int = 100_000
    eval_episodes: int = 1
    log_every: int = 20_000
    steering_noise: float = 0.2
    device: str = "auto"
    cpu_threads: int = 0

    def __post_init__(self):
        object.__setattr__(self, "tracks", tuple(dict.fromkeys(self.tracks)))
        object.__setattr__(self, "eval_tracks", tuple(dict.fromkeys(self.eval_tracks)))
        if not self.tracks or not self.eval_tracks:
            raise ValueError("Choose at least one training track and one evaluation track")
        for spec in self.tracks:
            check_track(spec, allow_pool=True)
        for spec in self.eval_tracks:
            check_track(spec)
        for key in (
            "total_steps",
            "num_envs",
            "track_pool",
            "eval_every",
            "eval_episodes",
            "log_every",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.track_pool > VALIDATION_SEED_BASE:
            raise ValueError("track_pool would overlap the held-out validation seeds")
        if not 0 <= self.steering_noise <= 1:
            raise ValueError("steering_noise must be between 0 and 1")
        if self.seed < 0 or self.cpu_threads < 0:
            raise ValueError("seed and cpu_threads must be nonnegative (0 threads = automatic)")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")

    def training_specs(self):
        """Expand the generated pool into explicit specs, e.g. random:0..random:511."""
        specs = []
        for spec in self.tracks:
            specs.extend(generated_tracks(0, self.track_pool) if spec == RANDOM_POOL else [spec])
        return tuple(dict.fromkeys(specs))
