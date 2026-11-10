import math
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from race_trainer.config import OBSERVATION_SIZE, EnvConfig
from race_trainer.environment import RaceCarEnv
from race_trainer.simulation import SENSOR_ANGLES, RaceSim
from race_trainer.tracks import make_track

NAMES = ("oval", "s_track", "square", "random:0", "random:3")


@pytest.mark.parametrize("name", NAMES)
def test_gymnasium_contract_and_seed(name):
    env = RaceCarEnv(EnvConfig(track=name))
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_array_equal(first, second)
    assert env.observation_space.contains(first)
    env.close()


def test_observation_holds_only_car_mounted_measurements():
    env = RaceCarEnv(EnvConfig(track="random:1"))
    env.reset(seed=3)
    for action in ([1, 0.5], [1, -1], [-1, 0.5]):
        observation, *_ = env.step(action)
    rays = len(SENSOR_ANGLES)
    assert observation.shape == (OBSERVATION_SIZE,) == (rays + 3,)
    directions = np.column_stack(
        (np.cos(env.angle + SENSOR_ANGLES), np.sin(env.angle + SENSOR_ANGLES))
    )
    walls = env.track.raycast(env.position, directions, env.config.sensor_range)
    np.testing.assert_allclose(observation[:rays], walls / env.config.sensor_range, rtol=1e-6)
    assert observation[rays] == pytest.approx(env.speed / env.config.max_speed)
    # The last two inputs are the controls just applied: steering, then throttle.
    assert tuple(observation[rays + 1 :]) == (0.5, -1.0)
    # A car with the same pose, speed, and controls reads the same inputs: nothing in
    # the observation depends on lap progress, gates, or the centerline.
    other = RaceSim(env.config, 1)
    other.reset_env(0, env.track)
    other.position[0], other.angle[0] = env.position, env.angle
    other.speed[0], other.previous_action[0] = env.speed, (-1.0, 0.5)
    other.current_checkpoint[0], other.route_distance[0] = 3, 500.0
    np.testing.assert_array_equal(other.observe()[0], observation)


@pytest.mark.parametrize("name,count", [("oval", 12), ("s_track", 6), ("square", 12)])
def test_each_track_finishes_after_its_own_gates(name, count):
    env = RaceCarEnv(EnvConfig(track=name, random_start=False))
    env.reset()
    assert env.total_checkpoints == count
    for index, gate in enumerate(env.track.gates):
        # Cross real gates in sequence with one physics step; no policy or learning.
        env.position = gate.position - gate.tangent * 0.5
        env.angle = math.atan2(gate.tangent[1], gate.tangent[0])
        env.speed = 60
        env._last_distance = env.track.project(env.position)[0]
        _, _, terminated, truncated, info = env.step([0, 0])
        assert info["checkpoints_passed"] == index + 1
        assert not info["crashed"]
        assert not truncated
        assert terminated == (index == count - 1)
    assert info["completed"] and info["termination_reason"] == "finish"
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])


def test_reverse_gate_crossing_and_skipping_cannot_count():
    env = RaceCarEnv(EnvConfig(random_start=False))
    env.reset()
    gate = env.track.gates[0]
    assert not gate.crossed(gate.position + gate.tangent, gate.position - gate.tangent, 44)
    assert gate.crossed(gate.position - gate.tangent, gate.position + gate.tangent, 44)
    wrong = env.track.gates[2]
    env.position = wrong.position - wrong.tangent * 0.5
    env.angle = math.atan2(wrong.tangent[1], wrong.tangent[0])
    env.speed = 60
    _, _, _, _, info = env.step([0, 0])
    assert info["checkpoints_passed"] == 0


def test_braking_coasting_stall_and_time_limit():
    env = RaceCarEnv(EnvConfig(max_steps=2, random_start=False))
    env.reset()
    env.speed = 80
    env.step([0, 0])
    assert env.speed < 80
    speed = env.speed
    _, _, terminated, truncated, info = env.step([-1, 0])
    assert env.speed < speed
    assert truncated and not terminated and info["termination_reason"] == "time_limit"
    env = RaceCarEnv(EnvConfig(stall_steps=2))
    env.reset(seed=0)
    env.step([0, 0])
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert reward == 0 and truncated and not terminated
    assert info["termination_reason"] == "stall"


def test_rewards_follow_progress_smoothness_crash_and_finish():
    cfg = EnvConfig(random_start=False)
    env = RaceCarEnv(cfg)
    env.reset()
    env.speed = 90
    _, reward, *_ = env.step([0, 0])
    assert reward == pytest.approx(cfg.progress_reward * env.speed * cfg.dt, rel=0.02)
    _, turning, *_ = env.step([0, 1])
    assert turning < reward - cfg.steering_penalty * 0.9
    gate = env.track.gates[-1]
    env.reset()
    env.sim.checkpoints_passed[0], env.sim.current_checkpoint[0] = 11, 11
    env.position = gate.position - gate.tangent * 0.5
    env.angle = math.atan2(gate.tangent[1], gate.tangent[0])
    env.speed = 60
    env._last_distance = env.track.project(env.position)[0]
    _, reward, terminated, _, info = env.step([0, 0])
    assert terminated and info["completed"] and reward > cfg.finish_bonus


def test_crash_is_terminal_and_invalid_actions_are_rejected():
    env = RaceCarEnv()
    env.reset(seed=0)
    for action in ([1], [np.nan, 0], [0, np.inf]):
        with pytest.raises(ValueError, match="finite"):
            env.step(action)
    env.position = np.array([0.0, 0.0])
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert terminated and not truncated and info["crashed"]
    assert reward <= -env.config.crash_penalty + 0.5
    assert info["termination_reason"] == "crash"


def test_grip_limits_cornering_at_speed_but_not_when_slow():
    cfg = EnvConfig(random_start=False)
    turned = {}
    for speed in (40.0, 170.0):
        env = RaceCarEnv(cfg)
        env.reset()
        env.speed = speed
        before = env.angle
        env.step([0, 1])
        turned[speed] = (
            abs((env.angle - before + math.pi) % (2 * math.pi) - math.pi) / cfg.dt,
            env.speed,
        )
    rate, speed = turned[40.0]
    assert rate == pytest.approx(speed / cfg.wheelbase * math.tan(cfg.max_steering), rel=1e-6)
    rate, speed = turned[170.0]
    assert rate == pytest.approx(cfg.grip / speed, rel=1e-6)


def test_sprints_reject_multiple_laps():
    with pytest.raises(ValueError, match="sprint"):
        EnvConfig(track="s_track", laps=2)
    sprint = next(f"random:{s}" for s in range(50) if not make_track(f"random:{s}").closed)
    with pytest.raises(ValueError, match="sprint"):
        RaceCarEnv(EnvConfig(track=sprint, laps=2))


def test_many_cars_match_the_single_car_environment_exactly():
    names = ("oval", "random:2", "s_track", "random:5", "square", "random:7")
    cfg = EnvConfig(max_steps=150)
    rng = np.random.default_rng(11)
    actions = rng.uniform(-1, 1, (150, len(names), 2)).astype(np.float32)
    actions[:, :, 0] = np.abs(actions[:, :, 0])
    sim = RaceSim(cfg, len(names))
    singles = []
    for index, name in enumerate(names):
        sim.reset_env(index, make_track(name), np.random.default_rng(index))
        env = RaceCarEnv(replace(cfg, track=name))
        env.reset(seed=index)
        singles.append(env)
    np.testing.assert_array_equal(sim.observe(), [env._observation() for env in singles])
    endings = set()
    for step in range(150):
        rewards, terminated, truncated = sim.step(actions[step])
        batched = sim.observe()
        for index, env in enumerate(singles):
            if env._finished:
                # A finished car is frozen in the batch and contributes nothing.
                assert rewards[index] == 0 and not terminated[index] and not truncated[index]
                continue
            observation, reward, ended, cut, info = env.step(actions[step, index])
            np.testing.assert_array_equal(batched[index], observation)
            assert (reward, ended, cut) == (rewards[index], terminated[index], truncated[index])
            assert info == sim.info(index)
            if ended or cut:
                endings.add(info["termination_reason"])
    assert sim.finished.all() and "crash" in endings


def test_random_start_positions_cover_the_track_and_still_finish_one_lap():
    track = make_track("random:2")
    assert track.closed
    sim = RaceSim(EnvConfig(), 64)
    rng = np.random.default_rng(0)
    for index in range(64):
        sim.reset_env(index, track, rng, random_position=True)
    assert sim.geometry.on_road(sim.corners()).all()
    assert np.ptp(sim.last_distance) > 0.8 * track.length
    assert len(set(sim.current_checkpoint.tolist())) > len(track.gates) // 2
    assert (sim.gates_to_finish == len(track.gates)).all()
    spacing = track.length / len(track.gates)
    assert (sim.route_length > track.length - spacing - 1e-6).all()
    assert (sim.route_length <= track.length + 1e-6).all()


@pytest.mark.parametrize("field", ["dt", "max_speed", "drag", "grip"])
def test_environment_rejects_nonfinite_physics(field):
    with pytest.raises(ValueError, match="finite"):
        EnvConfig(**{field: float("nan")})
