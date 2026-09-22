import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from race_trainer.config import TRACK_NAMES, EnvConfig
from race_trainer.environment import RaceCarEnv
from race_trainer.tracks import make_track


@pytest.mark.parametrize("name", TRACK_NAMES)
def test_gymnasium_contract_and_seed(name):
    env = RaceCarEnv(EnvConfig(track=name))
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_array_equal(first, second)
    assert env.observation_space.contains(first)
    env.close()


@pytest.mark.parametrize("name", TRACK_NAMES)
def test_sensors_intersect_the_actual_collision_boundaries(name):
    track = make_track(name)
    for distance in np.linspace(track.start_distance + 10, track.finish_distance - 10, 8):
        position, tangent = track.sample(distance)
        angle = math.atan2(tangent[1], tangent[0])
        directions = np.column_stack(
            (np.cos(angle + RaceCarEnv.sensor_angles), np.sin(angle + RaceCarEnv.sensor_angles))
        )
        lengths = track.raycast(position, directions, 1000)
        before = position + directions * (lengths - 0.01)[:, None]
        after = position + directions * (lengths + 0.01)[:, None]
        assert track.contains(before).all()
        assert not track.contains(after).any()


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
    assert reward < 0 and truncated and not terminated
    assert info["termination_reason"] == "stall"


def test_crash_is_terminal_and_invalid_actions_are_rejected():
    env = RaceCarEnv()
    env.reset(seed=0)
    for action in ([1], [np.nan, 0], [0, np.inf]):
        with pytest.raises(ValueError, match="finite"):
            env.step(action)
    env.position = np.array([0.0, 0.0])
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert terminated and not truncated and info["crashed"] and reward < 0


def test_sprint_rejects_multiple_laps():
    with pytest.raises(ValueError, match="sprint"):
        EnvConfig(track="s_track", laps=2)
