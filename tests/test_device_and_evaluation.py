import json

import numpy as np
import pytest
import torch

from race_trainer import device
from race_trainer.agent import ACTIONS
from race_trainer.config import EnvConfig
from race_trainer.environment import RaceCarEnv
from race_trainer.evaluation import evaluate_policy, ranking

COAST = 7
FULL_THROTTLE = 2


@pytest.mark.parametrize("power", [False, None])
def test_auto_avoids_cuda_probe_on_battery_or_unknown_power(monkeypatch, power):
    monkeypatch.setattr(device, "ac_power_connected", lambda: power)

    def forbidden():
        raise AssertionError("Auto should not probe CUDA on battery")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert device.choose_device("auto").name == "cpu"
    assert device.choose_device("cpu").name == "cpu"


def test_cuda_selection_is_explicit_and_missing_cuda_is_actionable(monkeypatch):
    monkeypatch.setattr(device, "ac_power_connected", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.choose_device("auto").name == "cuda"
    assert device.choose_device("cuda").name == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.choose_device("auto").name == "cpu"
    with pytest.raises(RuntimeError, match="uv sync"):
        device.choose_device("cuda")
    assert device.default_cpu_threads("cuda") >= 1 and device.default_cpu_threads("cpu") >= 1


def test_evaluation_is_repeatable_and_reports_failures_honestly():
    class CoastPolicy:
        def act(self, observation, *, explore=False):
            assert not explore
            return COAST

    cfg = EnvConfig(max_steps=3)
    tracks = ("oval", "s_track", "square", "random:4")
    args = (CoastPolicy(), cfg, tracks, 2, 100)
    first, second = evaluate_policy(*args), evaluate_policy(*args)
    assert first == second
    assert first["summary"]["episodes"] == 8
    assert first["summary"]["completion_rate"] == 0
    assert first["summary"]["mean_finish_seconds"] is None
    assert set(first["tracks"]) == set(tracks)
    assert all(r["termination_reason"] == "time_limit" for r in first["episodes"])
    assert [r["seed"] for r in first["episodes"][:2]] == [100, 101]
    assert ranking(first)[:2] == (0.0, 0.0)
    json.dumps(first, allow_nan=False)


def test_batched_evaluation_matches_driving_each_episode_alone():
    class FloorIt:
        def act_batch(self, observations, *, explore=False):
            return np.full(len(observations), FULL_THROTTLE)

    cfg = EnvConfig(max_steps=400)
    report = evaluate_policy(FloorIt(), cfg, ("oval", "random:6"), 2, 7)
    assert report["summary"]["crash_rate"] == 1
    for record in report["episodes"]:
        env = RaceCarEnv(EnvConfig(track=record["track"], max_steps=400))
        env.reset(seed=record["seed"])
        total, ended = 0.0, False
        while not ended:
            _, reward, terminated, truncated, info = env.step(ACTIONS[FULL_THROTTLE])
            total += reward
            ended = terminated or truncated
        assert record["steps"] == info["steps"] and record["progress"] == info["progress"]
        assert record["reward"] == pytest.approx(total)
        assert 0 < record["mean_speed"] <= cfg.max_speed
