import json

import pytest
import torch

from race_trainer import device
from race_trainer.config import EnvConfig
from race_trainer.evaluation import evaluate_policy, ranking


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


def test_evaluation_is_repeatable_and_reports_failures_honestly():
    class CoastPolicy:
        def act(self, observation, *, explore=False):
            assert not explore
            return 4

    cfg = EnvConfig(max_steps=3)
    args = (CoastPolicy(), cfg, ("oval", "s_track", "square"), 2, 100)
    first, second = evaluate_policy(*args), evaluate_policy(*args)
    assert first == second
    assert first["summary"]["episodes"] == 6
    assert first["summary"]["completion_rate"] == 0
    assert first["summary"]["mean_finish_seconds"] is None
    assert all(r["termination_reason"] == "time_limit" for r in first["episodes"])
    assert ranking(first)[:2] == (0.0, 0.0)
    json.dumps(first, allow_nan=False)
