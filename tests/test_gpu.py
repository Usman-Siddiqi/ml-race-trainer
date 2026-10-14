"""Explicit CUDA verification, with tiny disposable runs rather than policy training."""

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pygame
import pytest
import torch

from race_trainer.agent import DQNAgent
from race_trainer.app import RaceApp
from race_trainer.checkpoint import load_checkpoint, load_policy, save_checkpoint
from race_trainer.cli import main
from race_trainer.config import OBSERVATION_SIZE, AgentConfig, EnvConfig, TrainConfig
from race_trainer.evaluation import evaluate_policy
from race_trainer.training import run_training

pytestmark = pytest.mark.gpu


def populated_agent():
    config = AgentConfig(
        batch_size=128,
        replay_capacity=256,
        learning_starts=128,
        train_every=1,
        target_update_every=2,
    )
    agent = DQNAgent(config, "cuda", seed=17)
    samples = np.random.default_rng(7).uniform(-1, 1, (129, OBSERVATION_SIZE)).astype(np.float32)
    for i in range(128):
        agent.observe(samples[i], i % 9, (i % 7) / 10, samples[i + 1], i % 11 == 0)
    return agent, samples


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cuda_inference_matches_cpu_and_evaluates_every_track():
    cpu = DQNAgent(seed=19)
    gpu = DQNAgent(device="cuda", seed=19)
    samples = np.random.default_rng(11).uniform(-1, 1, (16, OBSERVATION_SIZE)).astype(np.float32)
    with torch.inference_mode():
        expected = cpu.online(torch.from_numpy(samples))
        actual = gpu.online(torch.as_tensor(samples, device="cuda"))
    assert actual.is_cuda and not actual.requires_grad
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    assert [gpu.act(sample) for sample in samples] == [cpu.act(sample) for sample in samples]
    cfg = EnvConfig(max_steps=24)
    tracks = ("oval", "s_track", "square")
    first = evaluate_policy(gpu, cfg, tracks, episodes=1, seed=100)
    assert first == evaluate_policy(gpu, cfg, tracks, episodes=1, seed=100)
    assert first == evaluate_policy(cpu, cfg, tracks, episodes=1, seed=100)
    assert first["summary"]["episodes"] == 3 and gpu.updates == 0


def test_cuda_updates_change_weights_and_synchronize_the_target():
    agent, _ = populated_agent()
    before = {key: value.clone() for key, value in agent.online.state_dict().items()}
    loss = agent.learn_if_due()
    assert math.isfinite(loss) and agent.updates == 1
    assert any(
        not torch.equal(value, before[key]) for key, value in agent.online.state_dict().items()
    )
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.is_cuda and torch.isfinite(p.grad).all() for p in agent.online.parameters())
    assert all(p.grad is None for p in agent.target.parameters())
    assert isinstance(agent.replay.states, np.ndarray)
    for _ in range(3):
        assert math.isfinite(agent.learn_if_due())
    assert agent.updates == 4
    for key, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, agent.target.state_dict()[key], rtol=0, atol=0)
    for state in agent.optimizer.state.values():
        assert state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda


def test_trained_cuda_checkpoint_restores_on_cpu_and_cuda(tmp_path):
    agent, samples = populated_agent()
    agent.learn_if_due()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, agent, EnvConfig(), TrainConfig(device="cuda"), 1, (0.0, 0.0, 0.0))
    payload = load_checkpoint(path)
    assert payload["cuda_rng"] and payload["cuda_rng"][0].device.type == "cpu"
    assert all(value.device.type == "cpu" for value in payload["agent"]["online"].values())
    restored = DQNAgent(agent.config, "cuda", seed=99)
    restored.load_state_dict(payload["agent"])
    cpu = DQNAgent(agent.config, "cpu", seed=99)
    cpu.load_state_dict(payload["agent"])
    assert cpu.env_steps == restored.env_steps == 128
    assert cpu.updates == restored.updates == 1
    gpu_policy, _ = load_policy(path, "cuda")
    cpu_policy, _ = load_policy(path, "cpu")
    assert gpu_policy.act(samples[0]) == cpu_policy.act(samples[0]) == agent.act(samples[0])
    for state in restored.optimizer.state.values():
        assert state["exp_avg"].is_cuda and state["step"].item() == 1
    # Identical replay RNG and Adam state must produce the same next CUDA update.
    expected_loss = agent.learn_if_due()
    assert restored.learn_if_due() == pytest.approx(expected_loss, rel=1e-6)
    assert cpu.learn_if_due() == pytest.approx(expected_loss, rel=1e-5)
    for key, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, restored.online.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value.cpu(), cpu.online.state_dict()[key], rtol=1e-5, atol=1e-6)


def test_cuda_cli_training_and_resume_are_bounded(tmp_path, capsys):
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    assert (
        main(
            [
                "train",
                "--device",
                "cuda",
                "--episodes",
                "3",
                "--max-steps",
                "24",
                "--batch-size",
                "8",
                "--learning-starts",
                "8",
                "--eval-every",
                "1",
                "--eval-episodes",
                "1",
                "--save-every",
                "1",
                "--cpu-threads",
                "1",
                "--output",
                str(first),
            ]
        )
        == 0
    )
    initial = load_checkpoint(first / "latest.pt")
    assert initial["completed_episodes"] == 3 and initial["episode_boundary"]
    assert 0 < initial["agent"]["updates"] <= 18
    assert initial["agent"]["env_steps"] <= 72
    config = json.loads((first / "config.json").read_text(encoding="utf-8"))
    assert config["resolved_device"] == "cuda"
    episodes = [
        event for event in read_events(first / "metrics.jsonl") if event["event"] == "episode"
    ]
    assert [event["track"] for event in episodes] == ["oval", "s_track", "square"]
    assert (first / "best.pt").exists() and len(read_events(first / "evaluations.jsonl")) == 3
    assert (
        main(
            [
                "train",
                "--resume",
                str(first / "latest.pt"),
                "--episodes",
                "4",
                "--device",
                "cuda",
                "--output",
                str(resumed),
            ]
        )
        == 0
    )
    final = load_checkpoint(resumed / "latest.pt")
    assert final["completed_episodes"] == 4 and final["episode_boundary"]
    assert initial["agent"]["updates"] < final["agent"]["updates"] <= 24
    assert final["agent"]["env_steps"] <= 96
    assert final["agent"]["replay"]["size"] > initial["agent"]["replay"]["size"]
    assert final["agent"]["config"] == initial["agent"]["config"]
    assert '"event": "finished"' in capsys.readouterr().out


def test_cuda_partial_stop_saves_and_resumes(tmp_path, monkeypatch):
    output, resumed = tmp_path / "stopped", tmp_path / "resumed"
    config = TrainConfig(episodes=2, tracks=("oval",), device="cuda", eval_episodes=1)
    env_config = EnvConfig(max_steps=40, random_start=False)
    agent_config = AgentConfig(batch_size=8, replay_capacity=128, learning_starts=8)
    observe = DQNAgent.observe

    def request_stop_after_warmup(agent, *args):
        observe(agent, *args)
        if agent.env_steps == 8:
            (output / "stop.request").touch()

    monkeypatch.setattr(DQNAgent, "observe", request_stop_after_warmup)
    monkeypatch.setattr(DQNAgent, "act", lambda *args, **kwargs: 4)  # Coast; cannot crash.
    result = run_training(config, env_config, agent_config, output=output)
    assert result["event"] == "stopped" and result["episodes"] == 0
    assert result["env_steps"] == 32
    payload = load_checkpoint(output / "latest.pt")
    assert not payload["episode_boundary"] and payload["agent"]["updates"] > 0
    result = run_training(
        replace(config, episodes=1),
        env_config,
        agent_config,
        output=resumed,
        resume=output / "latest.pt",
    )
    final = load_checkpoint(resumed / "latest.pt")
    assert result["event"] == "finished" and result["episodes"] == 1
    assert final["episode_boundary"] and final["agent"]["env_steps"] == 72
    assert final["agent"]["updates"] > payload["agent"]["updates"]
    assert read_events(resumed / "metrics.jsonl")[0]["resume_restarts_partial_episode"]


def test_desktop_monitor_runs_a_real_bounded_cuda_child(tmp_path, monkeypatch):
    import subprocess

    source = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(source), os.getenv("PYTHONPATH")]))
    )
    popen = subprocess.Popen

    def bounded_child(command, **kwargs):
        assert command[1:4] == ["-m", "race_trainer", "train"]
        return popen(
            [
                *command,
                "--max-steps",
                "24",
                "--batch-size",
                "8",
                "--learning-starts",
                "8",
                "--eval-every",
                "1",
                "--eval-episodes",
                "1",
                "--save-every",
                "1",
                "--cpu-threads",
                "1",
            ],
            **kwargs,
        )

    monkeypatch.setattr("race_trainer.app.subprocess.Popen", bounded_child)
    app = RaceApp()
    try:
        app.episodes, app.device_index, app.track_index = 2, 2, 2
        app.click("setup")
        app.click("start")
        deadline = time.monotonic() + 40
        while (app.busy or app.latest_event != "finished") and time.monotonic() < deadline:
            app.poll_job()
            app.draw()
            time.sleep(0.01)
        app.poll_job()
        assert app.process.returncode == 0, app.message
        assert app.latest_event == "finished", app.message
        assert len(app.rewards) == 2 and app.last_episode["loss"] is not None
        assert app.model_path.exists() and app.last_evaluation["episodes"] == 1
        config = json.loads((app.output / "config.json").read_text(encoding="utf-8"))
        assert config["resolved_device"] == "cuda"
        payload = load_checkpoint(app.model_path)
        assert 0 < payload["agent"]["updates"] <= 12 and payload["agent"]["env_steps"] <= 48
    finally:
        if app.busy:
            app.process.terminate()
            app.process.wait(timeout=5)
        pygame.display.quit()
