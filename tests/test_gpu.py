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
from race_trainer.environment import VectorRaceEnv
from race_trainer.evaluation import evaluate_policy
from race_trainer.training import run_training

pytestmark = pytest.mark.gpu

EVERY = AgentConfig().train_every
BOUNDED = ["--envs", "4", "--max-steps", "40", "--batch-size", "16", "--learning-starts", "64"]
BOUNDED += ["--eval-every", "200", "--log-every", "100", "--track-pool", "4", "--cpu-threads", "1"]


def populated_agent(device="cuda"):
    config = AgentConfig(
        batch_size=128,
        replay_capacity=1024,
        learning_starts=128,
        train_every=8,
        target_update_every=2,
        actor_sync_every=2,
    )
    agent = DQNAgent(config, device, seed=17, num_envs=8)
    rng = np.random.default_rng(7)
    samples = rng.uniform(-1, 1, (41, 8, OBSERVATION_SIZE)).astype(np.float32)
    for i in range(40):
        ended = rng.random(8) < 0.1
        agent.observe(
            samples[i],
            rng.integers(0, 15, 8),
            rng.random(8),
            samples[i + 1],
            ended & (i % 2 == 0),
            ended,
        )
    return agent, samples[0, 0]


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cuda_learner_drives_from_a_cpu_copy_that_matches_a_cpu_agent():
    cpu = DQNAgent(seed=19)
    gpu = DQNAgent(device="cuda", seed=19)
    samples = np.random.default_rng(11).uniform(-1, 1, (16, OBSERVATION_SIZE)).astype(np.float32)
    with torch.inference_mode():
        expected = cpu.online(torch.from_numpy(samples))
        actual = gpu.online(torch.as_tensor(samples, device="cuda"))
    assert actual.is_cuda and not actual.requires_grad
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    # Actions come from the CPU copy, which must hold exactly the learner's weights.
    np.testing.assert_array_equal(gpu.act_batch(samples), cpu.act_batch(samples))
    assert all(not p.is_cuda and not p.requires_grad for p in gpu.actor.parameters())
    for key, value in gpu.online.state_dict().items():
        torch.testing.assert_close(gpu.actor.state_dict()[key], value.cpu(), rtol=0, atol=0)
    cfg = EnvConfig(max_steps=24)
    tracks = ("oval", "s_track", "square", "random:5")
    first = evaluate_policy(gpu, cfg, tracks, episodes=1, seed=100)
    assert first == evaluate_policy(gpu, cfg, tracks, episodes=1, seed=100)
    assert first == evaluate_policy(cpu, cfg, tracks, episodes=1, seed=100)
    assert first["summary"]["episodes"] == 4 and gpu.updates == 0


def test_cuda_updates_change_weights_and_synchronize_target_and_actor():
    agent, sample = populated_agent()
    assert agent.updates_due() == 24
    before = {key: value.clone() for key, value in agent.online.state_dict().items()}
    first_action = agent.act(sample)
    loss = agent.learn()
    assert math.isfinite(loss) and agent.updates == 1
    assert any(
        not torch.equal(value, before[key]) for key, value in agent.online.state_dict().items()
    )
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.is_cuda and torch.isfinite(p.grad).all() for p in agent.online.parameters())
    assert all(p.grad is None for p in agent.target.parameters())
    assert isinstance(agent.replay.states, np.ndarray)
    # One update is below the sync interval, so the driving copy is still the old policy.
    torch.testing.assert_close(
        agent.actor.state_dict()["value.bias"], before["value.bias"].cpu(), rtol=0, atol=0
    )
    assert agent.act(sample) == first_action
    for _ in range(3):
        assert math.isfinite(agent.learn())
    assert agent.updates == 4
    for key, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, agent.target.state_dict()[key], rtol=0, atol=0)
    agent.act(sample)
    for key, value in agent.online.state_dict().items():
        torch.testing.assert_close(agent.actor.state_dict()[key], value.cpu(), rtol=0, atol=0)
    for state in agent.optimizer.state.values():
        assert state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda


def test_learning_fits_a_predictable_reward_on_cuda():
    # With no bootstrap the target is the reward itself, here the first sensor reading.
    config = AgentConfig(
        batch_size=128, replay_capacity=1024, learning_starts=128, gamma=0.0, learning_rate=0.003
    )
    agent = DQNAgent(config, "cuda", seed=3, num_envs=8)
    rng = np.random.default_rng(5)
    for _ in range(64):
        states = rng.uniform(0, 1, (8, OBSERVATION_SIZE)).astype(np.float32)
        ended = np.zeros(8, dtype=bool)
        agent.observe(states, rng.integers(0, 15, 8), states[:, 0], states, ended, ended)
    losses = [agent.learn() for _ in range(200)]
    assert all(math.isfinite(loss) for loss in losses)
    assert np.mean(losses[-20:]) < 0.2 * np.mean(losses[:5])


def test_trained_cuda_checkpoint_restores_on_cpu_and_cuda(tmp_path):
    agent, sample = populated_agent()
    agent.learn()
    path = tmp_path / "latest.pt"
    train_cfg = TrainConfig(device="cuda", num_envs=8)
    save_checkpoint(path, agent, EnvConfig(), train_cfg, (0.0, 0.0, 0.0))
    payload = load_checkpoint(path)
    assert payload["cuda_rng"] and payload["cuda_rng"][0].device.type == "cpu"
    assert all(value.device.type == "cpu" for value in payload["agent"]["online"].values())
    restored = DQNAgent(agent.config, "cuda", seed=99, num_envs=8)
    restored.load_state_dict(payload["agent"])
    cpu = DQNAgent(agent.config, "cpu", seed=99, num_envs=8)
    cpu.load_state_dict(payload["agent"])
    assert cpu.env_steps == restored.env_steps == 320
    assert cpu.updates == restored.updates == 1
    gpu_policy, _ = load_policy(path, "cuda")
    cpu_policy, _ = load_policy(path, "cpu")
    assert gpu_policy.act(sample) == cpu_policy.act(sample) == restored.act(sample)
    for state in restored.optimizer.state.values():
        assert state["exp_avg"].is_cuda and state["step"].item() == 1
    # Identical replay RNG and Adam state must produce the same next CUDA update.
    expected_loss = agent.learn()
    assert restored.learn() == pytest.approx(expected_loss, rel=1e-6)
    assert cpu.learn() == pytest.approx(expected_loss, rel=1e-4)
    for key, value in agent.online.state_dict().items():
        torch.testing.assert_close(value, restored.online.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value.cpu(), cpu.online.state_dict()[key], rtol=1e-4, atol=1e-5)


def test_cuda_cli_training_and_resume_are_bounded(tmp_path, capsys):
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    train = ["train", "--device", "cuda", "--steps", "400", "--output", str(first), "--json"]
    assert main([*train, *BOUNDED]) == 0
    initial = load_checkpoint(first / "latest.pt")
    assert initial["completed_steps"] == 400
    assert initial["agent"]["updates"] == (400 - 64) // EVERY > 0
    config = json.loads((first / "config.json").read_text(encoding="utf-8"))
    assert config["resolved_device"] == "cuda"
    progress = [e for e in read_events(first / "metrics.jsonl") if e["event"] == "progress"]
    assert [event["steps"] for event in progress] == [100, 200, 300, 400]
    assert math.isfinite(progress[-1]["loss"]) and progress[-1]["updates"] == (400 - 64) // EVERY
    assert (first / "best.pt").exists() and len(read_events(first / "evaluations.jsonl")) == 2
    resume = ["train", "--resume", str(first / "latest.pt"), "--steps", "600", "--device", "cuda"]
    assert main([*resume, "--output", str(resumed), "--json"]) == 0
    final = load_checkpoint(resumed / "latest.pt")
    assert final["completed_steps"] == 600
    assert final["agent"]["updates"] == (600 - 64) // EVERY
    assert final["agent"]["replay"]["size"] == 150
    assert final["agent"]["config"] == initial["agent"]["config"]
    assert '"event": "finished"' in capsys.readouterr().out
    policy, _ = load_policy(resumed / "best.pt", "cuda")
    report = evaluate_policy(policy, EnvConfig(max_steps=30), ("oval", "random:2000000000"))
    assert report["summary"]["episodes"] == 2


def test_cuda_stop_request_saves_and_resumes(tmp_path, monkeypatch):
    output, resumed = tmp_path / "stopped", tmp_path / "resumed"
    config = TrainConfig(
        total_steps=4000,
        num_envs=4,
        track_pool=4,
        eval_tracks=("oval",),
        eval_every=10**6,
        log_every=200,
        device="cuda",
    )
    env_config = EnvConfig(max_steps=40)
    agent_config = AgentConfig(
        batch_size=16, replay_capacity=4000, learning_starts=64, train_every=16
    )
    step = VectorRaceEnv.step
    seen = []

    def request_stop(env, actions):
        seen.append(None)
        if len(seen) == 100:
            (output / "stop.request").touch()
        return step(env, actions)

    monkeypatch.setattr(VectorRaceEnv, "step", request_stop)
    result = run_training(config, env_config, agent_config, output=output)
    monkeypatch.setattr(VectorRaceEnv, "step", step)
    assert result["event"] == "stopped" and 400 <= result["steps"] < 480
    payload = load_checkpoint(output / "latest.pt")
    assert payload["agent"]["updates"] == (result["steps"] - 64) // 16 > 0
    result = run_training(
        replace(config, total_steps=800),
        env_config,
        agent_config,
        output=resumed,
        resume=output / "latest.pt",
    )
    final = load_checkpoint(resumed / "latest.pt")
    assert result["event"] == "finished" and final["completed_steps"] == 800
    assert final["agent"]["updates"] == (800 - 64) // 16
    assert (
        read_events(resumed / "metrics.jsonl")[0]["completed_steps"] == payload["completed_steps"]
    )


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
        return popen([*command, *BOUNDED], **kwargs)

    monkeypatch.setattr("race_trainer.app.subprocess.Popen", bounded_child)
    app = RaceApp()
    try:
        app.steps, app.device_index = 400, 2
        app.click("setup")
        app.click("start")
        deadline = time.monotonic() + 60
        while (app.busy or app.latest_event != "finished") and time.monotonic() < deadline:
            app.poll_job()
            app.draw()
            time.sleep(0.01)
        app.poll_job()
        assert app.process.returncode == 0, app.message
        assert app.latest_event == "finished", app.message
        assert len(app.progress) == 4 and app.last_progress["loss"] is not None
        assert app.model_path.exists() and app.last_evaluation["episodes"] == 16
        config = json.loads((app.output / "config.json").read_text(encoding="utf-8"))
        assert config["resolved_device"] == "cuda"
        payload = load_checkpoint(app.model_path)
        assert payload["agent"]["updates"] == (400 - 64) // EVERY
        assert payload["completed_steps"] == 400
    finally:
        if app.busy:
            app.process.terminate()
            app.process.wait(timeout=5)
        pygame.display.quit()


def test_benchmark_measures_simulator_and_both_learners(capsys):
    assert main(["benchmark", "--seconds", "0.2"]) == 0
    output = capsys.readouterr().out
    assert "64 cars" in output and "cuda batch  2048" in output and "cpu  batch  2048" in output
    assert "Suggested: --device" in output
