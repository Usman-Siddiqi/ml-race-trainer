import json

import numpy as np
import pytest
import torch

from race_trainer.agent import DQNAgent
from race_trainer.checkpoint import FORMAT_VERSION, load_checkpoint, load_policy, save_checkpoint
from race_trainer.config import (
    ACTION_VERSION,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    AgentConfig,
    EnvConfig,
    TrainConfig,
)

SMALL = AgentConfig(hidden_size=8, replay_capacity=8, batch_size=2)
SCORE = (0.0, 0.5, 4.0)


def stepped_agent(seed=8):
    agent = DQNAgent(SMALL, seed=seed, num_envs=2)
    observations = np.zeros((2, OBSERVATION_SIZE), np.float32)
    agent.observe(observations, [1, 2], [1.0, 0.5], observations, [False, True], [False, True])
    return agent, observations


def fake_adam_state(agent):
    """Populate Adam metadata without performing any optimizer update."""
    parameter = next(agent.online.parameters())
    agent.optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
    }
    return parameter


def test_checkpoint_roundtrip_restores_training_state_without_training(tmp_path):
    agent, observations = stepped_agent()
    fake_adam_state(agent)
    path = tmp_path / "latest.pt"
    env_cfg = EnvConfig(track="s_track")
    train_cfg = TrainConfig(tracks=("random", "oval"), num_envs=2, track_pool=3)
    save_checkpoint(path, agent, env_cfg, train_cfg, SCORE)
    payload = load_checkpoint(path)
    restored = DQNAgent(SMALL, seed=99, num_envs=2)
    restored.load_state_dict(payload["agent"])
    assert restored.env_steps == 2 and restored.replay.size == 1
    assert restored.rng.bit_generator.state == agent.rng.bit_generator.state
    assert payload["env_config"]["track"] == "s_track"
    assert payload["completed_steps"] == 2 and tuple(payload["best_score"]) == SCORE
    saved = TrainConfig(**payload["train_config"])
    assert saved == train_cfg
    assert saved.training_specs() == ("random:0", "random:1", "random:2", "oval")
    state_of_adam = restored.optimizer.state[next(restored.online.parameters())]
    assert state_of_adam["step"].item() == 3
    policy, _ = load_policy(path, "cpu")
    assert policy.act(observations[0]) == agent.act(observations[0])
    np.testing.assert_array_equal(policy.act_batch(observations), agent.act_batch(observations))
    with pytest.raises(ValueError, match="inference"):
        policy.act(observations[0], explore=True)


def test_legacy_or_wrong_contract_is_rejected_clearly(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"fc1.weight": torch.zeros(4, 7)}, path)
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(path)
    torch.save({"format_version": 900}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path)
    # A version 2 model used centerline guidance inputs and cannot drive from sensors.
    torch.save({"format_version": 2, "observation_version": 2, "action_version": 2}, path)
    with pytest.raises(ValueError, match="retrained"):
        load_checkpoint(path)


def test_optimizer_restores_do_not_share_state_with_each_other_or_the_checkpoint():
    original = DQNAgent(SMALL)
    parameter = fake_adam_state(original)
    checkpoint = original.state_dict()
    left, right = DQNAgent(SMALL), DQNAgent(SMALL)
    left.load_state_dict(checkpoint)
    right.load_state_dict(checkpoint)
    left_state = left.optimizer.state[next(left.online.parameters())]
    right_state = right.optimizer.state[next(right.online.parameters())]
    left_state["step"].add_(1)
    left_state["exp_avg"].fill_(0.5)
    assert right_state["step"].item() == original.optimizer.state[parameter]["step"].item() == 3
    assert not torch.count_nonzero(right_state["exp_avg"])
    assert not torch.count_nonzero(original.optimizer.state[parameter]["exp_avg"])


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous checkpoint")
    agent = DQNAgent(SMALL)

    def broken_save(payload, temporary):
        from pathlib import Path

        Path(temporary).write_bytes(b"partial")
        raise OSError("disk unavailable")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk unavailable"):
        save_checkpoint(path, agent, EnvConfig(), TrainConfig(), SCORE)
    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_resume_dry_run_preserves_saved_settings_without_constructing_an_agent(
    tmp_path, monkeypatch, capsys
):
    from race_trainer.cli import main

    agent, _ = stepped_agent()
    path = tmp_path / "resume.pt"
    train_cfg = TrainConfig(tracks=("square",), seed=9, num_envs=2, total_steps=100)
    save_checkpoint(path, agent, EnvConfig(max_steps=50), train_cfg, SCORE)

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume dry run must not construct an agent or train")

    monkeypatch.setattr(DQNAgent, "__init__", forbidden)
    resume = ["train", "--resume", str(path), "--dry-run", "--device", "cpu"]
    assert main([*resume, "--steps", "200"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["training"]["seed"] == 9 and config["training"]["total_steps"] == 200
    assert config["training"]["tracks"] == ["square"] and config["training"]["num_envs"] == 2
    assert config["environment"]["max_steps"] == 50
    assert config["agent"]["hidden_size"] == 8
    # The target is a total; it must exceed what is done, and saved settings are fixed.
    assert main([*resume, "--steps", "2"]) == 2
    assert main([*resume, "--steps", "200", "--envs", "4"]) == 2
    assert main([*resume, "--steps", "200", "--tracks", "oval"]) == 2


def test_missing_agent_state_is_a_clear_error(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_VERSION,
            "env_config": {},
            "train_config": {},
            "agent": {},
            "completed_steps": 0,
            "best_score": [0.0, 0.0, 0.0],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
        },
        path,
    )
    with pytest.raises(ValueError, match="required agent state"):
        load_checkpoint(path)
