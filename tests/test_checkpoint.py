import numpy as np
import pytest
import torch

from race_trainer.agent import DQNAgent
from race_trainer.checkpoint import load_checkpoint, load_policy, save_checkpoint
from race_trainer.config import OBSERVATION_SIZE, AgentConfig, EnvConfig, TrainConfig


def test_checkpoint_roundtrip_restores_training_state_without_training(tmp_path):
    config = AgentConfig(hidden_size=8, replay_capacity=4, batch_size=2)
    agent = DQNAgent(config, seed=8)
    state = np.zeros(OBSERVATION_SIZE, np.float32)
    agent.observe(state, 1, 1.0, state, False)
    # Populate Adam metadata without performing any optimizer update.
    parameter = next(agent.online.parameters())
    agent.optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
    }
    path = tmp_path / "latest.pt"
    env_cfg, train_cfg = EnvConfig(track="s_track"), TrainConfig(tracks=("s_track",))
    save_checkpoint(path, agent, env_cfg, train_cfg, 12, (0.0, 0.5, 4.0))
    payload = load_checkpoint(path)
    restored = DQNAgent(config, seed=99)
    restored.load_state_dict(payload["agent"])
    assert restored.env_steps == 1 and restored.replay.size == 1
    assert restored.rng.bit_generator.state == agent.rng.bit_generator.state
    assert payload["env_config"]["track"] == "s_track"
    assert payload["completed_episodes"] == 12 and payload["episode_boundary"]
    state_of_adam = restored.optimizer.state[next(restored.online.parameters())]
    assert state_of_adam["step"].item() == 3
    policy, _ = load_policy(path, "cpu")
    assert policy.act(state) == agent.act(state)


def test_legacy_or_wrong_contract_is_rejected_clearly(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"fc1.weight": torch.zeros(4, 7)}, path)
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(path)
    torch.save({"format_version": 900}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path)


def test_optimizer_restores_do_not_share_state_with_each_other_or_the_checkpoint():
    config = AgentConfig(hidden_size=8, replay_capacity=4, batch_size=2)
    original = DQNAgent(config)
    parameter = next(original.online.parameters())
    original.optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
    }
    checkpoint = original.state_dict()
    left, right = DQNAgent(config), DQNAgent(config)
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
    agent = DQNAgent(AgentConfig(hidden_size=8, replay_capacity=4, batch_size=2))

    def broken_save(payload, temporary):
        from pathlib import Path

        Path(temporary).write_bytes(b"partial")
        raise OSError("disk unavailable")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk unavailable"):
        save_checkpoint(path, agent, EnvConfig(), TrainConfig(), 0, (0.0, 0.0, 0.0))
    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_resume_dry_run_preserves_saved_settings_without_constructing_an_agent(
    tmp_path, monkeypatch, capsys
):
    from race_trainer.cli import main

    agent = DQNAgent(AgentConfig(hidden_size=8, replay_capacity=4, batch_size=2))
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        agent,
        EnvConfig(track="square"),
        TrainConfig(tracks=("square",), seed=9),
        10,
        (0.0, 0.0, 0.0),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume dry run must not construct an agent or train")

    monkeypatch.setattr(DQNAgent, "__init__", forbidden)
    assert main(["train", "--resume", str(path), "--episodes", "20", "--dry-run"]) == 0
    import json

    config = json.loads(capsys.readouterr().out)
    assert config["training"]["seed"] == 9
    assert config["environment"]["track"] == "square"
    assert config["agent"]["hidden_size"] == 8
    assert main(["train", "--resume", str(path), "--episodes", "5", "--dry-run"]) == 2


def test_missing_agent_state_is_a_clear_error(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save(
        {
            "format_version": 2,
            "observation_version": 2,
            "action_version": 2,
            "env_config": {},
            "train_config": {},
            "agent": {},
            "completed_episodes": 0,
            "best_score": [0.0, 0.0, 0.0],
            "episode_boundary": True,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
        },
        path,
    )
    with pytest.raises(ValueError, match="required agent state"):
        load_checkpoint(path)
