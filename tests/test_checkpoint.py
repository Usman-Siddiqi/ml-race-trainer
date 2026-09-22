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
