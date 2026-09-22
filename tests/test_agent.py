import numpy as np
import pytest
import torch

from race_trainer.agent import ACTIONS, DQNAgent, double_dqn_targets
from race_trainer.config import OBSERVATION_SIZE, AgentConfig


def small_config(**overrides):
    return AgentConfig(
        **{
            "hidden_size": 8,
            "batch_size": 2,
            "replay_capacity": 4,
            "learning_starts": 2,
            "exploration_steps": 100,
            **overrides,
        }
    )


def test_double_dqn_selects_with_online_and_bootstraps_through_time_limits():
    online = torch.tensor([[1.0, 5.0], [1.0, 5.0], [6.0, 2.0]])
    target = torch.tensor([[8.0, 3.0], [8.0, 3.0], [4.0, 9.0]])
    # Second row is terminal; first is truncated but must still bootstrap.
    rewards = torch.tensor([1.0, 1.0, 1.0])
    terminated = torch.tensor([False, True, False])
    values = double_dqn_targets(online, target, rewards, terminated, 0.5)
    torch.testing.assert_close(values, torch.tensor([2.5, 1.0, 3.0]))


def test_exploration_uses_environment_steps_not_action_calls_or_updates():
    agent = DQNAgent(small_config(), seed=7)
    state = np.zeros(OBSERVATION_SIZE, np.float32)
    for _ in range(4):
        agent.act(state, explore=True)
    assert agent.epsilon == 1.0 and agent.env_steps == 0
    agent.observe(state, 1, 1.0, state, False)
    assert agent.env_steps == 1 and agent.epsilon == 1.0
    agent.env_steps = 52
    assert agent.epsilon == pytest.approx(0.525)
    agent.env_steps = 10000
    assert agent.epsilon == pytest.approx(0.05)


def test_inference_is_repeatable_without_gradients_or_weight_changes():
    first = DQNAgent(small_config(), seed=5)
    second = DQNAgent(small_config(), seed=5)
    state = np.linspace(0, 1, OBSERVATION_SIZE, dtype=np.float32)
    before = {k: v.clone() for k, v in first.online.state_dict().items()}
    grad_flags = []
    handle = first.online.register_forward_hook(
        lambda *args: grad_flags.append(torch.is_grad_enabled())
    )
    assert first.act(state) == second.act(state)
    assert grad_flags == [False]
    handle.remove()
    for key, value in first.online.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert first.updates == 0
    assert not any(p.requires_grad for p in first.target.parameters())
    assert set(ACTIONS[:, 0]) == {-1, 0, 1}


def test_replay_restores_wraparound_and_sampling_rng():
    first, second = DQNAgent(small_config(), seed=5), DQNAgent(small_config(), seed=8)
    for i in range(7):
        state = np.full(OBSERVATION_SIZE, i, np.float32)
        first.observe(state, i % 9, float(i), state + 1, i % 2 == 0)
    second.load_state_dict(first.state_dict())
    assert second.replay.size == 4 and second.replay.position == 3
    left = first.replay.sample(2, first.rng, first.device)
    right = second.replay.sample(2, second.rng, second.device)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b)
