import numpy as np
import pytest
import torch

from race_trainer.agent import (
    ACTIONS,
    EXPLORATION_WEIGHTS,
    DQNAgent,
    ReplayBuffer,
    double_dqn_targets,
)
from race_trainer.config import OBSERVATION_SIZE, AgentConfig, EnvConfig


def small_config(**overrides):
    return AgentConfig(
        **{
            "hidden_size": 8,
            "batch_size": 2,
            "replay_capacity": 8,
            "learning_starts": 2,
            "exploration_steps": 100,
            **overrides,
        }
    )


def state(value, count=1):
    return np.full((count, OBSERVATION_SIZE), value, np.float32)


def test_double_dqn_selects_with_online_and_bootstraps_through_time_limits():
    online = torch.tensor([[1.0, 5.0], [1.0, 5.0], [6.0, 2.0]])
    target = torch.tensor([[8.0, 3.0], [8.0, 3.0], [4.0, 9.0]])
    # Second row is terminal; first is truncated but must still bootstrap.
    returns = torch.tensor([1.0, 1.0, 1.0])
    terminated = torch.tensor([False, True, False])
    discount = torch.tensor([0.5, 0.5, 0.25])
    values = double_dqn_targets(online, target, returns, terminated, discount)
    torch.testing.assert_close(values, torch.tensor([2.5, 1.0, 2.0]))


def test_action_table_and_exploration_bias_keep_the_car_moving():
    assert ACTIONS.shape == (15, 2)
    assert set(ACTIONS[:, 0]) == {-1, 0, 1} and set(ACTIONS[:, 1]) == {-1, -0.5, 0, 0.5, 1}
    assert tuple(ACTIONS[2]) == (1, 0) and tuple(ACTIONS[7]) == (0, 0)
    assert EXPLORATION_WEIGHTS.sum() == pytest.approx(1)
    cfg = EnvConfig()
    throttle = ACTIONS[:, 0]
    force = np.where(throttle >= 0, throttle * cfg.acceleration, throttle * cfg.braking)
    # Random driving must accelerate on average; a uniform choice would not.
    assert EXPLORATION_WEIGHTS @ force - cfg.drag > 0
    assert force.mean() - cfg.drag < 0
    assert EXPLORATION_WEIGHTS @ ACTIONS[:, 1] == pytest.approx(0)


def test_exploration_clock_floors_and_held_random_actions():
    agent = DQNAgent(small_config(exploration_hold=3), seed=7, num_envs=4)
    observations = state(0.5, 4)
    assert agent.epsilon == 1.0 and agent.env_steps == 0
    actions = np.array([agent.act_batch(observations, explore=True) for _ in range(300)])
    assert agent.env_steps == 0 and agent.epsilon == 1.0
    # Fully random and sticky: actions repeat far more often than independent draws would.
    repeats = np.mean(actions[1:] == actions[:-1])
    assert len(np.unique(actions)) > 8 and 0.5 < repeats < 0.9
    assert np.mean(ACTIONS[actions][:, :, 0] > 0) > 0.5
    agent.observe(observations, actions[0], np.zeros(4), observations, [False] * 4, [False] * 4)
    assert agent.env_steps == 4
    agent.env_steps = 52
    np.testing.assert_allclose(agent.epsilons, 0.5)
    agent.env_steps = 10_000
    floors = agent.epsilons
    assert floors[0] == pytest.approx(agent.config.epsilon_end)
    assert floors[-1] == pytest.approx(agent.config.epsilon_end * agent.config.epsilon_spread)
    assert (np.diff(floors) > 0).all()
    with pytest.raises(ValueError, match="act_batch"):
        agent.act(observations[0], explore=True)


def test_inference_is_repeatable_without_gradients_or_weight_changes():
    first = DQNAgent(small_config(), seed=5)
    second = DQNAgent(small_config(), seed=5)
    observation = np.linspace(0, 1, OBSERVATION_SIZE, dtype=np.float32)
    before = {k: v.clone() for k, v in first.online.state_dict().items()}
    grad_flags = []
    handle = first.actor.register_forward_hook(
        lambda *args: grad_flags.append(torch.is_grad_enabled())
    )
    assert first.act(observation) == second.act(observation)
    batch = np.stack([observation, observation[::-1].copy()])
    np.testing.assert_array_equal(first.act_batch(batch), second.act_batch(batch))
    assert grad_flags == [False, False]
    handle.remove()
    for key, value in first.online.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert first.updates == 0
    # The cars are driven by a separate copy that starts equal to the learner.
    for key, value in first.online.state_dict().items():
        torch.testing.assert_close(first.actor.state_dict()[key], value)
        assert first.actor.state_dict()[key] is not value
    assert not any(p.requires_grad for p in first.actor.parameters())
    assert not any(p.requires_grad for p in first.target.parameters())


class Sequence:
    """A generator stand-in that returns chosen sample positions."""

    def __init__(self, ages, envs):
        self.values = [np.array(ages), np.array(envs)]

    def integers(self, low, high, size):
        return self.values.pop(0)


def fill(replay, rewards, terminated=(), truncated=()):
    for step, reward in enumerate(rewards):
        replay.push(
            state(step, 2),
            np.array([step, step]),
            np.array([reward, 10 * reward], np.float32),
            state(step + 0.5, 2),
            np.array([step in terminated, False]),
            np.array([step in terminated or step in truncated, False]),
        )


def test_n_step_returns_stop_at_episode_ends_and_the_newest_step():
    replay = ReplayBuffer(16, 2)
    fill(replay, [1, 2, 4, 8, 16, 32], terminated={2}, truncated={4})
    gamma = 0.5
    states, actions, returns, bootstrap, terminal, discount = replay.sample(
        6, Sequence([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]), 3, gamma
    )
    np.testing.assert_array_equal(actions, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(states[:, 0], [0, 1, 2, 3, 4, 5])
    # Step 0 spans three steps into the crash; step 3 stops at the time limit at step 4;
    # step 5 is the newest and uses a single step.
    np.testing.assert_allclose(returns, [1 + 1 + 1, 2 + 2, 4, 8 + 8, 16, 32])
    np.testing.assert_allclose(discount, [0.125, 0.25, 0.5, 0.25, 0.5, 0.5])
    np.testing.assert_array_equal(bootstrap[:, 0], [2.5, 2.5, 2.5, 4.5, 4.5, 5.5])
    # Only the crash is terminal; the time limit still bootstraps from its successor.
    np.testing.assert_array_equal(terminal, [True, True, True, False, False, False])
    # The second environment never ends: full three-step returns until data runs out.
    _, _, returns, bootstrap, terminal, discount = replay.sample(
        3, Sequence([0, 3, 4], [1, 1, 1]), 3, gamma
    )
    np.testing.assert_allclose(returns, [10 + 10 + 10, 80 + 80 + 80, 160 + 160])
    np.testing.assert_allclose(discount, [0.125, 0.125, 0.25])
    np.testing.assert_array_equal(bootstrap[:, 0], [2.5, 5.5, 5.5])
    assert not terminal.any()


def test_n_step_returns_follow_the_ring_and_respect_resume_breaks():
    replay = ReplayBuffer(8, 2)
    fill(replay, [1, 2, 4, 8, 16, 32])
    assert replay.horizon == 4 and replay.size == 4 and replay.position == 2
    # Oldest stored step is 2 (reward 4); it chains across the wrap to steps 3 and 4.
    _, actions, returns, bootstrap, _, _ = replay.sample(2, Sequence([0, 3], [0, 0]), 3, 1.0)
    np.testing.assert_array_equal(actions, [2, 5])
    np.testing.assert_allclose(returns, [4 + 8 + 16, 32])
    np.testing.assert_array_equal(bootstrap[:, 0], [4.5, 5.5])
    replay.mark_break()
    fill(replay, [64])
    # The step after a resume starts a new episode: nothing chains across the break.
    _, actions, returns, bootstrap, terminal, _ = replay.sample(2, Sequence([1, 2], [0, 0]), 3, 1.0)
    np.testing.assert_array_equal(actions, [4, 5])
    np.testing.assert_allclose(returns, [16 + 32, 32])
    np.testing.assert_array_equal(bootstrap[:, 0], [5.5, 5.5])
    assert not terminal.any()


def test_replay_restores_wraparound_and_sampling_rng():
    first = DQNAgent(small_config(), seed=5, num_envs=2)
    second = DQNAgent(small_config(), seed=8, num_envs=2)
    for i in range(7):
        ended = np.array([i % 2 == 0, False])
        first.observe(state(i, 2), [i % 15, 1], [float(i), 0.5], state(i + 1, 2), ended, ended)
    second.load_state_dict(first.state_dict())
    assert second.replay.size == 4 and second.replay.position == 3 and second.env_steps == 14
    left = first.replay.sample(4, first.rng, 3, 0.9)
    right = second.replay.sample(4, second.rng, 3, 0.9)
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="does not match"):
        DQNAgent(small_config(), num_envs=4).load_state_dict(first.state_dict())


def test_update_schedule_counts_environment_steps():
    agent = DQNAgent(small_config(learning_starts=8, train_every=4), num_envs=2)
    for step in range(10):
        assert agent.updates_due() == max(0, (2 * step - 8) // 4)
        agent.observe(state(0, 2), [0, 0], [0, 0], state(0, 2), [False] * 2, [False] * 2)


def test_learning_rate_anneals_linearly_with_training_progress():
    config = small_config(learning_rate=0.001, final_learning_rate=0.0001)
    agent = DQNAgent(config)
    rates = [agent.set_progress(fraction) for fraction in (-1, 0, 0.5, 1, 7)]
    assert rates == pytest.approx([0.001, 0.001, 0.00055, 0.0001, 0.0001])
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="final_learning_rate"):
        small_config(learning_rate=0.001, final_learning_rate=0.002)
