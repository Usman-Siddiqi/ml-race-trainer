"""Double DQN with an environment-step exploration clock and bounded CPU replay."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .config import OBSERVATION_SIZE, AgentConfig

# Each throttle state offers left/straight/right, including brake and coast.
ACTIONS = np.array(
    [(throttle, steer) for throttle in (1, 0, -1) for steer in (-1, 0, 1)], dtype=np.float32
)
ACTIONS.setflags(write=False)


class DQN(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )

    def forward(self, inputs):
        return self.net(inputs)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = self.position = 0
        self.states = np.zeros((capacity, OBSERVATION_SIZE), np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.bool_)

    def push(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device):
        indices = rng.choice(self.size, batch_size, replace=False)
        return tuple(
            torch.as_tensor(data[indices], device=device)
            for data in (self.states, self.actions, self.rewards, self.next_states, self.terminated)
        )

    def state_dict(self):
        result = {"capacity": self.capacity, "size": self.size, "position": self.position}
        for name in ("states", "next_states", "actions", "rewards", "terminated"):
            result[name] = torch.from_numpy(getattr(self, name)[: self.size].copy())
        return result

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or not 0 <= state["size"] <= self.capacity:
            raise ValueError("Checkpoint replay capacity does not match")
        size, position = state["size"], state["position"]
        if not 0 <= position < self.capacity or (size < self.capacity and position != size):
            raise ValueError("Invalid replay cursor")
        for name in ("states", "next_states", "actions", "rewards", "terminated"):
            source = state[name].cpu().numpy()
            target = getattr(self, name)[:size]
            if source.shape != target.shape:
                raise ValueError(f"Invalid replay shape for {name}")
            target[:] = source
        self.size, self.position = size, position


def double_dqn_targets(online_next, target_next, rewards, terminated, gamma):
    """Only a true terminal state suppresses bootstrap; time limits do not."""
    actions = online_next.argmax(dim=1, keepdim=True)
    values = target_next.gather(1, actions).squeeze(1)
    return rewards + gamma * values * (~terminated)


class DQNAgent:
    def __init__(self, config=None, device="cpu", seed=42):
        self.config = config or AgentConfig()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        # Initialization is reproducible without disturbing the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = DQN(self.config.hidden_size).to(self.device)
            self.target = DQN(self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.env_steps = self.updates = 0

    @property
    def epsilon(self):
        elapsed = max(0, self.env_steps - self.config.learning_starts)
        fraction = min(1.0, elapsed / self.config.exploration_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @torch.inference_mode()
    def act(self, state, *, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        observation = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(observation).argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated):
        self.replay.push(state, action, reward, next_state, terminated)
        self.env_steps += 1

    def learn_if_due(self):
        cfg = self.config
        if (
            self.env_steps < cfg.learning_starts
            or self.env_steps % cfg.train_every
            or self.replay.size < cfg.batch_size
        ):
            return None
        states, actions, rewards, next_states, terminated = self.replay.sample(
            cfg.batch_size, self.rng, self.device
        )
        with torch.no_grad():
            targets = double_dqn_targets(
                self.online(next_states), self.target(next_states), rewards, terminated, cfg.gamma
            )
        predictions = self.online(states).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite training loss; stopped before applying an optimizer update"
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def state_dict(self):
        return {
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "rng": self.rng.bit_generator.state,
            "env_steps": self.env_steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if AgentConfig(**state["config"]) != self.config:
            raise ValueError("Checkpoint agent configuration does not match")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        # PyTorch moves moments to the parameter device but can retain shared CPU
        # step tensors. Each restore must own its optimizer state independently.
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.replay.load_state_dict(state["replay"])
        self.rng.bit_generator.state = state["rng"]
        self.env_steps, self.updates = state["env_steps"], state["updates"]
