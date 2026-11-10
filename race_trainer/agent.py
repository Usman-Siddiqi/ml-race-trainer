"""Dueling Double DQN with n-step returns, batched acting, and CPU replay."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from .config import OBSERVATION_SIZE, AgentConfig

THROTTLES = (1.0, 0.0, -1.0)
STEERINGS = (-1.0, -0.5, 0.0, 0.5, 1.0)
# Throttle-major table: accelerate, coast, brake; each with five steering angles.
ACTIONS = np.array([(t, s) for t in THROTTLES for s in STEERINGS], dtype=np.float32)
ACTIONS.setflags(write=False)
# Random actions favour throttle and gentle steering. A uniform choice brakes harder
# than it accelerates, so an exploring car would sit still and learn nothing.
EXPLORATION_WEIGHTS = np.outer((0.6, 0.25, 0.15), (0.15, 0.2, 0.3, 0.2, 0.15)).ravel()
EXPLORATION_WEIGHTS.setflags(write=False)


class DQN(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, len(ACTIONS))

    def forward(self, inputs):
        features = self.body(inputs)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """A time-ordered CPU ring of (horizon, num_envs) transitions.

    Keeping the steps of each environment in order lets sampling build n-step returns on
    the fly, stopping at episode ends and at the newest data.
    """

    FIELDS = ("states", "next_states", "actions", "rewards", "terminated", "ended")

    def __init__(self, capacity, num_envs):
        self.num_envs = num_envs
        self.horizon = capacity // num_envs
        if self.horizon < 1:
            raise ValueError("Replay capacity must hold at least one step of every environment")
        shape = (self.horizon, num_envs)
        self.states = np.zeros((*shape, OBSERVATION_SIZE), np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(shape, np.int64)
        self.rewards = np.zeros(shape, np.float32)
        self.terminated = np.zeros(shape, np.bool_)
        self.ended = np.zeros(shape, np.bool_)
        self.size = self.position = 0

    def __len__(self):
        return self.size * self.num_envs

    def push(self, states, actions, rewards, next_states, terminated, ended):
        """Store one step of every environment. ``ended`` includes time limits and stalls."""
        row = self.position
        self.states[row], self.next_states[row] = states, next_states
        self.actions[row], self.rewards[row] = actions, rewards
        self.terminated[row], self.ended[row] = terminated, ended
        self.position = (row + 1) % self.horizon
        self.size = min(self.size + 1, self.horizon)

    def mark_break(self):
        """Stop n-step returns at the newest step, e.g. before resumed episodes follow it."""
        if self.size:
            self.ended[self.position - 1] = True

    def sample(self, batch_size, rng, n_step, gamma):
        """Return states, actions, n-step returns, bootstrap states, terminal flags, discounts."""
        age = rng.integers(0, self.size, batch_size)
        env = rng.integers(0, self.num_envs, batch_size)
        row = (self.position - self.size + age) % self.horizon
        remaining = self.size - age
        returns = self.rewards[row, env]
        discount = np.full(batch_size, gamma, np.float32)
        last = row
        running = ~self.ended[row, env]
        for step in range(1, n_step):
            following = (row + step) % self.horizon
            running = running & (remaining > step)
            scale = np.float32(gamma**step)
            returns = returns + np.where(running, scale * self.rewards[following, env], 0)
            last = np.where(running, following, last)
            discount = np.where(running, np.float32(gamma ** (step + 1)), discount)
            running = running & ~self.ended[following, env]
        return (
            self.states[row, env],
            self.actions[row, env],
            returns.astype(np.float32),
            self.next_states[last, env],
            self.terminated[last, env],
            discount,
        )

    def state_dict(self):
        result = {
            "horizon": self.horizon,
            "num_envs": self.num_envs,
            "size": self.size,
            "position": self.position,
        }
        for name in self.FIELDS:
            result[name] = torch.from_numpy(getattr(self, name)[: self.size].copy())
        return result

    def load_state_dict(self, state):
        if (state["horizon"], state["num_envs"]) != (self.horizon, self.num_envs):
            raise ValueError("Checkpoint replay capacity or environment count does not match")
        size, position = state["size"], state["position"]
        if not 0 <= size <= self.horizon:
            raise ValueError("Invalid replay size")
        if not 0 <= position < self.horizon or (size < self.horizon and position != size):
            raise ValueError("Invalid replay cursor")
        for name in self.FIELDS:
            source = state[name].cpu().numpy()
            target = getattr(self, name)[:size]
            if source.shape != target.shape:
                raise ValueError(f"Invalid replay shape for {name}")
            target[:] = source
        self.size, self.position = size, position


def double_dqn_targets(online_next, target_next, returns, terminated, discount):
    """Only a true terminal state suppresses bootstrap; time limits do not."""
    actions = online_next.argmax(dim=1, keepdim=True)
    values = target_next.gather(1, actions).squeeze(1)
    return returns + discount * values * (~terminated)


class DQNAgent:
    def __init__(self, config=None, device="cpu", seed=42, num_envs=1):
        self.config = config or AgentConfig()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        # Initialization is reproducible without disturbing the caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = DQN(self.config.hidden_size).to(self.device)
            self.target = DQN(self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False).eval()
        on_gpu = self.device.type == "cuda"
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.config.learning_rate, fused=on_gpu
        )
        # Choosing actions for a few dozen cars is faster on the CPU than a GPU round
        # trip, so a CUDA learner keeps a CPU copy of the network for driving.
        self.actor = DQN(self.config.hidden_size).requires_grad_(False).eval() if on_gpu else None
        self._actor_updates = None
        self.replay = ReplayBuffer(self.config.replay_capacity, num_envs)
        self.env_steps = self.updates = 0
        # Each environment explores at its own final rate, from careful to adventurous.
        spread = np.linspace(0, 1, num_envs) if num_envs > 1 else np.zeros(1)
        self.epsilon_floors = self.config.epsilon_end * self.config.epsilon_spread**spread
        self._held_action = np.zeros(num_envs, dtype=np.int64)
        self._hold = np.zeros(num_envs, dtype=np.int64)

    @property
    def epsilons(self):
        elapsed = max(0, self.env_steps - self.config.learning_starts)
        fraction = min(1.0, elapsed / self.config.exploration_steps)
        return np.maximum(self.config.epsilon_start * (1 - fraction), self.epsilon_floors)

    @property
    def epsilon(self):
        return float(self.epsilons.mean())

    def sync_actor(self):
        """Copy the latest weights to the CPU driving network (a no-op on a CPU learner)."""
        if self.actor is not None and self._actor_updates != self.updates:
            self.actor.load_state_dict({k: v.cpu() for k, v in self.online.state_dict().items()})
            self._actor_updates = self.updates

    @torch.inference_mode()
    def greedy(self, states):
        network = self.online
        if self.actor is not None:
            stale = self._actor_updates is None or (
                self.updates - self._actor_updates >= self.config.actor_sync_every
            )
            if stale:
                self.sync_actor()
            network = self.actor
        return network(torch.as_tensor(states, dtype=torch.float32)).argmax(dim=1).numpy()

    def act(self, state, *, explore=False):
        """Greedy action for one observation; exploration needs the batched form."""
        if explore:
            raise ValueError("Use act_batch for exploration across the training environments")
        return int(self.greedy(np.asarray(state)[None])[0])

    def act_batch(self, states, *, explore=False):
        actions = self.greedy(states)
        if not explore:
            return actions
        # A random action is held for a few steps: single-step noise at 30 Hz averages
        # out and never moves the car off the greedy line.
        held = self._hold > 0
        start = ~held & (self.rng.random(self.num_envs) < self.epsilons)
        count = int(start.sum())
        if count:
            self._held_action[start] = self.rng.choice(len(ACTIONS), count, p=EXPLORATION_WEIGHTS)
            self._hold[start] = self.rng.geometric(1 / self.config.exploration_hold, count)
        exploring = held | start
        self._hold[exploring] -= 1
        return np.where(exploring, self._held_action, actions)

    def observe(self, states, actions, rewards, next_states, terminated, ended):
        """Record one step of every environment. ``ended`` = terminated or truncated."""
        self.replay.push(states, actions, rewards, next_states, terminated, ended)
        self.env_steps += self.num_envs

    def updates_due(self):
        cfg = self.config
        if self.env_steps < cfg.learning_starts or len(self.replay) < cfg.batch_size:
            return 0
        return max(0, (self.env_steps - cfg.learning_starts) // cfg.train_every - self.updates)

    def learn(self):
        cfg = self.config
        states, actions, returns, next_states, terminated, discount = self.replay.sample(
            cfg.batch_size, self.rng, cfg.n_step, cfg.gamma
        )
        # One packed host-to-device copy per update: on a GPU, many small transfers and
        # kernel launches cost more than the arithmetic of this small network.
        size = OBSERVATION_SIZE
        packed = np.empty((cfg.batch_size, 2 * size + 4), np.float32)
        packed[:, :size], packed[:, size : 2 * size] = states, next_states
        packed[:, -4], packed[:, -3] = returns, discount
        packed[:, -2], packed[:, -1] = terminated, actions
        batch = torch.from_numpy(packed).to(self.device)
        values = self.online(batch[:, : 2 * size].reshape(-1, size)).view(cfg.batch_size, 2, -1)
        with torch.no_grad():
            targets = double_dqn_targets(
                values[:, 1],
                self.target(batch[:, size : 2 * size]),
                batch[:, -4],
                batch[:, -2] > 0.5,
                batch[:, -3],
            )
        predictions = values[:, 0].gather(1, batch[:, -1:].long()).squeeze(1)
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
        return float(loss.detach())

    def state_dict(self):
        return {
            "config": asdict(self.config),
            "num_envs": self.num_envs,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "rng": self.rng.bit_generator.state,
            "env_steps": self.env_steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if AgentConfig(**state["config"]) != self.config or state["num_envs"] != self.num_envs:
            raise ValueError("Checkpoint agent configuration does not match")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        # PyTorch moves moments to the parameter device but can retain shared CPU
        # step tensors. Each restore must own its optimizer state independently.
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.replay.load_state_dict(state["replay"])
        self.rng.bit_generator.state = state["rng"]
        self.env_steps, self.updates = state["env_steps"], state["updates"]
        self._actor_updates = None
