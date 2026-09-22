"""Atomic, versioned checkpoints loaded through PyTorch's weights-only unpickler."""

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ACTION_VERSION, OBSERVATION_VERSION, AgentConfig, EnvConfig, TrainConfig

FORMAT_VERSION = 2


def save_checkpoint(
    path, agent, env_config, train_config, completed_episodes, best_score, *, episode_boundary=True
):
    payload = {
        "format_version": FORMAT_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "env_config": asdict(env_config),
        "train_config": asdict(train_config),
        "agent": agent.state_dict(),
        "completed_episodes": completed_episodes,
        "best_score": best_score,
        "episode_boundary": episode_boundary,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(
            "This is a legacy weights-only model. Version 2 changes the sensors, "
            "actions, and physics, so those weights cannot be reused reliably. "
            "Keep the original checkpoint with the original project."
        )
    contract = (
        payload.get("format_version"),
        payload.get("observation_version"),
        payload.get("action_version"),
    )
    if contract != (FORMAT_VERSION, OBSERVATION_VERSION, ACTION_VERSION):
        raise ValueError(f"Unsupported checkpoint contract {contract}")
    for name in (
        "env_config",
        "train_config",
        "agent",
        "completed_episodes",
        "best_score",
        "episode_boundary",
        "torch_rng",
        "cuda_rng",
    ):
        if name not in payload:
            raise ValueError(f"Checkpoint is missing {name}")
    required_agent = {
        "config",
        "online",
        "target",
        "optimizer",
        "replay",
        "rng",
        "env_steps",
        "updates",
    }
    if not isinstance(payload["agent"], dict) or not required_agent.issubset(payload["agent"]):
        raise ValueError("Checkpoint is missing required agent state")
    if not isinstance(payload["completed_episodes"], int) or payload["completed_episodes"] < 0:
        raise ValueError("Checkpoint has an invalid completed episode count")
    EnvConfig(**payload["env_config"])
    TrainConfig(**payload["train_config"])
    AgentConfig(**payload["agent"]["config"])
    return payload


def load_policy(path, device="cpu"):
    from .agent import DQN

    payload = load_checkpoint(path)
    network = DQN(payload["agent"]["config"]["hidden_size"]).to(device)
    network.load_state_dict(payload["agent"]["online"])
    network.eval()

    class Policy:
        @torch.inference_mode()
        def act(self, observation, *, explore=False):
            if explore:
                raise ValueError("A saved policy is for inference; use resume for training")
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            return int(network(tensor).argmax(dim=1).item())

    return Policy(), payload
