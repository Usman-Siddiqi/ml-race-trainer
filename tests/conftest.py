"""All verification is CPU only. No test may train a neural network."""

import os

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    import torch

    torch.set_num_threads(1)

    def blocked(*args, **kwargs):
        raise AssertionError("Training is prohibited during this verification run")

    monkeypatch.setattr(torch.Tensor, "backward", blocked)
    monkeypatch.setattr(torch.autograd, "backward", blocked)
    monkeypatch.setattr(torch.optim.Adam, "step", blocked)
