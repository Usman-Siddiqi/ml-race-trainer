"""CPU checks by default; bounded learning checks require explicit GPU opt-in."""

import os

import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        help="Enable bounded CUDA learning tests; requires a working NVIDIA GPU",
    )


def pytest_configure(config):
    if not config.getoption("--run-gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-gpu"):
        skip = pytest.mark.skip(reason="CUDA learning tests require --run-gpu")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def verification_limits(monkeypatch, request):
    import torch

    torch.set_num_threads(1)
    if request.node.get_closest_marker("gpu"):
        if not request.config.getoption("--run-gpu"):
            pytest.fail("GPU tests require --run-gpu")
        if not torch.cuda.is_available():
            pytest.fail("--run-gpu was requested, but CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()
        yield
        torch.cuda.synchronize()
        request.node.user_properties.append(
            ("peak_cuda_allocated_mib", torch.cuda.max_memory_allocated() / 1024**2)
        )
        torch.cuda.empty_cache()
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Training is prohibited during this verification run")

    monkeypatch.setattr(torch.Tensor, "backward", blocked)
    monkeypatch.setattr(torch.autograd, "backward", blocked)
    monkeypatch.setattr(torch.optim.Adam, "step", blocked)
    yield
