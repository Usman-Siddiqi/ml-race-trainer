"""Lightweight device selection. Auto uses CUDA only on confirmed AC power."""

import ctypes
import platform
from dataclasses import dataclass


def ac_power_connected():
    """Return True/False on Windows, or None when power status is unknown."""
    if platform.system() != "Windows":
        return None

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("flags", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("life", ctypes.c_ulong),
            ("full_life", ctypes.c_ulong),
        ]

    status = PowerStatus()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(status.ac)
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class DeviceChoice:
    name: str
    reason: str


def choose_device(requested="auto"):
    if requested == "cpu":
        return DeviceChoice("cpu", "CPU explicitly selected")
    if requested not in ("auto", "cuda"):
        raise ValueError("Choose auto, cpu, or cuda")
    if requested == "auto":
        power = ac_power_connected()
        if power is not True:
            reason = "running on battery" if power is False else "AC power status is unknown"
            return DeviceChoice("cpu", f"Auto selected CPU: {reason}")
    import torch

    if torch.cuda.is_available():
        return DeviceChoice(
            "cuda",
            "CUDA explicitly selected"
            if requested == "cuda"
            else "Auto selected CUDA: AC power and a CUDA device are available",
        )
    if requested == "cuda":
        raise RuntimeError(
            "CUDA is unavailable. Install the CUDA extra with 'uv sync --extra cuda', "
            "enable the NVIDIA GPU, or choose --device cpu."
        )
    return DeviceChoice("cpu", "Auto selected CPU: CUDA is unavailable")


def default_cpu_threads(device="cpu"):
    """PyTorch CPU threads when none are requested.

    On CUDA the CPU only drives the cars and feeds batches. A CPU learner uses the
    physical cores for its large batches; the simulator runs between updates.
    """
    import os

    return 2 if device == "cuda" else max(1, min(8, (os.cpu_count() or 2) // 2))


def diagnostics():
    """Read metadata only; this never allocates a GPU tensor or runs a benchmark."""
    import importlib.metadata
    import sys

    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ac_power": ac_power_connected(),
        "packages": {},
    }
    for package in ("numpy", "gymnasium", "pygame-ce", "torch"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = "not installed"
    try:
        import torch

        result["cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
        choice = choose_device("auto")
        result["auto_device"], result["auto_reason"] = choice.name, choice.reason
    except (ImportError, OSError) as exc:
        result["torch_error"] = str(exc)
    return result
