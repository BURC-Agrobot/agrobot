"""Choose the device that models run on."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def select_device() -> torch.device:
    """Select the best available inference device.

    Check MPS first so runs on a Mac host use the GPU.

    Set AGROBOT_FORCE_CPU=1 to bypass GPU selection entirely.
    NucBox (gfx1151/Strix Halo) requires this setting until Sprint 3 supplies
    the MIGraphX path. The prebuilt ROCm 6.4 PyTorch wheels target discrete
    RDNA 3 (gfx1100). They fault on gfx1151 unified memory.
    """
    # Import here so scripts can parse arguments without the cost of loading torch
    import torch

    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
