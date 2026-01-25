"""Shared pytest fixtures for pydiffvg tests."""

import pytest
import torch
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def device() -> torch.device:
    """Default test device."""
    return torch.device("cpu")


@pytest.fixture
def canvas_64() -> tuple[int, int]:
    """64x64 canvas for fast tests."""
    return (64, 64)


@pytest.fixture
def canvas_256() -> tuple[int, int]:
    """256x256 canvas for realistic tests."""
    return (256, 256)
