"""Tests for utility modules (device, image)."""

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

import pydiffvg


class TestDeviceUtils:
    """Tests for device management utilities."""

    def test_get_device_returns_device(self):
        """get_device returns a torch.device."""
        device = pydiffvg.get_device()
        assert isinstance(device, torch.device)

    def test_get_use_gpu_returns_bool(self):
        """get_use_gpu returns a boolean."""
        use_gpu = pydiffvg.get_use_gpu()
        assert isinstance(use_gpu, bool)

    def test_set_use_gpu_to_false(self):
        """set_use_gpu(False) switches to CPU."""
        original = pydiffvg.get_use_gpu()
        try:
            pydiffvg.set_use_gpu(False)
            assert pydiffvg.get_use_gpu() is False
            assert pydiffvg.get_device().type == 'cpu'
        finally:
            pydiffvg.set_use_gpu(original)

    def test_set_device(self):
        """set_device changes the device."""
        original = pydiffvg.get_device()
        try:
            pydiffvg.set_device(torch.device('cpu'))
            assert pydiffvg.get_device().type == 'cpu'
        finally:
            pydiffvg.set_device(original)


class TestImageUtils:
    """Tests for image I/O utilities."""

    def test_imwrite_tensor(self, tmp_path):
        """imwrite saves a tensor to file."""
        img = torch.rand(32, 32, 4)
        output_file = tmp_path / "test.png"

        pydiffvg.imwrite(img, str(output_file))

        assert output_file.exists()

    def test_imwrite_numpy(self, tmp_path):
        """imwrite saves a numpy array to file."""
        img = np.random.rand(32, 32, 4).astype(np.float32)
        output_file = tmp_path / "test.png"

        pydiffvg.imwrite(img, str(output_file))

        assert output_file.exists()

    def test_imwrite_grayscale(self, tmp_path):
        """imwrite handles grayscale images."""
        img = torch.rand(32, 32)
        output_file = tmp_path / "test.png"

        pydiffvg.imwrite(img, str(output_file))

        assert output_file.exists()

    def test_imwrite_creates_directory(self, tmp_path):
        """imwrite creates output directory if needed."""
        img = torch.rand(32, 32, 4)
        output_file = tmp_path / "subdir" / "test.png"

        pydiffvg.imwrite(img, str(output_file))

        assert output_file.exists()

    def test_imwrite_with_normalize(self, tmp_path):
        """imwrite can normalize image values."""
        img = torch.rand(32, 32, 4) * 10 - 5  # Values in [-5, 5]
        output_file = tmp_path / "test.png"

        pydiffvg.imwrite(img, str(output_file), normalize=True)

        assert output_file.exists()

    def test_imwrite_rgb(self, tmp_path):
        """imwrite handles RGB images (3 channels)."""
        img = torch.rand(32, 32, 3)
        output_file = tmp_path / "test.png"

        pydiffvg.imwrite(img, str(output_file))

        assert output_file.exists()


class TestApiExports:
    """Test that utility functions are exported."""

    def test_device_functions_exported(self):
        """Device management functions are accessible."""
        assert hasattr(pydiffvg, "get_device")
        assert hasattr(pydiffvg, "get_use_gpu")
        assert hasattr(pydiffvg, "set_device")
        assert hasattr(pydiffvg, "set_use_gpu")

    def test_imwrite_exported(self):
        """imwrite is accessible from pydiffvg."""
        assert hasattr(pydiffvg, "imwrite")
        assert callable(pydiffvg.imwrite)
