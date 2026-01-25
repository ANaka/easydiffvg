"""Tests for color classes."""

import pytest
import torch

from pydiffvg import SolidColor, LinearGradient, RadialGradient


class TestSolidColor:
    def test_solid_color_creation(self, device):
        """SolidColor stores RGBA as [4] tensor."""
        rgba = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)

        color = SolidColor(color=rgba)

        assert color.color.shape == (4,)
        torch.testing.assert_close(color.color, rgba)

    def test_solid_color_semitransparent(self, device):
        """SolidColor supports alpha < 1."""
        rgba = torch.tensor([0.0, 1.0, 0.0, 0.5], device=device)

        color = SolidColor(color=rgba)

        assert color.color[3] == 0.5


class TestLinearGradient:
    def test_linear_gradient_creation(self, device):
        """LinearGradient has begin, end, offsets, stop_colors."""
        begin = torch.tensor([0.0, 0.0], device=device)
        end = torch.tensor([64.0, 64.0], device=device)
        offsets = torch.tensor([0.0, 1.0], device=device)
        stop_colors = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],  # red
                [0.0, 0.0, 1.0, 1.0],  # blue
            ],
            device=device,
        )

        grad = LinearGradient(
            begin=begin,
            end=end,
            offsets=offsets,
            stop_colors=stop_colors,
        )

        assert grad.begin.shape == (2,)
        assert grad.end.shape == (2,)
        assert grad.offsets.shape == (2,)
        assert grad.stop_colors.shape == (2, 4)

    def test_linear_gradient_multi_stop(self, device):
        """LinearGradient supports multiple color stops."""
        grad = LinearGradient(
            begin=torch.tensor([0.0, 0.0]),
            end=torch.tensor([100.0, 0.0]),
            offsets=torch.tensor([0.0, 0.5, 1.0]),
            stop_colors=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ),
        )

        assert grad.offsets.shape == (3,)
        assert grad.stop_colors.shape == (3, 4)


class TestRadialGradient:
    def test_radial_gradient_creation(self, device):
        """RadialGradient has center, radius, offsets, stop_colors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor([20.0, 20.0], device=device)  # rx, ry
        offsets = torch.tensor([0.0, 1.0], device=device)
        stop_colors = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # white center
                [0.0, 0.0, 0.0, 1.0],  # black edge
            ],
            device=device,
        )

        grad = RadialGradient(
            center=center,
            radius=radius,
            offsets=offsets,
            stop_colors=stop_colors,
        )

        assert grad.center.shape == (2,)
        assert grad.radius.shape == (2,)
        assert grad.offsets.shape == (2,)
        assert grad.stop_colors.shape == (2, 4)

    def test_radial_gradient_elliptical(self, device):
        """RadialGradient supports elliptical shape (rx != ry)."""
        grad = RadialGradient(
            center=torch.tensor([50.0, 50.0]),
            radius=torch.tensor([30.0, 15.0]),  # wide ellipse
            offsets=torch.tensor([0.0, 1.0]),
            stop_colors=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ),
        )

        assert grad.radius[0] != grad.radius[1]
