"""Tests for gradient computation.

NOTE: The current implementation provides the forward pass for rendering.
Full backward pass with boundary sampling is not yet implemented.
These tests verify the current state of gradient support.
"""

import pytest
import torch

import easydiffvg
from easydiffvg import Circle, ShapeGroup, SolidColor, render


class TestForwardPassWorks:
    def test_render_produces_tensor(self, device):
        """Verify render produces valid output tensor."""
        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(10.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        image = render(32, 32, [circle], [group], num_samples_x=1, num_samples_y=1)

        assert isinstance(image, torch.Tensor)
        assert image.shape == (32, 32, 4)
        assert image.dtype == torch.float32

    def test_render_with_requires_grad_shape(self, device):
        """Render works when shape params have requires_grad."""
        center = torch.tensor([16.0, 16.0], device=device, requires_grad=True)
        radius = torch.tensor(10.0, device=device, requires_grad=True)

        circle = Circle(center=center, radius=radius)
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        # Forward pass should work
        image = render(32, 32, [circle], [group], num_samples_x=1, num_samples_y=1)

        assert image.shape == (32, 32, 4)
        # Center pixel should have color (circle is at center)
        assert image[16, 16, 0] > 0.5  # Red channel

    def test_render_with_requires_grad_color(self, device):
        """Render works when color params have requires_grad."""
        color = torch.tensor([0.5, 0.5, 0.5, 1.0], device=device, requires_grad=True)

        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(8.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=color),
        )

        # Forward pass should work
        image = render(32, 32, [circle], [group], num_samples_x=1, num_samples_y=1)

        assert image.shape == (32, 32, 4)


class TestGradientStatusDocumented:
    """Tests documenting current gradient support status."""

    def test_backward_not_implemented_note(self):
        """Document that full backward pass is not yet implemented.

        The RenderFunction.backward currently returns None for all gradients.
        This is expected behavior for the current implementation phase.

        Full backward pass with boundary sampling (as in original diffvg)
        would require:
        1. Finding pixels near shape boundaries
        2. Sampling along boundaries
        3. Computing how boundary movement affects coverage
        4. Chain rule back to shape parameters

        For now, the library provides forward-only rendering.
        """
        pass  # This is documentation test
