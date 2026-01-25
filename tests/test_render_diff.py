"""Tests for differentiable rendering using soft rasterization."""

import pytest
import torch

import easydiffvg
from easydiffvg import (
    Circle,
    Ellipse,
    Rect,
    Polygon,
    ShapeGroup,
    SolidColor,
    LinearGradient,
    render_differentiable,
)


class TestDifferentiableRenderBasic:
    """Basic tests for the differentiable renderer."""

    def test_render_empty_scene(self, device):
        """Render with no shapes returns zeros."""
        image = render_differentiable(32, 32, [], [])
        assert image.shape == (32, 32, 4)
        assert image.sum() == 0.0

    def test_render_circle(self, device):
        """Render a simple circle."""
        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(8.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group])

        assert image.shape == (32, 32, 4)
        # Center should have some red color
        assert image[16, 16, 0] > 0.5  # Red channel
        assert image[16, 16, 3] > 0.5  # Alpha channel

    def test_render_rect(self, device):
        """Render a simple rectangle."""
        rect = Rect(
            p_min=torch.tensor([8.0, 8.0], device=device),
            p_max=torch.tensor([24.0, 24.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [rect], [group])

        assert image.shape == (32, 32, 4)
        # Center should have green color
        assert image[16, 16, 1] > 0.5  # Green channel

    def test_render_ellipse(self, device):
        """Render an ellipse."""
        ellipse = Ellipse(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor([12.0, 6.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [ellipse], [group])

        assert image.shape == (32, 32, 4)
        # Center should have blue color
        assert image[16, 16, 2] > 0.5  # Blue channel

    def test_render_polygon(self, device):
        """Render a triangle (closed polygon)."""
        polygon = Polygon(
            points=torch.tensor([
                [16.0, 4.0],
                [28.0, 28.0],
                [4.0, 28.0],
            ], device=device),
            is_closed=True,
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 1.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [polygon], [group])

        assert image.shape == (32, 32, 4)
        # Center of triangle should have color
        assert image[16, 16, 0] > 0.3  # Some yellow present


class TestDifferentiableGradients:
    """Tests verifying gradients flow through the differentiable renderer."""

    def test_gradient_flows_to_circle_center(self, device):
        """Gradients flow back to circle center."""
        center = torch.tensor([16.0, 16.0], device=device, requires_grad=True)
        radius = torch.tensor(8.0, device=device)

        circle = Circle(center=center, radius=radius)
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group], softness=1.0)

        # Compute loss and backward
        loss = image.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert center.grad is not None
        assert center.grad.abs().sum() > 0

    def test_gradient_flows_to_circle_radius(self, device):
        """Gradients flow back to circle radius."""
        center = torch.tensor([16.0, 16.0], device=device)
        radius = torch.tensor(8.0, device=device, requires_grad=True)

        circle = Circle(center=center, radius=radius)
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group], softness=1.0)

        loss = image.sum()
        loss.backward()

        assert radius.grad is not None
        # Larger radius = more pixels covered = larger sum
        # So gradient should be positive
        assert radius.grad > 0

    def test_gradient_flows_to_color(self, device):
        """Gradients flow back to fill color."""
        color = torch.tensor([0.5, 0.5, 0.5, 1.0], device=device, requires_grad=True)

        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(8.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=color),
        )

        image = render_differentiable(32, 32, [circle], [group], softness=1.0)

        # Loss on red channel only
        loss = image[..., 0].sum()
        loss.backward()

        assert color.grad is not None
        # Red channel gradient should be positive (more red = larger sum)
        assert color.grad[0] > 0

    def test_gradient_flows_to_rect_corners(self, device):
        """Gradients flow back to rectangle corners."""
        p_min = torch.tensor([8.0, 8.0], device=device, requires_grad=True)
        p_max = torch.tensor([24.0, 24.0], device=device, requires_grad=True)

        rect = Rect(p_min=p_min, p_max=p_max)
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [rect], [group], softness=1.0)

        loss = image.sum()
        loss.backward()

        assert p_min.grad is not None
        assert p_max.grad is not None
        # Moving p_min right/down decreases area, so gradient should be positive
        # Moving p_max right/down increases area, so gradient should be positive
        assert p_max.grad.sum() > 0

    def test_optimization_moves_circle(self, device):
        """Circle can be moved via gradient descent."""
        # Start circle at offset position
        center = torch.tensor([10.0, 10.0], device=device, requires_grad=True)
        target_center = torch.tensor([20.0, 20.0], device=device)

        circle = Circle(
            center=center,
            radius=torch.tensor(5.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )

        # Create target image
        target_circle = Circle(
            center=target_center,
            radius=torch.tensor(5.0, device=device),
        )
        target_group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
        )
        target_image = render_differentiable(32, 32, [target_circle], [target_group], softness=1.0)

        initial_center = center.clone().detach()

        # A few optimization steps
        optimizer = torch.optim.SGD([center], lr=1.0)
        for _ in range(5):
            optimizer.zero_grad()
            image = render_differentiable(32, 32, [circle], [group], softness=1.0)
            loss = ((image - target_image) ** 2).sum()
            loss.backward()
            optimizer.step()

        # Center should have moved toward target
        initial_dist = (initial_center - target_center).norm()
        final_dist = (center - target_center).norm()
        assert final_dist < initial_dist


class TestDifferentiableGradientColors:
    """Tests for gradient colors in the differentiable renderer."""

    def test_linear_gradient(self, device):
        """Render with linear gradient fill."""
        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(12.0, device=device),
        )
        gradient = LinearGradient(
            begin=torch.tensor([0.0, 0.0], device=device),
            end=torch.tensor([32.0, 32.0], device=device),
            offsets=torch.tensor([0.0, 1.0], device=device),
            stop_colors=torch.tensor([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=gradient,
        )

        image = render_differentiable(32, 32, [circle], [group])

        assert image.shape == (32, 32, 4)
        # Should have color variation across the circle
        assert image[8, 8, 0] > image[24, 24, 0]  # More red at top-left


class TestDifferentiableStrokes:
    """Tests for stroke rendering in the differentiable renderer."""

    def test_render_circle_with_stroke(self, device):
        """Render a circle with stroke only (no fill)."""
        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(10.0, device=device),
            stroke_width=torch.tensor(2.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=None,
            stroke_color=SolidColor(color=torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group])

        assert image.shape == (32, 32, 4)
        # Edge of circle (at radius 10 from center 16,16) should have color
        # Point at (16, 6) is on the edge (distance 10 from center)
        assert image[6, 16, 1] > 0.3  # Green channel at top edge
        # Center should have no color (stroke only, no fill)
        assert image[16, 16, 3] < 0.5  # Low alpha at center

    def test_render_circle_with_fill_and_stroke(self, device):
        """Render a circle with both fill and stroke."""
        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(8.0, device=device),
            stroke_width=torch.tensor(3.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)),
            stroke_color=SolidColor(color=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group])

        assert image.shape == (32, 32, 4)
        # Center should have red fill
        assert image[16, 16, 0] > 0.5  # Red at center
        # Edge should have blue stroke (stroke renders on top)
        # At radius 8, the edge is around (16, 8)
        assert image[8, 16, 2] > 0.3  # Blue at edge

    def test_gradient_flows_to_stroke_width(self, device):
        """Gradients flow back to stroke width."""
        stroke_width = torch.tensor(2.0, device=device, requires_grad=True)

        circle = Circle(
            center=torch.tensor([16.0, 16.0], device=device),
            radius=torch.tensor(10.0, device=device),
            stroke_width=stroke_width,
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=None,
            stroke_color=SolidColor(color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)),
        )

        image = render_differentiable(32, 32, [circle], [group], softness=1.0)

        loss = image.sum()
        loss.backward()

        assert stroke_width.grad is not None
        # Larger stroke width = more pixels covered = larger sum
        # So gradient should be positive
        assert stroke_width.grad > 0


class TestApiExport:
    """Test that render_differentiable is properly exported."""

    def test_exported_from_package(self):
        """render_differentiable should be accessible from easydiffvg."""
        assert hasattr(easydiffvg, "render_differentiable")
        assert callable(easydiffvg.render_differentiable)
