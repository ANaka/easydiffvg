"""Tests for rendering functionality."""

import pytest
import torch

import pydiffvg
from pydiffvg import Circle, Rect, ShapeGroup, SolidColor, LinearGradient, render


class TestRenderBasic:
    def test_render_empty_scene(self, device):
        """Rendering empty scene returns transparent black image."""
        image = render(64, 64, [], [])

        assert image.shape == (64, 64, 4)
        # Should be all zeros (transparent black)
        assert torch.allclose(image, torch.zeros_like(image))

    def test_render_single_circle(self, device):
        """Render a single filled circle."""
        circle = Circle(
            center=torch.tensor([32.0, 32.0], device=device),
            radius=torch.tensor(20.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        image = render(64, 64, [circle], [group], num_samples_x=1, num_samples_y=1)

        assert image.shape == (64, 64, 4)
        # Center pixel should be red
        center_pixel = image[32, 32]
        assert center_pixel[0] > 0.5  # Red channel
        assert center_pixel[3] > 0.5  # Alpha channel

    def test_render_single_rect(self, device):
        """Render a single filled rectangle."""
        rect = Rect(
            p_min=torch.tensor([10.0, 10.0], device=device),
            p_max=torch.tensor([50.0, 40.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
            ),
        )

        image = render(64, 64, [rect], [group], num_samples_x=1, num_samples_y=1)

        assert image.shape == (64, 64, 4)
        # Center of rectangle should be green
        center_pixel = image[25, 30]  # Roughly center of rect
        assert center_pixel[1] > 0.5  # Green channel
        assert center_pixel[3] > 0.5  # Alpha channel

    def test_render_image_dimensions(self, device):
        """Render produces correct dimensions."""
        circle = Circle(
            center=torch.tensor([50.0, 50.0], device=device),
            radius=torch.tensor(10.0, device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
            ),
        )

        # Test various dimensions
        for width, height in [(32, 32), (64, 64), (128, 96)]:
            image = render(width, height, [circle], [group], num_samples_x=1, num_samples_y=1)
            assert image.shape == (height, width, 4)


class TestRenderGradients:
    def test_render_linear_gradient(self, device):
        """Render shape with linear gradient fill."""
        rect = Rect(
            p_min=torch.tensor([0.0, 0.0], device=device),
            p_max=torch.tensor([64.0, 64.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=LinearGradient(
                begin=torch.tensor([0.0, 0.0], device=device),
                end=torch.tensor([64.0, 0.0], device=device),
                offsets=torch.tensor([0.0, 1.0], device=device),
                stop_colors=torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 1.0],  # Red at left
                        [0.0, 0.0, 1.0, 1.0],  # Blue at right
                    ],
                    device=device,
                ),
            ),
        )

        image = render(64, 64, [rect], [group], num_samples_x=1, num_samples_y=1)

        # Left side should be redder
        left_pixel = image[32, 5]
        # Right side should be bluer
        right_pixel = image[32, 58]

        assert left_pixel[0] > left_pixel[2]  # More red than blue
        assert right_pixel[2] > right_pixel[0]  # More blue than red


class TestRenderMultipleShapes:
    def test_render_overlapping_shapes(self, device):
        """Render two overlapping shapes with correct z-order."""
        # Back circle (blue)
        circle1 = Circle(
            center=torch.tensor([30.0, 32.0], device=device),
            radius=torch.tensor(20.0, device=device),
        )
        # Front circle (red)
        circle2 = Circle(
            center=torch.tensor([34.0, 32.0], device=device),
            radius=torch.tensor(20.0, device=device),
        )

        # First group is behind, second is in front
        group1 = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
            ),
        )
        group2 = ShapeGroup(
            shape_ids=torch.tensor([1], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        image = render(
            64, 64, [circle1, circle2], [group1, group2], num_samples_x=1, num_samples_y=1
        )

        # Center should be red (front circle)
        center = image[32, 32]
        assert center[0] > 0.5  # Red
        assert center[2] < 0.5  # Not blue


class TestApiCompatibility:
    def test_render_function_exists(self):
        """Verify render function is exported."""
        assert hasattr(pydiffvg, "render")
        assert callable(pydiffvg.render)

    def test_render_function_class_exists(self):
        """Verify RenderFunction is exported."""
        assert hasattr(pydiffvg, "RenderFunction")

    def test_shape_classes_exist(self):
        """Verify all shape classes are exported."""
        for name in ["Circle", "Ellipse", "Path", "Polygon", "Rect"]:
            assert hasattr(pydiffvg, name)

    def test_color_classes_exist(self):
        """Verify all color classes are exported."""
        for name in ["SolidColor", "LinearGradient", "RadialGradient"]:
            assert hasattr(pydiffvg, name)
