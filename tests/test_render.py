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



class TestSdfOutput:
    def test_sdf_output_mode(self, device):
        """Render SDF instead of color image."""
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        # Render SDF
        args = pydiffvg.RenderFunction.serialize_scene(
            64, 64, [circle], [group],
            output_type=pydiffvg.OutputType.sdf
        )
        sdf = pydiffvg.RenderFunction.apply(64, 64, 2, 2, 0, None, *args)

        assert sdf.shape == (64, 64, 1)
        # Center should be inside (negative)
        assert sdf[32, 32, 0] < 0
        # Corner should be outside (positive)
        assert sdf[0, 0, 0] > 0

    def test_sdf_at_eval_positions(self, device):
        """Evaluate SDF at specific positions."""
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        # Evaluate at center and corner
        eval_positions = torch.tensor([
            [32.0, 32.0],  # center - should be inside
            [0.0, 0.0],    # corner - should be outside
        ], device=device)

        args = pydiffvg.RenderFunction.serialize_scene(
            64, 64, [circle], [group],
            output_type=pydiffvg.OutputType.sdf,
            eval_positions=eval_positions
        )
        sdf = pydiffvg.RenderFunction.apply(64, 64, 2, 2, 0, None, *args)

        assert sdf.shape == (2, 1)
        assert sdf[0, 0] < 0  # center inside
        assert sdf[1, 0] > 0  # corner outside

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

    def test_filter_type_enum_exists(self):
        """Verify FilterType enum is exported."""
        assert hasattr(pydiffvg, "FilterType")
        assert pydiffvg.FilterType.box == 0
        assert pydiffvg.FilterType.tent == 1
        assert pydiffvg.FilterType.radial_paraboloid == 2
        assert pydiffvg.FilterType.hann == 3

    def test_output_type_enum_exists(self):
        """Verify OutputType enum is exported."""
        assert hasattr(pydiffvg, "OutputType")
        assert pydiffvg.OutputType.color == 1
        assert pydiffvg.OutputType.sdf == 2

    def test_pixel_filter_exists(self):
        """Verify PixelFilter class is exported."""
        assert hasattr(pydiffvg, "PixelFilter")
        filt = pydiffvg.PixelFilter(type=pydiffvg.FilterType.box, radius=torch.tensor(0.5))
        assert filt.type == pydiffvg.FilterType.box

    def test_from_svg_path_exists(self):
        """Verify from_svg_path function is exported."""
        assert hasattr(pydiffvg, "from_svg_path")
        assert callable(pydiffvg.from_svg_path)

    def test_print_timing_exists(self):
        """Verify print_timing and set_print_timing are exported."""
        assert hasattr(pydiffvg, "print_timing")
        assert hasattr(pydiffvg, "set_print_timing")
        assert callable(pydiffvg.set_print_timing)

    def test_serialize_scene_exists(self):
        """Verify RenderFunction.serialize_scene exists."""
        assert hasattr(pydiffvg.RenderFunction, "serialize_scene")
        assert callable(pydiffvg.RenderFunction.serialize_scene)

    def test_shape_group_accepts_raw_tensor_color(self, device):
        """ShapeGroup accepts raw tensor for fill_color (original API)."""
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device),
        )
        # Should be automatically wrapped in SolidColor
        assert isinstance(group.fill_color, pydiffvg.SolidColor)

    def test_serialize_scene_and_apply(self, device):
        """Test original API pattern: serialize_scene + apply."""
        circle = pydiffvg.Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device),
        )

        # Original API pattern
        args = pydiffvg.RenderFunction.serialize_scene(64, 64, [circle], [group])
        img = pydiffvg.RenderFunction.apply(64, 64, 2, 2, 0, None, *args)

        assert img.shape == (64, 64, 4)
        # Center should be red
        assert img[32, 32, 0] > 0.5

    def test_shape_constructor_matches_original(self, device):
        """Verify shape constructors match original API."""
        # Circle: radius, center, stroke_width, id
        circle = pydiffvg.Circle(
            radius=torch.tensor(10.0),
            center=torch.tensor([50.0, 50.0]),
            stroke_width=torch.tensor(2.0),
            id="circle1",
        )
        assert circle.radius.item() == 10.0
        assert circle.stroke_width.item() == 2.0
        assert circle.id == "circle1"

        # Path: num_control_points, points, is_closed, stroke_width, id, use_distance_approx
        path = pydiffvg.Path(
            num_control_points=torch.tensor([0, 0]),
            points=torch.tensor([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]),
            is_closed=True,
            stroke_width=torch.tensor(1.0),
            id="path1",
            use_distance_approx=True,
        )
        assert path.is_closed
        assert path.use_distance_approx
        assert path.id == "path1"

    def test_shape_group_constructor_matches_original(self, device):
        """Verify ShapeGroup constructor matches original API."""
        # Original order: shape_ids, fill_color, use_even_odd_rule, stroke_color, shape_to_canvas, id
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]),
            use_even_odd_rule=False,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            shape_to_canvas=torch.eye(3) * 2,
            id="group1",
        )
        assert group.use_even_odd_rule is False
        assert isinstance(group.fill_color, pydiffvg.SolidColor)
        assert isinstance(group.stroke_color, pydiffvg.SolidColor)
        assert group.id == "group1"
