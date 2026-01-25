"""Tests for ShapeGroup."""

import pytest
import torch

from easydiffvg import ShapeGroup, SolidColor, LinearGradient


class TestShapeGroup:
    def test_shape_group_with_fill(self, device):
        """ShapeGroup bundles shapes with fill color."""
        fill = SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
        shape_ids = torch.tensor([0], dtype=torch.int32)

        group = ShapeGroup(
            shape_ids=shape_ids,
            fill_color=fill,
        )

        assert group.shape_ids.shape == (1,)
        assert group.fill_color is not None
        assert group.stroke_color is None

    def test_shape_group_with_stroke(self, device):
        """ShapeGroup can have stroke color."""
        stroke = SolidColor(color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_ids = torch.tensor([0, 1], dtype=torch.int32)

        group = ShapeGroup(
            shape_ids=shape_ids,
            fill_color=None,
            stroke_color=stroke,
        )

        assert group.fill_color is None
        assert group.stroke_color is not None

    def test_shape_group_transform(self, device):
        """ShapeGroup has shape_to_canvas transform matrix."""
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 1.0, 1.0, 1.0])),
        )

        assert group.shape_to_canvas.shape == (3, 3)
        # Default is identity
        torch.testing.assert_close(group.shape_to_canvas, torch.eye(3))

    def test_shape_group_custom_transform(self, device):
        """ShapeGroup accepts custom transform."""
        # Translation matrix
        transform = torch.tensor(
            [
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ]
        )

        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
            shape_to_canvas=transform,
        )

        torch.testing.assert_close(group.shape_to_canvas, transform)

    def test_shape_group_even_odd_rule(self, device):
        """ShapeGroup has use_even_odd_rule flag."""
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
            use_even_odd_rule=True,
        )

        assert group.use_even_odd_rule is True
