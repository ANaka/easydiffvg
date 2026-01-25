"""Tests for SVG parsing and saving."""

import pytest
import tempfile
from pathlib import Path

import torch

import pydiffvg
from pydiffvg import (
    Circle,
    Rect,
    ShapeGroup,
    SolidColor,
    LinearGradient,
    parse_svg,
    save_svg,
)


class TestSvgParse:
    def test_parse_svg_circle(self, tmp_path):
        """Parse simple SVG with circle."""
        svg_content = """<?xml version="1.0"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <circle cx="50" cy="50" r="30" fill="red"/>
        </svg>"""

        svg_file = tmp_path / "test.svg"
        svg_file.write_text(svg_content)

        width, height, shapes, groups = parse_svg(str(svg_file))

        assert width == 100
        assert height == 100
        assert len(shapes) == 1
        assert len(groups) == 1
        assert isinstance(shapes[0], Circle)
        assert float(shapes[0].center[0]) == 50.0
        assert float(shapes[0].center[1]) == 50.0
        assert float(shapes[0].radius) == 30.0

    def test_parse_svg_rect(self, tmp_path):
        """Parse simple SVG with rectangle."""
        svg_content = """<?xml version="1.0"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <rect x="10" y="20" width="60" height="40" fill="blue"/>
        </svg>"""

        svg_file = tmp_path / "test.svg"
        svg_file.write_text(svg_content)

        width, height, shapes, groups = parse_svg(str(svg_file))

        assert len(shapes) == 1
        assert isinstance(shapes[0], Rect)
        assert float(shapes[0].p_min[0]) == 10.0
        assert float(shapes[0].p_min[1]) == 20.0
        assert float(shapes[0].p_max[0]) == 70.0
        assert float(shapes[0].p_max[1]) == 60.0

    def test_parse_svg_viewbox(self, tmp_path):
        """Parse SVG with viewBox dimensions."""
        svg_content = """<?xml version="1.0"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150">
            <circle cx="100" cy="75" r="50" fill="green"/>
        </svg>"""

        svg_file = tmp_path / "test.svg"
        svg_file.write_text(svg_content)

        width, height, shapes, groups = parse_svg(str(svg_file))

        assert width == 200
        assert height == 150


class TestSvgSave:
    def test_save_svg_circle(self, tmp_path):
        """Save and reload a circle."""
        circle = Circle(
            center=torch.tensor([50.0, 50.0]),
            radius=torch.tensor(30.0),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        svg_file = tmp_path / "output.svg"
        save_svg(str(svg_file), 100, 100, [circle], [group])

        # Verify file exists
        assert svg_file.exists()

        # Reload and check
        width, height, shapes, groups = parse_svg(str(svg_file))
        assert len(shapes) == 1
        assert isinstance(shapes[0], Circle)

    def test_save_svg_rect(self, tmp_path):
        """Save and reload a rectangle."""
        rect = Rect(
            p_min=torch.tensor([10.0, 20.0]),
            p_max=torch.tensor([70.0, 60.0]),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([0.0, 0.0, 1.0, 1.0])),
        )

        svg_file = tmp_path / "output.svg"
        save_svg(str(svg_file), 100, 100, [rect], [group])

        # Reload and check
        width, height, shapes, groups = parse_svg(str(svg_file))
        assert len(shapes) == 1
        assert isinstance(shapes[0], Rect)


class TestSvgRoundtrip:
    def test_roundtrip_preserves_shapes(self, tmp_path):
        """Roundtrip SVG preserves shape geometry."""
        circle = Circle(
            center=torch.tensor([32.0, 32.0]),
            radius=torch.tensor(15.0),
        )
        rect = Rect(
            p_min=torch.tensor([50.0, 10.0]),
            p_max=torch.tensor([90.0, 50.0]),
        )
        group1 = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )
        group2 = ShapeGroup(
            shape_ids=torch.tensor([1], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([0.0, 1.0, 0.0, 1.0])),
        )

        svg_file = tmp_path / "roundtrip.svg"
        save_svg(str(svg_file), 100, 100, [circle, rect], [group1, group2])

        # Reload
        width, height, shapes2, groups2 = parse_svg(str(svg_file))

        assert width == 100
        assert height == 100
        assert len(shapes2) == 2

        # Check circle
        assert isinstance(shapes2[0], Circle)
        torch.testing.assert_close(
            shapes2[0].center, circle.center, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            shapes2[0].radius, circle.radius, rtol=1e-3, atol=1e-3
        )

        # Check rect
        assert isinstance(shapes2[1], Rect)
        torch.testing.assert_close(
            shapes2[1].p_min, rect.p_min, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            shapes2[1].p_max, rect.p_max, rtol=1e-3, atol=1e-3
        )


class TestApiExports:
    def test_parse_svg_exported(self):
        """Verify parse_svg is exported from pydiffvg."""
        assert hasattr(pydiffvg, "parse_svg")
        assert callable(pydiffvg.parse_svg)

    def test_save_svg_exported(self):
        """Verify save_svg is exported from pydiffvg."""
        assert hasattr(pydiffvg, "save_svg")
        assert callable(pydiffvg.save_svg)
