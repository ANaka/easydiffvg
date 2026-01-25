"""Integration tests for pydiffvg end-to-end workflows."""

import pytest
import tempfile
from pathlib import Path

import torch

import pydiffvg


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_render_and_optimize_with_soft_raster(self, device):
        """End-to-end: render with soft rasterization, compute loss, backprop, update."""
        # Create a circle we want to optimize
        center = torch.tensor([30.0, 30.0], device=device, requires_grad=True)
        circle = pydiffvg.Circle(
            radius=torch.tensor(15.0, device=device),
            center=center,
        )
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=pydiffvg.SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        # Target: circle at center of canvas
        target_center = torch.tensor([32.0, 32.0], device=device)

        # Create target image
        target_circle = pydiffvg.Circle(
            radius=torch.tensor(15.0, device=device),
            center=target_center,
        )
        target_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=pydiffvg.SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )
        target_img = pydiffvg.render_differentiable(
            64, 64, [target_circle], [target_group]
        )

        # Optimization loop
        optimizer = torch.optim.Adam([center], lr=1.0)

        initial_dist = (center.detach() - target_center).pow(2).sum().sqrt()

        for _ in range(10):
            optimizer.zero_grad()
            img = pydiffvg.render_differentiable(64, 64, [circle], [group])
            loss = ((img - target_img) ** 2).sum()
            loss.backward()
            optimizer.step()

        final_dist = (center.detach() - target_center).pow(2).sum().sqrt()

        # Center should have moved toward target
        assert final_dist < initial_dist
        assert center.requires_grad

    def test_svg_roundtrip_and_render(self, device):
        """Load SVG, render, verify output."""
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="20" fill="red"/>
</svg>"""

        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as f:
            f.write(svg_content)
            svg_path = f.name

        try:
            # Parse SVG
            width, height, shapes, groups = pydiffvg.parse_svg(svg_path)

            assert width == 64
            assert height == 64
            assert len(shapes) == 1
            assert isinstance(shapes[0], pydiffvg.Circle)

            # Render
            img = pydiffvg.render(width, height, shapes, groups)

            assert img.shape == (64, 64, 4)
            # Center should have red color
            assert img[32, 32, 0] > 0.5  # Red
            assert img[32, 32, 3] > 0.5  # Alpha

            # Save and reload
            output_path = svg_path + ".out.svg"
            pydiffvg.save_svg(output_path, width, height, shapes, groups)

            width2, height2, shapes2, groups2 = pydiffvg.parse_svg(output_path)
            assert width2 == width
            assert height2 == height
            assert len(shapes2) == 1

            Path(output_path).unlink()
        finally:
            Path(svg_path).unlink()

    def test_multiple_shapes_optimization(self, device):
        """Optimize multiple shapes simultaneously."""
        # Two circles with optimizable centers
        center1 = torch.tensor([20.0, 20.0], device=device, requires_grad=True)
        center2 = torch.tensor([44.0, 44.0], device=device, requires_grad=True)

        circle1 = pydiffvg.Circle(
            radius=torch.tensor(10.0, device=device),
            center=center1,
        )
        circle2 = pydiffvg.Circle(
            radius=torch.tensor(10.0, device=device),
            center=center2,
        )

        group1 = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32, device=device),
            fill_color=pydiffvg.SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )
        group2 = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([1], dtype=torch.int32, device=device),
            fill_color=pydiffvg.SolidColor(
                color=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
            ),
        )

        # Target: both circles at center
        target = torch.tensor([32.0, 32.0], device=device)

        optimizer = torch.optim.Adam([center1, center2], lr=1.0)

        initial_dist1 = (center1.detach() - target).pow(2).sum().sqrt()
        initial_dist2 = (center2.detach() - target).pow(2).sum().sqrt()

        for _ in range(5):
            optimizer.zero_grad()
            img = pydiffvg.render_differentiable(
                64, 64, [circle1, circle2], [group1, group2]
            )
            # Loss: mean squared pixel distance from center
            y, x = torch.meshgrid(
                torch.arange(64, device=device, dtype=torch.float32),
                torch.arange(64, device=device, dtype=torch.float32),
                indexing="ij",
            )
            center_dist = ((x - 32) ** 2 + (y - 32) ** 2).unsqueeze(-1)
            loss = (img * center_dist).sum()
            loss.backward()
            optimizer.step()

        # Both should have moved toward center
        final_dist1 = (center1.detach() - target).pow(2).sum().sqrt()
        final_dist2 = (center2.detach() - target).pow(2).sum().sqrt()

        assert final_dist1 < initial_dist1
        assert final_dist2 < initial_dist2


class TestApiCompleteness:
    """Verify all expected API elements are exported."""

    def test_all_shapes_exported(self):
        """All shape classes are accessible."""
        assert hasattr(pydiffvg, "Circle")
        assert hasattr(pydiffvg, "Ellipse")
        assert hasattr(pydiffvg, "Rect")
        assert hasattr(pydiffvg, "Polygon")
        assert hasattr(pydiffvg, "Path")

    def test_all_colors_exported(self):
        """All color classes are accessible."""
        assert hasattr(pydiffvg, "SolidColor")
        assert hasattr(pydiffvg, "LinearGradient")
        assert hasattr(pydiffvg, "RadialGradient")
        assert hasattr(pydiffvg, "Color")

    def test_all_render_functions_exported(self):
        """All render functions are accessible."""
        assert hasattr(pydiffvg, "render")
        assert hasattr(pydiffvg, "render_differentiable")
        assert hasattr(pydiffvg, "RenderFunction")

    def test_svg_functions_exported(self):
        """SVG I/O functions are accessible."""
        assert hasattr(pydiffvg, "parse_svg")
        assert hasattr(pydiffvg, "save_svg")

    def test_version_defined(self):
        """Package version is defined."""
        assert hasattr(pydiffvg, "__version__")
        assert isinstance(pydiffvg.__version__, str)
