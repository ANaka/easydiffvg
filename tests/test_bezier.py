"""Tests for bezier curve utilities."""

import pytest
import torch

from easydiffvg.utils.bezier import (
    evaluate_quadratic,
    evaluate_cubic,
    quadratic_to_cubic,
    subdivide_cubic,
    cubic_bounding_box,
)


class TestBezierEvaluation:
    def test_evaluate_quadratic_at_start(self, device):
        """Quadratic bezier at t=0 returns start point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=0.0)

        torch.testing.assert_close(result, p0)

    def test_evaluate_quadratic_at_end(self, device):
        """Quadratic bezier at t=1 returns end point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=1.0)

        torch.testing.assert_close(result, p2)

    def test_evaluate_quadratic_at_midpoint(self, device):
        """Quadratic bezier at t=0.5 is correct."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=0.5)

        # B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2
        expected = 0.25 * p0 + 0.5 * p1 + 0.25 * p2
        torch.testing.assert_close(result, expected)

    def test_evaluate_cubic_at_start(self, device):
        """Cubic bezier at t=0 returns start point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([33.0, 100.0], device=device)
        p2 = torch.tensor([66.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_cubic(p0, p1, p2, p3, t=0.0)

        torch.testing.assert_close(result, p0)

    def test_evaluate_cubic_at_end(self, device):
        """Cubic bezier at t=1 returns end point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([33.0, 100.0], device=device)
        p2 = torch.tensor([66.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_cubic(p0, p1, p2, p3, t=1.0)

        torch.testing.assert_close(result, p3)


class TestQuadraticToCubic:
    def test_conversion_preserves_endpoints(self, device):
        """Converted cubic has same endpoints as quadratic."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        c0, c1, c2, c3 = quadratic_to_cubic(p0, p1, p2)

        torch.testing.assert_close(c0, p0)
        torch.testing.assert_close(c3, p2)

    def test_conversion_preserves_midpoint(self, device):
        """Converted cubic passes through same midpoint as quadratic."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        c0, c1, c2, c3 = quadratic_to_cubic(p0, p1, p2)

        quad_mid = evaluate_quadratic(p0, p1, p2, t=0.5)
        cubic_mid = evaluate_cubic(c0, c1, c2, c3, t=0.5)

        torch.testing.assert_close(quad_mid, cubic_mid)


class TestSubdivideCubic:
    def test_subdivision_at_midpoint(self, device):
        """Subdivision at t=0.5 creates two valid curves."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([33.0, 100.0], device=device)
        p2 = torch.tensor([66.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        left, right = subdivide_cubic(p0, p1, p2, p3, t=0.5)

        # Left curve starts at p0
        torch.testing.assert_close(left[0], p0)

        # Right curve ends at p3
        torch.testing.assert_close(right[3], p3)

        # Curves meet at subdivision point
        torch.testing.assert_close(left[3], right[0])

        # Meeting point is on original curve at t=0.5
        expected = evaluate_cubic(p0, p1, p2, p3, t=0.5)
        torch.testing.assert_close(left[3], expected)


class TestCubicBoundingBox:
    def test_bounding_box_contains_endpoints(self, device):
        """Bounding box contains curve endpoints."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([33.0, 100.0], device=device)
        p2 = torch.tensor([66.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        bbox_min, bbox_max = cubic_bounding_box(p0, p1, p2, p3)

        assert bbox_min[0] <= p0[0] and bbox_max[0] >= p0[0]
        assert bbox_min[1] <= p0[1] and bbox_max[1] >= p0[1]
        assert bbox_min[0] <= p3[0] and bbox_max[0] >= p3[0]
        assert bbox_min[1] <= p3[1] and bbox_max[1] >= p3[1]

    def test_bounding_box_contains_extrema(self, device):
        """Bounding box contains curve extrema (y max in this case)."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([33.0, 100.0], device=device)
        p2 = torch.tensor([66.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        bbox_min, bbox_max = cubic_bounding_box(p0, p1, p2, p3)

        # The curve bulges up, max y should be > 0
        assert bbox_max[1] > 0
