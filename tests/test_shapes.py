"""Tests for shape primitives."""

import pytest
import torch

from easydiffvg import Circle, Ellipse, Rect, Polygon, Path


class TestCircle:
    def test_circle_creation(self, device):
        """Circle stores center and radius as tensors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor(10.0, device=device)

        circle = Circle(radius=radius, center=center)

        assert circle.center.shape == (2,)
        assert circle.radius.shape == ()
        torch.testing.assert_close(circle.center, center)
        torch.testing.assert_close(circle.radius, radius)

    def test_circle_has_stroke_width(self, device):
        """Circle has stroke_width attribute with default."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor(10.0, device=device)

        circle = Circle(radius=radius, center=center)

        assert hasattr(circle, "stroke_width")
        assert circle.stroke_width.shape == ()

    def test_circle_custom_stroke_width(self, device):
        """Circle accepts custom stroke_width."""
        circle = Circle(
            radius=torch.tensor(10.0),
            center=torch.tensor([32.0, 32.0]),
            stroke_width=torch.tensor(2.5),
        )

        torch.testing.assert_close(circle.stroke_width, torch.tensor(2.5))


class TestEllipse:
    def test_ellipse_creation(self, device):
        """Ellipse stores center and radius (rx, ry) as tensors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor([20.0, 10.0], device=device)  # rx, ry

        ellipse = Ellipse(radius=radius, center=center)

        assert ellipse.center.shape == (2,)
        assert ellipse.radius.shape == (2,)
        torch.testing.assert_close(ellipse.center, center)
        torch.testing.assert_close(ellipse.radius, radius)

    def test_ellipse_has_stroke_width(self, device):
        """Ellipse has stroke_width attribute with default."""
        ellipse = Ellipse(
            radius=torch.tensor([20.0, 10.0]),
            center=torch.tensor([32.0, 32.0]),
        )

        assert hasattr(ellipse, "stroke_width")


class TestRect:
    def test_rect_creation(self, device):
        """Rect stores p_min and p_max corners as tensors."""
        p_min = torch.tensor([10.0, 10.0], device=device)
        p_max = torch.tensor([50.0, 40.0], device=device)

        rect = Rect(p_min=p_min, p_max=p_max)

        assert rect.p_min.shape == (2,)
        assert rect.p_max.shape == (2,)
        torch.testing.assert_close(rect.p_min, p_min)
        torch.testing.assert_close(rect.p_max, p_max)

    def test_rect_has_stroke_width(self, device):
        """Rect has stroke_width attribute with default."""
        rect = Rect(
            p_min=torch.tensor([10.0, 10.0]),
            p_max=torch.tensor([50.0, 40.0]),
        )

        assert hasattr(rect, "stroke_width")
        assert rect.stroke_width.shape == ()


class TestPolygon:
    def test_polygon_creation(self, device):
        """Polygon stores points as [N, 2] tensor."""
        points = torch.tensor(
            [
                [10.0, 10.0],
                [50.0, 10.0],
                [30.0, 50.0],
            ],
            device=device,
        )

        polygon = Polygon(points=points, is_closed=True)

        assert polygon.points.shape == (3, 2)
        assert polygon.is_closed is True
        torch.testing.assert_close(polygon.points, points)

    def test_polygon_open(self, device):
        """Polygon can be open (polyline)."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 20.0],
                [20.0, 0.0],
            ],
            device=device,
        )

        polygon = Polygon(points=points, is_closed=False)

        assert polygon.is_closed is False

    def test_polygon_has_stroke_width(self, device):
        """Polygon has stroke_width attribute."""
        polygon = Polygon(
            points=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            is_closed=False,
        )

        assert hasattr(polygon, "stroke_width")


class TestPath:
    def test_path_creation_cubic(self, device):
        """Path with cubic bezier segment (2 control points)."""
        # Cubic bezier: start, ctrl1, ctrl2, end
        points = torch.tensor(
            [
                [0.0, 0.0],  # start
                [10.0, 30.0],  # ctrl1
                [30.0, 30.0],  # ctrl2
                [40.0, 0.0],  # end
            ],
            device=device,
        )
        num_control_points = torch.tensor([2], dtype=torch.int32)  # cubic

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (4, 2)
        assert path.num_control_points.shape == (1,)
        assert path.is_closed is False

    def test_path_creation_quadratic(self, device):
        """Path with quadratic bezier segment (1 control point)."""
        points = torch.tensor(
            [
                [0.0, 0.0],  # start
                [20.0, 40.0],  # ctrl
                [40.0, 0.0],  # end
            ],
            device=device,
        )
        num_control_points = torch.tensor([1], dtype=torch.int32)  # quadratic

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (3, 2)

    def test_path_creation_line(self, device):
        """Path with line segment (0 control points)."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [40.0, 40.0],
            ],
            device=device,
        )
        num_control_points = torch.tensor([0], dtype=torch.int32)  # line

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (2, 2)

    def test_path_closed(self, device):
        """Closed path forms a loop."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [40.0, 0.0],
                [40.0, 40.0],
                [0.0, 40.0],
            ],
            device=device,
        )
        # 4 points, 4 line segments (closed)
        num_control_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=True,
        )

        assert path.is_closed is True

    def test_path_has_use_distance_approx(self, device):
        """Path has use_distance_approx flag (default False)."""
        path = Path(
            num_control_points=torch.tensor([0]),
            points=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            is_closed=False,
        )

        assert hasattr(path, "use_distance_approx")
        assert path.use_distance_approx is False
