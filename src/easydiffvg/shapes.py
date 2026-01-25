"""Shape primitives for easydiffvg."""

from dataclasses import dataclass, field

import torch


@dataclass
class Circle:
    """A circle shape defined by center point and radius."""

    radius: torch.Tensor
    center: torch.Tensor
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""


@dataclass
class Ellipse:
    """An ellipse shape defined by center and radii (rx, ry)."""

    radius: torch.Tensor  # [2] rx, ry
    center: torch.Tensor  # [2] x, y
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""


@dataclass
class Rect:
    """A rectangle defined by min and max corners."""

    p_min: torch.Tensor  # [2] top-left corner
    p_max: torch.Tensor  # [2] bottom-right corner
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""


@dataclass
class Polygon:
    """A polygon or polyline defined by a sequence of points."""

    points: torch.Tensor  # [N, 2] vertices
    is_closed: bool
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""


@dataclass
class Path:
    """A path composed of line and bezier curve segments.

    Each segment is defined by num_control_points:
    - 0: line segment (2 points: start, end)
    - 1: quadratic bezier (3 points: start, control, end)
    - 2: cubic bezier (4 points: start, ctrl1, ctrl2, end)

    Points are shared between segments (end of one = start of next).
    """

    num_control_points: torch.Tensor  # [M] per-segment control point count
    points: torch.Tensor  # [N, 2] all control points
    is_closed: bool
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
    use_distance_approx: bool = False


# Type alias for any shape
Shape = Circle | Ellipse | Rect | Polygon | Path
