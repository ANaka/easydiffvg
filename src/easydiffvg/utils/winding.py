"""Winding number computation for point-in-shape tests."""

import torch

from easydiffvg.utils.bezier import evaluate_quadratic, evaluate_cubic


def winding_number_line(
    point: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor
) -> torch.Tensor:
    """Compute winding number contribution of a line segment.

    Uses the crossing number method: count how many times a ray from the point
    in the +x direction crosses the segment, with sign based on direction.

    Args:
        point: Query point [2] or batch [N, 2]
        p0: Start of line segment [2] or batch [N, 2]
        p1: End of line segment [2] or batch [N, 2]

    Returns:
        Winding number contribution (scalar or batch)
    """
    # Translate so point is at origin
    v0 = p0 - point
    v1 = p1 - point

    # Check if segment crosses the positive x-axis
    # The segment crosses if y0 and y1 have different signs
    # and the x-intercept is positive

    y0 = v0[..., 1]
    y1 = v1[..., 1]
    x0 = v0[..., 0]
    x1 = v1[..., 0]

    # Compute x at y=0: x = x0 + (x1-x0) * (0-y0) / (y1-y0) = x0 - y0*(x1-x0)/(y1-y0)
    dy = y1 - y0
    # Avoid division by zero
    dy = torch.where(torch.abs(dy) < 1e-10, torch.ones_like(dy) * 1e-10, dy)
    t = -y0 / dy
    x_intercept = x0 + t * (x1 - x0)

    # Crossing conditions
    upward = (y0 <= 0) & (y1 > 0)  # Crosses upward
    downward = (y0 > 0) & (y1 <= 0)  # Crosses downward
    valid_t = (t >= 0) & (t < 1)  # t in [0, 1)
    positive_x = x_intercept > 0  # Crosses to the right

    # Winding contribution
    winding = torch.zeros_like(x0)
    winding = torch.where(upward & valid_t & positive_x, winding + 1, winding)
    winding = torch.where(downward & valid_t & positive_x, winding - 1, winding)

    return winding


def winding_number_quadratic(
    point: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    num_subdivisions: int = 8,
) -> torch.Tensor:
    """Compute winding number contribution of a quadratic bezier segment.

    Approximates by subdividing into line segments.

    Args:
        point: Query point [2]
        p0, p1, p2: Control points [2] each
        num_subdivisions: Number of line segments to use

    Returns:
        Winding number contribution
    """
    winding = torch.zeros((), dtype=point.dtype, device=point.device)

    t_values = torch.linspace(0, 1, num_subdivisions + 1, device=point.device)

    for i in range(num_subdivisions):
        t0 = t_values[i]
        t1 = t_values[i + 1]

        seg_p0 = evaluate_quadratic(p0, p1, p2, t0)
        seg_p1 = evaluate_quadratic(p0, p1, p2, t1)

        winding = winding + winding_number_line(point, seg_p0, seg_p1)

    return winding


def winding_number_cubic(
    point: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_subdivisions: int = 8,
) -> torch.Tensor:
    """Compute winding number contribution of a cubic bezier segment.

    Approximates by subdividing into line segments.

    Args:
        point: Query point [2]
        p0, p1, p2, p3: Control points [2] each
        num_subdivisions: Number of line segments to use

    Returns:
        Winding number contribution
    """
    winding = torch.zeros((), dtype=point.dtype, device=point.device)

    t_values = torch.linspace(0, 1, num_subdivisions + 1, device=point.device)

    for i in range(num_subdivisions):
        t0 = t_values[i]
        t1 = t_values[i + 1]

        seg_p0 = evaluate_cubic(p0, p1, p2, p3, t0)
        seg_p1 = evaluate_cubic(p0, p1, p2, p3, t1)

        winding = winding + winding_number_line(point, seg_p0, seg_p1)

    return winding


def winding_number_polygon(point: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Compute winding number of a point with respect to a closed polygon.

    Args:
        point: Query point [2]
        vertices: Polygon vertices [N, 2] (closed, first and last connected)

    Returns:
        Total winding number
    """
    winding = torch.zeros((), dtype=point.dtype, device=point.device)
    n = vertices.shape[0]

    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        winding = winding + winding_number_line(point, p0, p1)

    return winding
