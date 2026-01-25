"""Distance field utilities for easydiffvg."""

import torch

from easydiffvg.utils.bezier import evaluate_quadratic, evaluate_cubic


def distance_to_line_segment(
    point: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor
) -> torch.Tensor:
    """Compute distance from a point to a line segment.

    Args:
        point: Query point [2] or batch [..., 2]
        p0: Start of segment [2]
        p1: End of segment [2]

    Returns:
        Distance (scalar or batch matching point shape[:-1])
    """
    # Vector from p0 to p1
    v = p1 - p0
    # Vector from p0 to point
    w = point - p0

    # Project w onto v, clamped to [0, 1]
    v_dot_v = torch.dot(v, v)

    if v_dot_v < 1e-10:
        # Degenerate segment (p0 == p1)
        return torch.norm(w, dim=-1)

    t = torch.clamp(torch.dot(w, v) / v_dot_v, 0.0, 1.0)

    # Closest point on segment
    closest = p0 + t * v

    return torch.norm(point - closest, dim=-1)


def distance_to_quadratic_bezier(
    point: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    num_samples: int = 16,
) -> torch.Tensor:
    """Compute approximate distance from a point to a quadratic bezier curve.

    Uses sampling-based approximation for simplicity and differentiability.

    Args:
        point: Query point [2]
        p0, p1, p2: Control points [2] each
        num_samples: Number of samples along the curve

    Returns:
        Approximate minimum distance
    """
    t_values = torch.linspace(0, 1, num_samples, device=point.device, dtype=point.dtype)

    min_dist_sq = torch.tensor(float("inf"), device=point.device, dtype=point.dtype)

    for t in t_values:
        curve_point = evaluate_quadratic(p0, p1, p2, t)
        dist_sq = torch.sum((point - curve_point) ** 2)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return torch.sqrt(min_dist_sq)


def distance_to_cubic_bezier(
    point: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_samples: int = 16,
) -> torch.Tensor:
    """Compute approximate distance from a point to a cubic bezier curve.

    Uses sampling-based approximation for simplicity and differentiability.

    Args:
        point: Query point [2]
        p0, p1, p2, p3: Control points [2] each
        num_samples: Number of samples along the curve

    Returns:
        Approximate minimum distance
    """
    t_values = torch.linspace(0, 1, num_samples, device=point.device, dtype=point.dtype)

    min_dist_sq = torch.tensor(float("inf"), device=point.device, dtype=point.dtype)

    for t in t_values:
        curve_point = evaluate_cubic(p0, p1, p2, p3, t)
        dist_sq = torch.sum((point - curve_point) ** 2)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    return torch.sqrt(min_dist_sq)


def signed_distance_circle(
    point: torch.Tensor, center: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    """Compute signed distance from a point to a circle.

    Negative inside, positive outside.

    Args:
        point: Query point [2] or batch [..., 2]
        center: Circle center [2]
        radius: Circle radius (scalar)

    Returns:
        Signed distance
    """
    return torch.norm(point - center, dim=-1) - radius


def signed_distance_ellipse(
    point: torch.Tensor, center: torch.Tensor, radius: torch.Tensor
) -> torch.Tensor:
    """Compute approximate signed distance from a point to an ellipse.

    Uses a simple approximation that's not exact but is differentiable.

    Args:
        point: Query point [2]
        center: Ellipse center [2]
        radius: Ellipse radii [2] (rx, ry)

    Returns:
        Approximate signed distance (negative inside, positive outside)
    """
    # Normalize to unit circle
    normalized = (point - center) / radius
    dist_normalized = torch.norm(normalized, dim=-1)

    # Scale back using average radius as approximation
    avg_radius = (radius[0] + radius[1]) / 2.0

    # This is an approximation - exact ellipse SDF is complex
    return (dist_normalized - 1.0) * avg_radius


def signed_distance_rect(
    point: torch.Tensor, p_min: torch.Tensor, p_max: torch.Tensor
) -> torch.Tensor:
    """Compute signed distance from a point to an axis-aligned rectangle.

    Negative inside, positive outside.

    Args:
        point: Query point [2]
        p_min: Minimum corner [2]
        p_max: Maximum corner [2]

    Returns:
        Signed distance
    """
    center = (p_min + p_max) / 2.0
    half_size = (p_max - p_min) / 2.0

    # Distance from point to center, in local coords
    d = torch.abs(point - center) - half_size

    # Outside distance (positive when outside)
    outside = torch.norm(torch.clamp(d, min=0.0), dim=-1)

    # Inside distance (negative when inside)
    inside = torch.min(torch.clamp(d, max=0.0).max(dim=-1).values, torch.tensor(0.0))

    return outside + inside
