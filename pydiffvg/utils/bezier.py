"""Bezier curve utilities for pydiffvg."""

import torch


def evaluate_quadratic(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """Evaluate a quadratic bezier curve at parameter t.

    B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2

    Args:
        p0: Start point [2] or batch [N, 2]
        p1: Control point [2] or batch [N, 2]
        p2: End point [2] or batch [N, 2]
        t: Parameter in [0, 1], scalar or tensor

    Returns:
        Point(s) on the curve, same shape as input points
    """
    if isinstance(t, float):
        t = torch.tensor(t, dtype=p0.dtype, device=p0.device)

    one_minus_t = 1.0 - t
    return one_minus_t**2 * p0 + 2.0 * one_minus_t * t * p1 + t**2 * p2


def evaluate_cubic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate a cubic bezier curve at parameter t.

    B(t) = (1-t)^3 * P0 + 3*(1-t)^2*t * P1 + 3*(1-t)*t^2 * P2 + t^3 * P3

    Args:
        p0: Start point [2] or batch [N, 2]
        p1: First control point [2] or batch [N, 2]
        p2: Second control point [2] or batch [N, 2]
        p3: End point [2] or batch [N, 2]
        t: Parameter in [0, 1], scalar or tensor

    Returns:
        Point(s) on the curve, same shape as input points
    """
    if isinstance(t, float):
        t = torch.tensor(t, dtype=p0.dtype, device=p0.device)

    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t**2
    one_minus_t_cu = one_minus_t_sq * one_minus_t
    t_sq = t**2
    t_cu = t_sq * t

    return (
        one_minus_t_cu * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cu * p3
    )


def quadratic_to_cubic(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a quadratic bezier to a cubic bezier.

    Any quadratic B(t) can be exactly represented as a cubic.
    The cubic control points are:
        C0 = P0
        C1 = P0 + 2/3 * (P1 - P0) = 1/3 * P0 + 2/3 * P1
        C2 = P2 + 2/3 * (P1 - P2) = 2/3 * P1 + 1/3 * P2
        C3 = P2

    Args:
        p0: Start point [2]
        p1: Control point [2]
        p2: End point [2]

    Returns:
        Tuple of (c0, c1, c2, c3) cubic control points
    """
    c0 = p0
    c1 = (1.0 / 3.0) * p0 + (2.0 / 3.0) * p1
    c2 = (2.0 / 3.0) * p1 + (1.0 / 3.0) * p2
    c3 = p2
    return c0, c1, c2, c3


def subdivide_cubic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float = 0.5,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Subdivide a cubic bezier at parameter t using de Casteljau's algorithm.

    Args:
        p0, p1, p2, p3: Control points of the cubic bezier
        t: Subdivision parameter (default 0.5 for midpoint)

    Returns:
        Two tuples of control points for the left and right sub-curves
    """
    if isinstance(t, float):
        t = torch.tensor(t, dtype=p0.dtype, device=p0.device)

    # First level
    q0 = (1 - t) * p0 + t * p1
    q1 = (1 - t) * p1 + t * p2
    q2 = (1 - t) * p2 + t * p3

    # Second level
    r0 = (1 - t) * q0 + t * q1
    r1 = (1 - t) * q1 + t * q2

    # Third level - the point on the curve
    s = (1 - t) * r0 + t * r1

    # Left curve: p0 -> q0 -> r0 -> s
    # Right curve: s -> r1 -> q2 -> p3
    return (p0, q0, r0, s), (s, r1, q2, p3)


def cubic_bounding_box(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the axis-aligned bounding box of a cubic bezier.

    The bounding box is found by:
    1. Always including the endpoints
    2. Finding extrema by solving B'(t) = 0 for each dimension

    Args:
        p0, p1, p2, p3: Control points [2] each

    Returns:
        (min_point, max_point) tuple, each of shape [2]
    """
    # Start with endpoints
    min_pt = torch.minimum(p0, p3)
    max_pt = torch.maximum(p0, p3)

    # For each dimension, find extrema
    # B'(t) = 3(1-t)^2(p1-p0) + 6(1-t)t(p2-p1) + 3t^2(p3-p2)
    # This is a quadratic in t: at^2 + bt + c = 0
    # where a = 3(-p0 + 3p1 - 3p2 + p3)
    #       b = 6(p0 - 2p1 + p2)
    #       c = 3(p1 - p0)

    for dim in range(2):
        a = 3.0 * (-p0[dim] + 3.0 * p1[dim] - 3.0 * p2[dim] + p3[dim])
        b = 6.0 * (p0[dim] - 2.0 * p1[dim] + p2[dim])
        c = 3.0 * (p1[dim] - p0[dim])

        # Solve quadratic
        if torch.abs(a) < 1e-10:
            # Linear case
            if torch.abs(b) > 1e-10:
                t = -c / b
                if 0.0 < t < 1.0:
                    pt = evaluate_cubic(p0, p1, p2, p3, t)
                    min_pt[dim] = torch.minimum(min_pt[dim], pt[dim])
                    max_pt[dim] = torch.maximum(max_pt[dim], pt[dim])
        else:
            # Quadratic case
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0:
                sqrt_d = torch.sqrt(discriminant)
                t1 = (-b + sqrt_d) / (2.0 * a)
                t2 = (-b - sqrt_d) / (2.0 * a)

                for t in [t1, t2]:
                    if 0.0 < t < 1.0:
                        pt = evaluate_cubic(p0, p1, p2, p3, t)
                        min_pt[dim] = torch.minimum(min_pt[dim], pt[dim])
                        max_pt[dim] = torch.maximum(max_pt[dim], pt[dim])

    return min_pt, max_pt


def evaluate_cubic_derivative(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate the derivative of a cubic bezier at parameter t.

    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)

    Args:
        p0, p1, p2, p3: Control points
        t: Parameter in [0, 1]

    Returns:
        Tangent vector at t
    """
    if isinstance(t, float):
        t = torch.tensor(t, dtype=p0.dtype, device=p0.device)

    one_minus_t = 1.0 - t
    return (
        3.0 * one_minus_t**2 * (p1 - p0)
        + 6.0 * one_minus_t * t * (p2 - p1)
        + 3.0 * t**2 * (p3 - p2)
    )


def evaluate_quadratic_derivative(
    p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, t: float | torch.Tensor
) -> torch.Tensor:
    """Evaluate the derivative of a quadratic bezier at parameter t.

    B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)

    Args:
        p0, p1, p2: Control points
        t: Parameter in [0, 1]

    Returns:
        Tangent vector at t
    """
    if isinstance(t, float):
        t = torch.tensor(t, dtype=p0.dtype, device=p0.device)

    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1)
