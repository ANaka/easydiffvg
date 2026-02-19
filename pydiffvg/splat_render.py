"""Pure PyTorch Bezier splatting renderer.

Drop-in replacement for RenderFunction that uses Gaussian splatting
instead of diffvg's CUDA kernels. Works on all GPU architectures.

Based on "Bezier Splatting for Fast and Differentiable Vector Graphics Rendering"
(arXiv:2503.16424).
"""

import torch
from torch.utils.checkpoint import checkpoint


def _evaluate_bezier(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curves at parameter values t.

    Uses Bernstein basis:
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Args:
        control_points: (B, num_strokes, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, num_strokes, K, 2) positions on curves.
    """
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t

    b0 = omt * omt * omt
    b1 = 3.0 * omt * omt * t
    b2 = 3.0 * omt * t * t
    b3 = t * t * t

    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _evaluate_bezier_tangent(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate tangent (derivative) of cubic Bezier curves at parameter values t.

    B'(t) = 3[(1-t)^2 (P1-P0) + 2(1-t)t (P2-P1) + t^2 (P3-P2)]

    Args:
        control_points: (B, num_strokes, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, num_strokes, K, 2) tangent vectors.
    """
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t

    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    d0 = omt * omt
    d1 = 2.0 * omt * t
    d2 = t * t

    return 3.0 * (d0 * (p1 - p0) + d1 * (p2 - p1) + d2 * (p3 - p2))
