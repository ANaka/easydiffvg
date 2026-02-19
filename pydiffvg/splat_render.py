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


def splat_render_cubics(
    cubics: torch.Tensor,
    stroke_widths: torch.Tensor,
    canvas_size: int = 224,
    num_samples: int = 20,
    pixel_chunk_size: int = 2048,
) -> torch.Tensor:
    """Render cubic Bezier strokes via Gaussian splatting.

    Pure PyTorch, fully differentiable, batched.

    Args:
        cubics: (B, num_strokes, 4, 2) cubic Bezier control points in [-1, 1].
        stroke_widths: (B, num_strokes) stroke widths in pixels.
        canvas_size: Output image resolution (square).
        num_samples: Number of sample points per Bezier curve.
        pixel_chunk_size: Pixels to process at once (memory control).

    Returns:
        (B, H, W) grayscale image. White background (1.0), black strokes (0.0).
    """
    B, num_strokes, _, _ = cubics.shape
    device = cubics.device
    dtype = cubics.dtype
    H = W = canvas_size
    K = num_samples

    # Sample points and tangents along curves
    t_vals = torch.linspace(0, 1, K, device=device, dtype=dtype)
    positions = _evaluate_bezier(cubics, t_vals)  # (B, num_strokes, K, 2)
    tangents = _evaluate_bezier_tangent(cubics, t_vals)

    # Convert from [-1, 1] to pixel coordinates
    means = (positions + 1.0) / 2.0 * canvas_size  # (B, num_strokes, K, 2)

    # Compute rotation angles from tangents
    angles = torch.atan2(tangents[..., 1], tangents[..., 0])  # (B, num_strokes, K)

    # Compute sigma_along (half distance between consecutive samples)
    diffs = means[:, :, 1:, :] - means[:, :, :-1, :]
    dists = torch.norm(diffs, dim=-1)  # (B, num_strokes, K-1)

    sigma_along = torch.zeros(B, num_strokes, K, device=device, dtype=dtype)
    sigma_along[:, :, 1:] += dists
    sigma_along[:, :, :-1] += dists
    sigma_along[:, :, 1:-1] /= 2.0
    sigma_along = sigma_along * 0.5
    sigma_along = sigma_along.clamp(min=0.1)

    # sigma_across from stroke width
    sigma_across = stroke_widths.unsqueeze(-1).expand(-1, -1, K)  # (B, num_strokes, K)

    # Precompute rotation and inverse variance terms
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    inv_sa2 = 1.0 / (sigma_along * sigma_along + 1e-8)
    inv_sc2 = 1.0 / (sigma_across * sigma_across + 1e-8)

    # Flatten for efficient computation
    G = num_strokes * K
    means_flat = means.reshape(B, G, 2)
    cos_flat = cos_a.reshape(B, G)
    sin_flat = sin_a.reshape(B, G)
    inv_sa2_flat = inv_sa2.reshape(B, G)
    inv_sc2_flat = inv_sc2.reshape(B, G)

    # Create pixel grid
    py = torch.arange(H, device=device, dtype=dtype) + 0.5
    px = torch.arange(W, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    pixels = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    total_pixels = pixels.shape[0]

    # Adaptive chunk size for memory efficiency
    bytes_per_element = 4 if dtype == torch.float32 else 2
    max_chunk_memory = 512 * 1024 * 1024
    adaptive_chunk = max(64, max_chunk_memory // (B * G * bytes_per_element * 6))
    chunk_size = min(pixel_chunk_size, adaptive_chunk, total_pixels)

    def _splat_chunk(chunk_pixels, means_f, cos_f, sin_f, inv_sa2_f, inv_sc2_f):
        """Compute alpha for a chunk of pixels."""
        dx = chunk_pixels[None, None, :, 0] - means_f[:, :, None, 0]
        dy = chunk_pixels[None, None, :, 1] - means_f[:, :, None, 1]
        d_along = cos_f[:, :, None] * dx + sin_f[:, :, None] * dy
        d_across = -sin_f[:, :, None] * dx + cos_f[:, :, None] * dy
        mahal_sq = d_along.square() * inv_sa2_f[:, :, None] + d_across.square() * inv_sc2_f[:, :, None]
        alpha = torch.exp(-0.5 * mahal_sq.clamp(max=20.0))
        return alpha.sum(dim=1)

    # Process in chunks with gradient checkpointing
    chunks = []
    for start in range(0, total_pixels, chunk_size):
        end = min(start + chunk_size, total_pixels)
        chunk_pixels = pixels[start:end]
        chunk_out = checkpoint(
            _splat_chunk, chunk_pixels,
            means_flat, cos_flat, sin_flat, inv_sa2_flat, inv_sc2_flat,
            use_reentrant=False,
        )
        chunks.append(chunk_out)

    output = torch.cat(chunks, dim=1)
    output = output.reshape(B, H, W)
    result = 1.0 - output.clamp(0.0, 1.0)

    return result


def split_path_to_cubics(
    points: torch.Tensor,
    num_control_points: torch.Tensor,
) -> torch.Tensor:
    """Split a multi-segment pydiffvg Path into individual cubic Beziers.

    Converts lines (0 ctrl pts) and quadratics (1 ctrl pt) to cubics (3 ctrl pts).

    Args:
        points: (N, 2) all points in the path
        num_control_points: (num_segments,) control points per segment

    Returns:
        (num_segments, 4, 2) cubic Bezier control points
    """
    cubics = []
    point_idx = 0

    for seg_idx, n_ctrl in enumerate(num_control_points):
        n_ctrl = int(n_ctrl.item())

        p0 = points[point_idx]

        if n_ctrl == 0:
            # Line segment: P0 -> P1
            p3 = points[point_idx + 1]
            # Convert to cubic: control points at 1/3 and 2/3
            p1 = p0 + (p3 - p0) / 3.0
            p2 = p0 + 2.0 * (p3 - p0) / 3.0
            point_idx += 1

        elif n_ctrl == 1:
            # Quadratic: P0, C, P2 -> Cubic: P0, P0+2/3*(C-P0), P2+2/3*(C-P2), P2
            c = points[point_idx + 1]
            p3 = points[point_idx + 2]
            p1 = p0 + 2.0 / 3.0 * (c - p0)
            p2 = p3 + 2.0 / 3.0 * (c - p3)
            point_idx += 2

        elif n_ctrl == 2:
            # Already cubic: P0, C1, C2, P3
            p1 = points[point_idx + 1]
            p2 = points[point_idx + 2]
            p3 = points[point_idx + 3]
            point_idx += 3

        else:
            raise ValueError(f"Unsupported num_control_points: {n_ctrl}")

        cubic = torch.stack([p0, p1, p2, p3], dim=0)
        cubics.append(cubic)

    return torch.stack(cubics, dim=0)
