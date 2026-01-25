"""Differentiable rendering using soft rasterization.

This module provides an alternative rendering approach that is fully differentiable
through PyTorch's standard autograd. Instead of using boundary sampling (which requires
custom autograd implementation), it uses soft rasterization with smooth coverage functions.

The trade-off is slightly different rendering quality, but full differentiability
without custom backward passes.
"""

import math

import torch

from pydiffvg.shapes import Shape, Circle, Ellipse, Rect, Polygon, Path
from pydiffvg.groups import ShapeGroup
from pydiffvg.color import Color, SolidColor, LinearGradient, RadialGradient


def render_differentiable(
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    softness: float = 1.0,
) -> torch.Tensor:
    """Render shapes using soft rasterization for full differentiability.

    This renderer uses sigmoid-based soft coverage instead of hard thresholds,
    making the entire rendering pipeline differentiable via standard autograd.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of shapes
        shape_groups: List of shape groups
        num_samples_x: Samples per pixel in x
        num_samples_y: Samples per pixel in y
        softness: Controls the edge softness (higher = softer edges)

    Returns:
        Rendered image [H, W, 4] RGBA
    """
    if len(shapes) == 0:
        return torch.zeros(canvas_height, canvas_width, 4)

    # Get device/dtype from first shape
    first_shape = shapes[0]
    if isinstance(first_shape, (Circle, Ellipse)):
        device = first_shape.center.device
        dtype = first_shape.center.dtype
    elif isinstance(first_shape, Rect):
        device = first_shape.p_min.device
        dtype = first_shape.p_min.dtype
    else:
        device = first_shape.points.device
        dtype = first_shape.points.dtype

    # Create pixel coordinate grids
    y_coords = torch.arange(canvas_height, device=device, dtype=dtype)
    x_coords = torch.arange(canvas_width, device=device, dtype=dtype)

    # Initialize output image
    image = torch.zeros(canvas_height, canvas_width, 4, device=device, dtype=dtype)

    # Process each group (back to front)
    for group in shape_groups:
        for shape_idx in group.shape_ids:
            shape = shapes[int(shape_idx.item())]

            # Compute soft coverage for entire image at once
            coverage = _compute_soft_coverage_vectorized(
                shape, x_coords, y_coords, group.shape_to_canvas, softness
            )

            if group.fill_color is not None:
                # Get fill color
                fill = _sample_color_vectorized(
                    group.fill_color, x_coords, y_coords, device, dtype
                )

                # Apply coverage to alpha (avoid in-place modification for autograd)
                fill_rgb = fill[..., :3]
                fill_alpha = fill[..., 3:4] * coverage.unsqueeze(-1)
                fill_with_coverage = torch.cat([fill_rgb, fill_alpha], dim=-1)

                # Alpha composite
                image = _alpha_composite_vectorized(fill_with_coverage, image)

            if group.stroke_color is not None:
                # Get stroke width from shape
                stroke_width = getattr(shape, 'stroke_width', torch.tensor(1.0))
                if isinstance(stroke_width, (int, float)):
                    stroke_width = torch.tensor(stroke_width, device=device, dtype=dtype)

                # Compute stroke coverage (pixels near boundary)
                stroke_coverage = _compute_soft_stroke_coverage_vectorized(
                    shape, x_coords, y_coords, group.shape_to_canvas,
                    stroke_width, softness
                )

                # Get stroke color
                stroke = _sample_color_vectorized(
                    group.stroke_color, x_coords, y_coords, device, dtype
                )

                # Apply stroke coverage to alpha
                stroke_rgb = stroke[..., :3]
                stroke_alpha = stroke[..., 3:4] * stroke_coverage.unsqueeze(-1)
                stroke_with_coverage = torch.cat([stroke_rgb, stroke_alpha], dim=-1)

                # Alpha composite stroke over current image
                image = _alpha_composite_vectorized(stroke_with_coverage, image)

    return image


def _compute_soft_coverage_vectorized(
    shape: Shape,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    transform: torch.Tensor,
    softness: float,
) -> torch.Tensor:
    """Compute soft coverage for all pixels at once.

    Args:
        shape: Shape to rasterize
        x_coords: X coordinates [W]
        y_coords: Y coordinates [H]
        transform: Shape-to-canvas transform [3, 3]
        softness: Edge softness

    Returns:
        Coverage [H, W] with values in [0, 1]
    """
    H = len(y_coords)
    W = len(x_coords)

    # Create meshgrid of pixel centers
    yy, xx = torch.meshgrid(y_coords + 0.5, x_coords + 0.5, indexing='ij')
    points = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    # Transform points to shape coordinates
    canvas_to_shape = torch.linalg.inv(transform)
    points_flat = points.reshape(-1, 2)  # [H*W, 2]
    ones = torch.ones(H * W, 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points_flat, ones], dim=-1)  # [H*W, 3]
    points_shape = (points_h @ canvas_to_shape.T)[:, :2]  # [H*W, 2]
    points_shape = points_shape.reshape(H, W, 2)

    # Compute signed distance
    sdf = _compute_sdf_vectorized(shape, points_shape)

    # Soft coverage using sigmoid
    coverage = torch.sigmoid(-sdf / softness)

    return coverage


def _compute_soft_stroke_coverage_vectorized(
    shape: Shape,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    transform: torch.Tensor,
    stroke_width: torch.Tensor,
    softness: float,
) -> torch.Tensor:
    """Compute soft stroke coverage for all pixels at once.

    Stroke is rendered as the area within stroke_width/2 of the shape boundary.

    Args:
        shape: Shape to stroke
        x_coords: X coordinates [W]
        y_coords: Y coordinates [H]
        transform: Shape-to-canvas transform [3, 3]
        stroke_width: Width of the stroke
        softness: Edge softness

    Returns:
        Stroke coverage [H, W] with values in [0, 1]
    """
    H = len(y_coords)
    W = len(x_coords)

    # Create meshgrid of pixel centers
    yy, xx = torch.meshgrid(y_coords + 0.5, x_coords + 0.5, indexing='ij')
    points = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    # Transform points to shape coordinates
    canvas_to_shape = torch.linalg.inv(transform)
    points_flat = points.reshape(-1, 2)  # [H*W, 2]
    ones = torch.ones(H * W, 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points_flat, ones], dim=-1)  # [H*W, 3]
    points_shape = (points_h @ canvas_to_shape.T)[:, :2]  # [H*W, 2]
    points_shape = points_shape.reshape(H, W, 2)

    # Compute signed distance (distance to boundary)
    sdf = _compute_sdf_vectorized(shape, points_shape)

    # Stroke coverage: pixels where |sdf| < stroke_width/2
    # Use soft transition at both edges
    half_width = stroke_width / 2.0

    # Inner edge: sdf > -half_width (not too far inside)
    inner_coverage = torch.sigmoid((sdf + half_width) / softness)
    # Outer edge: sdf < half_width (not too far outside)
    outer_coverage = torch.sigmoid((-sdf + half_width) / softness)

    # Stroke is where both conditions are met
    stroke_coverage = inner_coverage * outer_coverage

    return stroke_coverage


def _compute_sdf_vectorized(
    shape: Shape,
    points: torch.Tensor,
) -> torch.Tensor:
    """Compute signed distance field for all points.

    Negative inside, positive outside.

    Args:
        shape: Shape
        points: Query points [H, W, 2] in shape coordinates

    Returns:
        SDF values [H, W]
    """
    if isinstance(shape, Circle):
        # Distance to circle boundary
        dist_to_center = torch.norm(points - shape.center, dim=-1)
        return dist_to_center - shape.radius

    elif isinstance(shape, Ellipse):
        # Approximate ellipse SDF
        normalized = (points - shape.center) / shape.radius
        dist_normalized = torch.norm(normalized, dim=-1)
        # Scale back
        avg_radius = (shape.radius[0] + shape.radius[1]) / 2.0
        return (dist_normalized - 1.0) * avg_radius

    elif isinstance(shape, Rect):
        # Rectangle SDF
        center = (shape.p_min + shape.p_max) / 2.0
        half_size = (shape.p_max - shape.p_min) / 2.0

        d = torch.abs(points - center) - half_size
        outside = torch.norm(torch.clamp(d, min=0.0), dim=-1)
        inside = torch.clamp(d.max(dim=-1).values, max=0.0)

        return outside + inside

    elif isinstance(shape, Polygon):
        if not shape.is_closed:
            # Open polygon has infinite SDF (no interior)
            return torch.full(points.shape[:-1], float('inf'), device=points.device, dtype=points.dtype)

        # Polygon SDF using winding number (approximate)
        return _polygon_sdf_vectorized(points, shape.points)

    elif isinstance(shape, Path):
        if not shape.is_closed:
            return torch.full(points.shape[:-1], float('inf'), device=points.device, dtype=points.dtype)

        # Approximate path as polygon
        return _polygon_sdf_vectorized(points, shape.points)

    else:
        return torch.full(points.shape[:-1], float('inf'), device=points.device, dtype=points.dtype)


def _polygon_sdf_vectorized(
    points: torch.Tensor,  # [H, W, 2]
    vertices: torch.Tensor,  # [N, 2]
) -> torch.Tensor:
    """Compute approximate SDF for a polygon.

    Uses ray casting for inside/outside and distance to edges for magnitude.
    """
    H, W, _ = points.shape
    N = vertices.shape[0]

    points_flat = points.reshape(-1, 2)  # [H*W, 2]
    num_points = points_flat.shape[0]

    # Compute distance to each edge
    min_dist = torch.full((num_points,), float('inf'), device=points.device, dtype=points.dtype)

    for i in range(N):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % N]

        # Distance to line segment
        v = p1 - p0
        w = points_flat - p0

        # Project onto line
        c1 = (w * v).sum(dim=-1)
        c2 = (v * v).sum()

        if c2 < 1e-10:
            dist = torch.norm(w, dim=-1)
        else:
            t = torch.clamp(c1 / c2, 0.0, 1.0)
            closest = p0 + t.unsqueeze(-1) * v
            dist = torch.norm(points_flat - closest, dim=-1)

        min_dist = torch.minimum(min_dist, dist)

    # Determine inside/outside using crossing number
    inside = _point_in_polygon_vectorized(points_flat, vertices)

    # SDF: negative inside, positive outside
    sdf = torch.where(inside, -min_dist, min_dist)

    return sdf.reshape(H, W)


def _point_in_polygon_vectorized(
    points: torch.Tensor,  # [M, 2]
    vertices: torch.Tensor,  # [N, 2]
) -> torch.Tensor:
    """Check if points are inside polygon using ray casting."""
    M = points.shape[0]
    N = vertices.shape[0]

    # Count ray crossings (ray goes in +x direction)
    crossings = torch.zeros(M, device=points.device, dtype=torch.int32)

    for i in range(N):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % N]

        y0, y1 = p0[1], p1[1]
        x0, x1 = p0[0], p1[0]

        py = points[:, 1]
        px = points[:, 0]

        # Check if ray crosses this edge
        cond1 = ((y0 <= py) & (y1 > py)) | ((y1 <= py) & (y0 > py))

        # Compute x at intersection
        dy = y1 - y0
        dy = torch.where(torch.abs(dy) < 1e-10, torch.ones_like(dy) * 1e-10, dy)
        t = (py - y0) / dy
        x_intersect = x0 + t * (x1 - x0)

        cond2 = px < x_intersect

        crossings = crossings + (cond1 & cond2).int()

    # Odd crossings = inside
    return (crossings % 2) == 1


def _sample_color_vectorized(
    color: Color,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample color for all pixels.

    Args:
        color: Color to sample
        x_coords: X coordinates [W]
        y_coords: Y coordinates [H]
        device: Output device
        dtype: Output dtype

    Returns:
        Color values [H, W, 4]
    """
    H = len(y_coords)
    W = len(x_coords)

    if isinstance(color, SolidColor):
        return color.color.unsqueeze(0).unsqueeze(0).expand(H, W, -1)

    # Create coordinate grid
    yy, xx = torch.meshgrid(y_coords + 0.5, x_coords + 0.5, indexing='ij')

    if isinstance(color, LinearGradient):
        # Project onto gradient line
        begin = color.begin
        end = color.end
        direction = end - begin
        length_sq = (direction * direction).sum()

        if length_sq < 1e-10:
            return color.stop_colors[0].unsqueeze(0).unsqueeze(0).expand(H, W, -1)

        # Compute t for each pixel
        points = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        t = ((points - begin) * direction).sum(dim=-1) / length_sq
        t = torch.clamp(t, 0.0, 1.0)

        # Interpolate colors
        return _interpolate_gradient_vectorized(t, color.offsets, color.stop_colors)

    elif isinstance(color, RadialGradient):
        # Distance from center
        points = torch.stack([xx, yy], dim=-1)
        normalized = (points - color.center) / color.radius
        t = torch.norm(normalized, dim=-1)
        t = torch.clamp(t, 0.0, 1.0)

        return _interpolate_gradient_vectorized(t, color.offsets, color.stop_colors)

    else:
        return torch.zeros(H, W, 4, device=device, dtype=dtype)


def _interpolate_gradient_vectorized(
    t: torch.Tensor,  # [H, W]
    offsets: torch.Tensor,  # [S]
    stop_colors: torch.Tensor,  # [S, 4]
) -> torch.Tensor:
    """Interpolate gradient colors for all pixels."""
    H, W = t.shape
    S = offsets.shape[0]

    if S == 1:
        return stop_colors[0].unsqueeze(0).unsqueeze(0).expand(H, W, -1)

    # Find segments for each pixel
    result = torch.zeros(H, W, 4, device=t.device, dtype=t.dtype)

    for i in range(S - 1):
        t0, t1 = offsets[i], offsets[i + 1]
        c0, c1 = stop_colors[i], stop_colors[i + 1]

        # Pixels in this segment
        mask = (t >= t0) & (t <= t1)

        if t1 - t0 < 1e-10:
            result = torch.where(mask.unsqueeze(-1), c0, result)
        else:
            local_t = (t - t0) / (t1 - t0)
            interp = (1.0 - local_t).unsqueeze(-1) * c0 + local_t.unsqueeze(-1) * c1
            result = torch.where(mask.unsqueeze(-1), interp, result)

    # Handle t past last offset
    mask = t > offsets[-1]
    result = torch.where(mask.unsqueeze(-1), stop_colors[-1], result)

    return result


def _alpha_composite_vectorized(
    src: torch.Tensor,  # [H, W, 4]
    dst: torch.Tensor,  # [H, W, 4]
) -> torch.Tensor:
    """Alpha composite source over destination (vectorized)."""
    src_a = src[..., 3:4]
    dst_a = dst[..., 3:4]

    out_a = src_a + dst_a * (1.0 - src_a)

    # Avoid division by zero
    out_a_safe = torch.where(out_a > 0, out_a, torch.ones_like(out_a))

    out_rgb = (src[..., :3] * src_a + dst[..., :3] * dst_a * (1.0 - src_a)) / out_a_safe
    out_rgb = torch.where(out_a > 0, out_rgb, torch.zeros_like(out_rgb))

    return torch.cat([out_rgb, out_a], dim=-1)
