"""Core rasterization logic for pydiffvg."""

from enum import Enum

import torch

from pydiffvg.shapes import Shape, Circle, Ellipse, Rect, Polygon, Path
from pydiffvg.groups import ShapeGroup
from pydiffvg.color import Color, SolidColor, LinearGradient, RadialGradient
from pydiffvg.bvh import build_bvh, query_bvh, compute_shape_bbox
from pydiffvg.utils.distance import (
    signed_distance_circle,
    signed_distance_ellipse,
    signed_distance_rect,
    distance_to_line_segment,
    distance_to_quadratic_bezier,
    distance_to_cubic_bezier,
)
from pydiffvg.utils.winding import (
    winding_number_line,
    winding_number_quadratic,
    winding_number_cubic,
    winding_number_polygon,
)
from pydiffvg.utils.bezier import evaluate_quadratic, evaluate_cubic


class PixelFilter(Enum):
    """Pixel filter for antialiasing."""

    BOX = "box"
    TENT = "tent"
    GAUSSIAN = "gaussian"


def sample_color(
    color: Color, point: torch.Tensor, shape_to_canvas: torch.Tensor
) -> torch.Tensor:
    """Sample a color at a given point.

    Args:
        color: Color to sample
        point: Point in canvas coordinates [2]
        shape_to_canvas: Transform matrix [3, 3]

    Returns:
        RGBA color [4]
    """
    if isinstance(color, SolidColor):
        return color.color

    elif isinstance(color, LinearGradient):
        # Project point onto gradient line
        begin = color.begin
        end = color.end
        direction = end - begin
        length_sq = torch.dot(direction, direction)

        if length_sq < 1e-10:
            # Degenerate gradient
            return color.stop_colors[0]

        t = torch.dot(point - begin, direction) / length_sq
        t = torch.clamp(t, 0.0, 1.0)

        # Interpolate between stops
        return _interpolate_gradient(t, color.offsets, color.stop_colors)

    elif isinstance(color, RadialGradient):
        # Compute normalized distance from center
        diff = point - color.center
        # Scale by radii for elliptical gradient
        normalized = diff / color.radius
        t = torch.norm(normalized)
        t = torch.clamp(t, 0.0, 1.0)

        return _interpolate_gradient(t, color.offsets, color.stop_colors)

    else:
        raise ValueError(f"Unknown color type: {type(color)}")


def _interpolate_gradient(
    t: torch.Tensor, offsets: torch.Tensor, stop_colors: torch.Tensor
) -> torch.Tensor:
    """Interpolate gradient colors.

    Args:
        t: Parameter in [0, 1]
        offsets: Stop offsets [S]
        stop_colors: Stop colors [S, 4]

    Returns:
        Interpolated color [4]
    """
    # Find which segment t falls in
    n_stops = offsets.shape[0]

    if n_stops == 1:
        return stop_colors[0]

    # Find the two stops bracketing t
    for i in range(n_stops - 1):
        if t <= offsets[i + 1]:
            # t is between offsets[i] and offsets[i+1]
            t0, t1 = offsets[i], offsets[i + 1]
            c0, c1 = stop_colors[i], stop_colors[i + 1]

            if t1 - t0 < 1e-10:
                return c0

            local_t = (t - t0) / (t1 - t0)
            return (1.0 - local_t) * c0 + local_t * c1

    # t is past the last stop
    return stop_colors[-1]


def compute_coverage(
    shape: Shape,
    point: torch.Tensor,
    shape_to_canvas: torch.Tensor,
    use_even_odd: bool = True,
) -> torch.Tensor:
    """Compute coverage of a shape at a point for fill.

    Args:
        shape: Shape to test
        point: Query point in canvas coordinates [2]
        shape_to_canvas: Transform matrix [3, 3]
        use_even_odd: Use even-odd fill rule (vs winding number)

    Returns:
        Coverage value in [0, 1]
    """
    # Transform point to shape coordinates
    canvas_to_shape = torch.linalg.inv(shape_to_canvas)
    point_h = torch.cat([point, torch.ones(1, device=point.device, dtype=point.dtype)])
    point_shape = (canvas_to_shape @ point_h)[:2]

    if isinstance(shape, Circle):
        sdf = signed_distance_circle(point_shape, shape.center, shape.radius)
        # Antialiasing: smooth step at boundary
        return _smooth_coverage(sdf)

    elif isinstance(shape, Ellipse):
        sdf = signed_distance_ellipse(point_shape, shape.center, shape.radius)
        return _smooth_coverage(sdf)

    elif isinstance(shape, Rect):
        sdf = signed_distance_rect(point_shape, shape.p_min, shape.p_max)
        return _smooth_coverage(sdf)

    elif isinstance(shape, Polygon):
        if shape.is_closed:
            winding = winding_number_polygon(point_shape, shape.points)
            if use_even_odd:
                # Even-odd: inside if winding is odd
                return (torch.abs(winding) % 2).float()
            else:
                # Non-zero: inside if winding != 0
                return (winding != 0).float()
        else:
            # Open polygon has no fill
            return torch.tensor(0.0, device=point.device, dtype=point.dtype)

    elif isinstance(shape, Path):
        if shape.is_closed:
            winding = _compute_path_winding(point_shape, shape)
            if use_even_odd:
                return (torch.abs(winding) % 2).float()
            else:
                return (winding != 0).float()
        else:
            return torch.tensor(0.0, device=point.device, dtype=point.dtype)

    else:
        raise ValueError(f"Unknown shape type: {type(shape)}")


def _smooth_coverage(sdf: torch.Tensor, pixel_width: float = 1.0) -> torch.Tensor:
    """Convert signed distance to smooth coverage.

    Uses a smooth step function for antialiasing.
    """
    # Smoothstep from -0.5 to 0.5 pixels
    half_width = pixel_width * 0.5
    t = torch.clamp((sdf + half_width) / pixel_width, 0.0, 1.0)
    # Hermite interpolation for smoothstep: 1 - (3t^2 - 2t^3)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


def _compute_path_winding(point: torch.Tensor, path: Path) -> torch.Tensor:
    """Compute winding number for a point with respect to a path."""
    winding = torch.zeros((), dtype=point.dtype, device=point.device)

    points = path.points
    num_control = path.num_control_points
    n_segments = len(num_control)

    idx = 0
    for i in range(n_segments):
        n_ctrl = int(num_control[i].item())

        if n_ctrl == 0:
            # Line segment
            p0 = points[idx]
            p1 = points[idx + 1] if idx + 1 < len(points) else points[0]
            winding = winding + winding_number_line(point, p0, p1)
            idx += 1

        elif n_ctrl == 1:
            # Quadratic bezier
            p0 = points[idx]
            p1 = points[idx + 1]
            p2 = points[idx + 2] if idx + 2 < len(points) else points[0]
            winding = winding + winding_number_quadratic(point, p0, p1, p2)
            idx += 2

        elif n_ctrl == 2:
            # Cubic bezier
            p0 = points[idx]
            p1 = points[idx + 1]
            p2 = points[idx + 2]
            p3 = points[idx + 3] if idx + 3 < len(points) else points[0]
            winding = winding + winding_number_cubic(point, p0, p1, p2, p3)
            idx += 3

    # Handle closing segment if needed
    if path.is_closed and idx < len(points):
        winding = winding + winding_number_line(point, points[idx], points[0])

    return winding


def compute_stroke_coverage(
    shape: Shape,
    point: torch.Tensor,
    shape_to_canvas: torch.Tensor,
    stroke_width: torch.Tensor,
) -> torch.Tensor:
    """Compute coverage of a shape's stroke at a point.

    Args:
        shape: Shape to test
        point: Query point in canvas coordinates [2]
        shape_to_canvas: Transform matrix [3, 3]
        stroke_width: Width of the stroke

    Returns:
        Coverage value in [0, 1]
    """
    # Transform point to shape coordinates
    canvas_to_shape = torch.linalg.inv(shape_to_canvas)
    point_h = torch.cat([point, torch.ones(1, device=point.device, dtype=point.dtype)])
    point_shape = (canvas_to_shape @ point_h)[:2]

    half_width = stroke_width / 2.0

    if isinstance(shape, Circle):
        dist = torch.abs(
            torch.norm(point_shape - shape.center) - shape.radius
        )
        return _smooth_coverage(dist - half_width)

    elif isinstance(shape, Ellipse):
        # Approximate ellipse stroke
        normalized = (point_shape - shape.center) / shape.radius
        dist_normalized = torch.norm(normalized)
        dist = torch.abs(dist_normalized - 1.0) * (shape.radius[0] + shape.radius[1]) / 2.0
        return _smooth_coverage(dist - half_width)

    elif isinstance(shape, Rect):
        # Distance to rectangle boundary
        dist = _distance_to_rect_boundary(point_shape, shape.p_min, shape.p_max)
        return _smooth_coverage(dist - half_width)

    elif isinstance(shape, Polygon):
        # Distance to polygon edges
        dist = _distance_to_polygon_edges(point_shape, shape.points, shape.is_closed)
        return _smooth_coverage(dist - half_width)

    elif isinstance(shape, Path):
        # Distance to path curves
        dist = _distance_to_path(point_shape, shape)
        return _smooth_coverage(dist - half_width)

    else:
        raise ValueError(f"Unknown shape type: {type(shape)}")


def _distance_to_rect_boundary(
    point: torch.Tensor, p_min: torch.Tensor, p_max: torch.Tensor
) -> torch.Tensor:
    """Compute distance from point to rectangle boundary."""
    # Rectangle edges
    edges = [
        (torch.stack([p_min[0], p_min[1]]), torch.stack([p_max[0], p_min[1]])),  # bottom
        (torch.stack([p_max[0], p_min[1]]), torch.stack([p_max[0], p_max[1]])),  # right
        (torch.stack([p_max[0], p_max[1]]), torch.stack([p_min[0], p_max[1]])),  # top
        (torch.stack([p_min[0], p_max[1]]), torch.stack([p_min[0], p_min[1]])),  # left
    ]

    min_dist = torch.tensor(float("inf"), device=point.device, dtype=point.dtype)
    for e0, e1 in edges:
        dist = distance_to_line_segment(point, e0, e1)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist


def _distance_to_polygon_edges(
    point: torch.Tensor, vertices: torch.Tensor, is_closed: bool
) -> torch.Tensor:
    """Compute distance from point to polygon edges."""
    n = vertices.shape[0]
    min_dist = torch.tensor(float("inf"), device=point.device, dtype=point.dtype)

    n_edges = n if is_closed else n - 1
    for i in range(n_edges):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        dist = distance_to_line_segment(point, p0, p1)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist


def _distance_to_path(point: torch.Tensor, path: Path) -> torch.Tensor:
    """Compute distance from point to path curves."""
    min_dist = torch.tensor(float("inf"), device=point.device, dtype=point.dtype)

    points = path.points
    num_control = path.num_control_points

    idx = 0
    for i in range(len(num_control)):
        n_ctrl = int(num_control[i].item())

        if n_ctrl == 0:
            # Line segment
            p0 = points[idx]
            p1 = points[idx + 1] if idx + 1 < len(points) else points[0]
            dist = distance_to_line_segment(point, p0, p1)
            min_dist = torch.minimum(min_dist, dist)
            idx += 1

        elif n_ctrl == 1:
            # Quadratic bezier
            p0 = points[idx]
            p1 = points[idx + 1]
            p2 = points[idx + 2] if idx + 2 < len(points) else points[0]
            dist = distance_to_quadratic_bezier(point, p0, p1, p2)
            min_dist = torch.minimum(min_dist, dist)
            idx += 2

        elif n_ctrl == 2:
            # Cubic bezier
            p0 = points[idx]
            p1 = points[idx + 1]
            p2 = points[idx + 2]
            p3 = points[idx + 3] if idx + 3 < len(points) else points[0]
            dist = distance_to_cubic_bezier(point, p0, p1, p2, p3)
            min_dist = torch.minimum(min_dist, dist)
            idx += 3

    return min_dist


def alpha_composite(
    src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    """Alpha composite source over destination.

    Args:
        src: Source color [4] RGBA with premultiplied alpha
        dst: Destination color [4] RGBA with premultiplied alpha

    Returns:
        Composited color [4]
    """
    src_a = src[3]
    dst_a = dst[3]

    out_a = src_a + dst_a * (1.0 - src_a)

    # Avoid division by zero
    out_a_safe = torch.where(out_a > 0, out_a, torch.ones_like(out_a))

    out_rgb = (src[:3] * src_a + dst[:3] * dst_a * (1.0 - src_a)) / out_a_safe
    out_rgb = torch.where(out_a > 0, out_rgb, torch.zeros_like(out_rgb))

    return torch.cat([out_rgb, out_a.unsqueeze(0)])


def _get_device_dtype(shapes: list[Shape]) -> tuple[torch.device, torch.dtype]:
    """Get device and dtype from shapes list."""
    if len(shapes) == 0:
        return torch.device("cpu"), torch.float32
    first_shape = shapes[0]
    if isinstance(first_shape, (Circle, Ellipse)):
        return first_shape.center.device, first_shape.center.dtype
    elif isinstance(first_shape, Rect):
        return first_shape.p_min.device, first_shape.p_min.dtype
    else:  # Polygon or Path
        return first_shape.points.device, first_shape.points.dtype


def rasterize_pixel(
    x: int,
    y: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
    bvh_nodes: list,
    num_samples: int = 2,
) -> torch.Tensor:
    """Rasterize a single pixel.

    Args:
        x, y: Pixel coordinates
        shapes: List of shapes
        shape_groups: List of shape groups
        bvh_nodes: BVH for acceleration
        num_samples: Antialiasing samples per dimension

    Returns:
        RGBA color [4]
    """
    device, dtype = _get_device_dtype(shapes)

    # Accumulate samples
    total_color = torch.zeros(4, device=device, dtype=dtype)

    # Generate sample positions within pixel
    for sx in range(num_samples):
        for sy in range(num_samples):
            # Stratified sampling
            sample_x = x + (sx + 0.5) / num_samples
            sample_y = y + (sy + 0.5) / num_samples
            point = torch.tensor([sample_x, sample_y], device=device, dtype=dtype)

            # Start with transparent black
            pixel_color = torch.zeros(4, device=device, dtype=dtype)

            # Process groups in order (back to front)
            for group in shape_groups:
                # Process each shape in the group
                for shape_idx_tensor in group.shape_ids:
                    shape_idx = int(shape_idx_tensor.item())
                    shape = shapes[shape_idx]

                    # Fill
                    if group.fill_color is not None:
                        coverage = compute_coverage(
                            shape,
                            point,
                            group.shape_to_canvas,
                            group.use_even_odd_rule,
                        )

                        if coverage > 0:
                            fill = sample_color(
                                group.fill_color, point, group.shape_to_canvas
                            )
                            # Apply coverage to alpha
                            fill = fill.clone()
                            fill[3] = fill[3] * coverage
                            pixel_color = alpha_composite(fill, pixel_color)

                    # Stroke
                    if group.stroke_color is not None:
                        stroke_width = shape.stroke_width
                        coverage = compute_stroke_coverage(
                            shape, point, group.shape_to_canvas, stroke_width
                        )

                        if coverage > 0:
                            stroke = sample_color(
                                group.stroke_color, point, group.shape_to_canvas
                            )
                            stroke = stroke.clone()
                            stroke[3] = stroke[3] * coverage
                            pixel_color = alpha_composite(stroke, pixel_color)

            total_color = total_color + pixel_color

    # Average samples
    return total_color / (num_samples * num_samples)


def rasterize(
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rasterize shapes to an image.

    Args:
        canvas_width: Output image width
        canvas_height: Output image height
        shapes: List of shapes
        shape_groups: List of shape groups
        num_samples_x: Antialiasing samples in x
        num_samples_y: Antialiasing samples in y
        background: Background color [4] (default: transparent black)

    Returns:
        Image tensor [H, W, 4] RGBA
    """
    device, dtype = _get_device_dtype(shapes)

    # Build BVH for acceleration (not used in simple version but available)
    bvh_nodes = build_bvh(shapes)

    # Initialize output image
    if background is None:
        image = torch.zeros(canvas_height, canvas_width, 4, device=device, dtype=dtype)
    else:
        image = background.unsqueeze(0).unsqueeze(0).expand(canvas_height, canvas_width, -1).clone()

    # Rasterize each pixel
    for y in range(canvas_height):
        for x in range(canvas_width):
            image[y, x] = rasterize_pixel(
                x, y, shapes, shape_groups, bvh_nodes, num_samples_x
            )

    return image
