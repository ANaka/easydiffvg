"""Boundary sampling for gradient computation.

This module implements the core of differentiable rendering - computing gradients
of the rendered image with respect to shape parameters. It uses boundary sampling
and the Reynolds transport theorem.

The key insight is that most pixels' values change smoothly with shape parameters,
which PyTorch's autograd handles. But at shape boundaries, there's a discontinuity
that standard autodiff cannot handle. Boundary sampling explicitly computes how
moving the boundary affects pixel coverage.
"""

import math

import torch

from pydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect, Shape


def compute_boundary_samples(
    shape: Shape,
    num_samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points along shape boundary with outward normals.

    Args:
        shape: Shape to sample
        num_samples: Number of boundary samples

    Returns:
        Tuple of (samples [N, 2], normals [N, 2])
    """
    if isinstance(shape, Circle):
        return _sample_circle_boundary(shape, num_samples)
    elif isinstance(shape, Ellipse):
        return _sample_ellipse_boundary(shape, num_samples)
    elif isinstance(shape, Rect):
        return _sample_rect_boundary(shape, num_samples)
    elif isinstance(shape, Polygon):
        return _sample_polygon_boundary(shape, num_samples)
    elif isinstance(shape, Path):
        return _sample_path_boundary(shape, num_samples)
    else:
        raise TypeError(f"Unknown shape type: {type(shape)}")


def _sample_circle_boundary(
    circle: Circle,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample circle boundary."""
    device = circle.center.device
    dtype = circle.center.dtype
    angles = torch.linspace(0, 2 * math.pi, num_samples + 1, device=device, dtype=dtype)[:-1]

    # Points on boundary
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    samples = circle.center + circle.radius * torch.stack([cos_a, sin_a], dim=1)

    # Normals (outward)
    normals = torch.stack([cos_a, sin_a], dim=1)

    return samples, normals


def _sample_ellipse_boundary(
    ellipse: Ellipse,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ellipse boundary."""
    device = ellipse.center.device
    dtype = ellipse.center.dtype
    angles = torch.linspace(0, 2 * math.pi, num_samples + 1, device=device, dtype=dtype)[:-1]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Points on boundary
    samples = ellipse.center + torch.stack([
        ellipse.radius[0] * cos_a,
        ellipse.radius[1] * sin_a,
    ], dim=1)

    # Normals (need to account for ellipse scaling)
    # Gradient of implicit function (x/rx)^2 + (y/ry)^2 = 1
    nx = cos_a / ellipse.radius[0]
    ny = sin_a / ellipse.radius[1]
    normals = torch.stack([nx, ny], dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return samples, normals


def _sample_rect_boundary(
    rect: Rect,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rectangle boundary."""
    device = rect.p_min.device
    dtype = rect.p_min.dtype

    # Distribute samples proportional to edge length
    w = rect.p_max[0] - rect.p_min[0]
    h = rect.p_max[1] - rect.p_min[1]
    perimeter = 2 * (w + h)

    if perimeter < 1e-10:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)

    samples_per_edge = [
        max(1, int(num_samples * w / perimeter)),
        max(1, int(num_samples * h / perimeter)),
        max(1, int(num_samples * w / perimeter)),
        max(1, int(num_samples * h / perimeter)),
    ]

    samples_list = []
    normals_list = []

    # Bottom edge
    for i in range(samples_per_edge[0]):
        t = i / max(1, samples_per_edge[0])
        x = rect.p_min[0] + t * w
        samples_list.append(torch.stack([x, rect.p_min[1]]))
        normals_list.append(torch.tensor([0.0, -1.0], device=device, dtype=dtype))

    # Right edge
    for i in range(samples_per_edge[1]):
        t = i / max(1, samples_per_edge[1])
        y = rect.p_min[1] + t * h
        samples_list.append(torch.stack([rect.p_max[0], y]))
        normals_list.append(torch.tensor([1.0, 0.0], device=device, dtype=dtype))

    # Top edge
    for i in range(samples_per_edge[2]):
        t = i / max(1, samples_per_edge[2])
        x = rect.p_max[0] - t * w
        samples_list.append(torch.stack([x, rect.p_max[1]]))
        normals_list.append(torch.tensor([0.0, 1.0], device=device, dtype=dtype))

    # Left edge
    for i in range(samples_per_edge[3]):
        t = i / max(1, samples_per_edge[3])
        y = rect.p_max[1] - t * h
        samples_list.append(torch.stack([rect.p_min[0], y]))
        normals_list.append(torch.tensor([-1.0, 0.0], device=device, dtype=dtype))

    if len(samples_list) == 0:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)

    return torch.stack(samples_list), torch.stack(normals_list)


def _sample_polygon_boundary(
    polygon: Polygon,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample polygon boundary."""
    return _sample_polyline_boundary(polygon.points, polygon.is_closed, num_samples)


def _sample_path_boundary(
    path: Path,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample path boundary (simplified: treat as polyline)."""
    return _sample_polyline_boundary(path.points, path.is_closed, num_samples)


def _sample_polyline_boundary(
    points: torch.Tensor,
    is_closed: bool,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample polyline boundary."""
    device = points.device
    dtype = points.dtype
    n = len(points)

    if n < 2:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)

    num_edges = n if is_closed else n - 1

    # Compute edge lengths
    edge_lengths = []
    for i in range(num_edges):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        edge_lengths.append(torch.norm(p1 - p0))

    total_length = sum(edge_lengths)

    if total_length < 1e-10:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)

    samples_list = []
    normals_list = []

    for i in range(num_edges):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        edge_samples = max(1, int(num_samples * edge_lengths[i] / total_length))

        direction = p1 - p0
        length = torch.norm(direction)
        if length < 1e-10:
            continue

        tangent = direction / length
        normal = torch.stack([-tangent[1], tangent[0]])  # Perpendicular (CCW rotation)

        for j in range(edge_samples):
            t = j / max(1, edge_samples)
            samples_list.append(p0 + t * direction)
            normals_list.append(normal)

    if len(samples_list) == 0:
        return torch.zeros(0, 2, device=device, dtype=dtype), torch.zeros(0, 2, device=device, dtype=dtype)

    return torch.stack(samples_list), torch.stack(normals_list)


def boundary_gradient_circle(
    grad_output: torch.Tensor,
    circle: Circle,
    width: int,
    height: int,
    fill_color: torch.Tensor,
    num_boundary_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of loss w.r.t. circle parameters.

    Uses boundary sampling / Reynolds transport theorem.

    Args:
        grad_output: [H, W, 4] gradient of loss w.r.t. image
        circle: Circle shape
        width, height: Canvas dimensions
        fill_color: [4] RGBA fill color
        num_boundary_samples: Number of boundary samples

    Returns:
        Tuple of (grad_center [2], grad_radius [])
    """
    samples, normals = compute_boundary_samples(circle, num_boundary_samples)

    device = circle.center.device
    dtype = circle.center.dtype

    # For each boundary sample, compute contribution to gradient
    grad_center = torch.zeros(2, device=device, dtype=dtype)
    grad_radius = torch.zeros((), device=device, dtype=dtype)

    for i in range(len(samples)):
        sample = samples[i]
        normal = normals[i]

        # Get pixel coordinates (with bounds check)
        px = int(sample[0].item())
        py = int(sample[1].item())

        if 0 <= px < width and 0 <= py < height:
            # Gradient contribution from this boundary point
            # The color that would be revealed/hidden by boundary movement
            pixel_grad = grad_output[py, px]

            # Compute color difference (filled vs background)
            # Assuming background is transparent black [0,0,0,0]
            color_diff = fill_color

            # Boundary integral contribution
            # d/d(param) = integral over boundary of (color_diff * normal_component * d/d(param)(boundary))

            # For center: moving center moves boundary by same amount (in opposite direction)
            # For radius: moving radius expands boundary outward along normal

            contrib = (pixel_grad * color_diff).sum()

            # Center gradient: boundary moves opposite to center movement
            grad_center -= contrib * normal

            # Radius gradient: boundary moves outward with radius increase
            grad_radius += contrib

    # Scale by arc length per sample (perimeter / num_samples)
    perimeter = 2 * math.pi * float(circle.radius.item())
    arc_length = perimeter / num_boundary_samples

    grad_center = grad_center * arc_length
    grad_radius = grad_radius * arc_length

    return grad_center, grad_radius


def boundary_gradient_rect(
    grad_output: torch.Tensor,
    rect: Rect,
    width: int,
    height: int,
    fill_color: torch.Tensor,
    num_boundary_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of loss w.r.t. rectangle parameters.

    Args:
        grad_output: [H, W, 4] gradient of loss w.r.t. image
        rect: Rect shape
        width, height: Canvas dimensions
        fill_color: [4] RGBA fill color
        num_boundary_samples: Number of boundary samples

    Returns:
        Tuple of (grad_p_min [2], grad_p_max [2])
    """
    samples, normals = compute_boundary_samples(rect, num_boundary_samples)

    device = rect.p_min.device
    dtype = rect.p_min.dtype

    grad_p_min = torch.zeros(2, device=device, dtype=dtype)
    grad_p_max = torch.zeros(2, device=device, dtype=dtype)

    w = rect.p_max[0] - rect.p_min[0]
    h = rect.p_max[1] - rect.p_min[1]
    perimeter = 2 * (w + h)

    if perimeter < 1e-10 or len(samples) == 0:
        return grad_p_min, grad_p_max

    arc_length = perimeter / len(samples)

    for i in range(len(samples)):
        sample = samples[i]
        normal = normals[i]

        px = int(sample[0].item())
        py = int(sample[1].item())

        if 0 <= px < width and 0 <= py < height:
            pixel_grad = grad_output[py, px]
            contrib = (pixel_grad * fill_color).sum()

            # Determine which edge this sample is on
            eps = 1e-6
            on_left = abs(sample[0] - rect.p_min[0]) < eps
            on_right = abs(sample[0] - rect.p_max[0]) < eps
            on_bottom = abs(sample[1] - rect.p_min[1]) < eps
            on_top = abs(sample[1] - rect.p_max[1]) < eps

            # p_min controls left and bottom edges (moving outward)
            # p_max controls right and top edges (moving outward)
            if on_left:
                grad_p_min[0] += contrib * arc_length  # Moving left expands shape
            if on_bottom:
                grad_p_min[1] += contrib * arc_length
            if on_right:
                grad_p_max[0] += contrib * arc_length
            if on_top:
                grad_p_max[1] += contrib * arc_length

    return grad_p_min, grad_p_max


def boundary_gradient_ellipse(
    grad_output: torch.Tensor,
    ellipse: Ellipse,
    width: int,
    height: int,
    fill_color: torch.Tensor,
    num_boundary_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of loss w.r.t. ellipse parameters.

    Args:
        grad_output: [H, W, 4] gradient of loss w.r.t. image
        ellipse: Ellipse shape
        width, height: Canvas dimensions
        fill_color: [4] RGBA fill color
        num_boundary_samples: Number of boundary samples

    Returns:
        Tuple of (grad_center [2], grad_radius [2])
    """
    samples, normals = compute_boundary_samples(ellipse, num_boundary_samples)

    device = ellipse.center.device
    dtype = ellipse.center.dtype

    grad_center = torch.zeros(2, device=device, dtype=dtype)
    grad_radius = torch.zeros(2, device=device, dtype=dtype)

    # Approximate perimeter using Ramanujan's formula
    a, b = float(ellipse.radius[0].item()), float(ellipse.radius[1].item())
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    arc_length = perimeter / num_boundary_samples

    angles = torch.linspace(0, 2 * math.pi, num_boundary_samples + 1, device=device, dtype=dtype)[:-1]

    for i in range(len(samples)):
        sample = samples[i]
        normal = normals[i]

        px = int(sample[0].item())
        py = int(sample[1].item())

        if 0 <= px < width and 0 <= py < height:
            pixel_grad = grad_output[py, px]
            contrib = (pixel_grad * fill_color).sum()

            # Center gradient
            grad_center -= contrib * normal * arc_length

            # Radius gradient: how boundary moves with radius change
            # At angle theta, point is center + (rx*cos, ry*sin)
            # d/d(rx) = (cos(theta), 0), d/d(ry) = (0, sin(theta))
            cos_a = torch.cos(angles[i])
            sin_a = torch.sin(angles[i])

            # Dot product with normal gives expansion along normal
            grad_radius[0] += contrib * (cos_a * normal[0]) * arc_length
            grad_radius[1] += contrib * (sin_a * normal[1]) * arc_length

    return grad_center, grad_radius


def compute_shape_gradients(
    grad_output: torch.Tensor,
    shapes: list[Shape],
    shape_groups: list,
    width: int,
    height: int,
    num_boundary_samples: int = 64,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute gradients for all shapes.

    Args:
        grad_output: [H, W, 4] gradient of loss w.r.t. image
        shapes: List of shapes
        shape_groups: List of ShapeGroups
        width, height: Canvas dimensions
        num_boundary_samples: Number of boundary samples per shape

    Returns:
        Dictionary mapping shape index to parameter gradients
    """
    grads: dict[int, dict[str, torch.Tensor]] = {}

    # Map shapes to their fill colors
    shape_colors: dict[int, torch.Tensor] = {}
    for group in shape_groups:
        if group.fill_color is not None:
            from pydiffvg.color import SolidColor
            if isinstance(group.fill_color, SolidColor):
                color = group.fill_color.color
            else:
                # For gradients, use average color as approximation
                color = group.fill_color.stop_colors.mean(dim=0)

            for shape_idx in group.shape_ids:
                shape_colors[int(shape_idx.item())] = color

    for idx, shape in enumerate(shapes):
        fill_color = shape_colors.get(idx, torch.tensor([0., 0., 0., 0.], device=grad_output.device))

        if isinstance(shape, Circle):
            grad_center, grad_radius = boundary_gradient_circle(
                grad_output, shape, width, height, fill_color, num_boundary_samples
            )
            grads[idx] = {"center": grad_center, "radius": grad_radius}

        elif isinstance(shape, Ellipse):
            grad_center, grad_radius = boundary_gradient_ellipse(
                grad_output, shape, width, height, fill_color, num_boundary_samples
            )
            grads[idx] = {"center": grad_center, "radius": grad_radius}

        elif isinstance(shape, Rect):
            grad_p_min, grad_p_max = boundary_gradient_rect(
                grad_output, shape, width, height, fill_color, num_boundary_samples
            )
            grads[idx] = {"p_min": grad_p_min, "p_max": grad_p_max}

        # TODO: Add polygon and path gradient computation

    return grads
