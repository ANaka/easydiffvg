"""SVG saving for pydiffvg."""

import torch

from pydiffvg.shapes import Shape, Circle, Ellipse, Rect, Polygon, Path
from pydiffvg.groups import ShapeGroup
from pydiffvg.color import Color, SolidColor, LinearGradient, RadialGradient


def save_svg(
    filename: str,
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
) -> None:
    """Save shapes to an SVG file.

    Args:
        filename: Output SVG file path
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        shapes: List of shape primitives
        shape_groups: List of ShapeGroup objects
    """
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">'
    )

    # Collect gradients for defs section
    gradients = []
    gradient_ids = {}

    for i, group in enumerate(shape_groups):
        if group.fill_color is not None and not isinstance(group.fill_color, SolidColor):
            grad_id = f"grad_fill_{i}"
            gradient_ids[id(group.fill_color)] = grad_id
            gradients.append((grad_id, group.fill_color))
        if group.stroke_color is not None and not isinstance(group.stroke_color, SolidColor):
            grad_id = f"grad_stroke_{i}"
            gradient_ids[id(group.stroke_color)] = grad_id
            gradients.append((grad_id, group.stroke_color))

    # Write defs if needed
    if gradients:
        lines.append("  <defs>")
        for grad_id, gradient in gradients:
            lines.append(_gradient_to_svg(grad_id, gradient))
        lines.append("  </defs>")

    # Write shapes
    for group in shape_groups:
        for shape_idx in group.shape_ids:
            shape = shapes[int(shape_idx.item())]

            fill_str = _color_to_svg(group.fill_color, gradient_ids)
            stroke_str = _color_to_svg(group.stroke_color, gradient_ids)
            transform_str = _transform_to_svg(group.shape_to_canvas)

            attrs = []
            if fill_str:
                attrs.append(f'fill="{fill_str}"')
            else:
                attrs.append('fill="none"')
            if stroke_str:
                attrs.append(f'stroke="{stroke_str}"')
                stroke_width = float(shape.stroke_width.item()) if hasattr(shape, "stroke_width") else 1.0
                attrs.append(f'stroke-width="{stroke_width}"')
            if transform_str:
                attrs.append(f'transform="{transform_str}"')

            attr_str = " ".join(attrs)

            if isinstance(shape, Circle):
                cx = float(shape.center[0].item())
                cy = float(shape.center[1].item())
                r = float(shape.radius.item())
                lines.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" {attr_str}/>')

            elif isinstance(shape, Ellipse):
                cx = float(shape.center[0].item())
                cy = float(shape.center[1].item())
                rx = float(shape.radius[0].item())
                ry = float(shape.radius[1].item())
                lines.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {attr_str}/>')

            elif isinstance(shape, Rect):
                x = float(shape.p_min[0].item())
                y = float(shape.p_min[1].item())
                w = float(shape.p_max[0].item()) - x
                h = float(shape.p_max[1].item()) - y
                lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" {attr_str}/>')

            elif isinstance(shape, Polygon):
                points_list = []
                for i in range(shape.points.shape[0]):
                    x = float(shape.points[i, 0].item())
                    y = float(shape.points[i, 1].item())
                    points_list.append(f"{x},{y}")
                points_str = " ".join(points_list)

                if shape.is_closed:
                    lines.append(f'  <polygon points="{points_str}" {attr_str}/>')
                else:
                    lines.append(f'  <polyline points="{points_str}" {attr_str}/>')

            elif isinstance(shape, Path):
                d = _path_to_d(shape)
                lines.append(f'  <path d="{d}" {attr_str}/>')

    lines.append("</svg>")

    with open(filename, "w") as f:
        f.write("\n".join(lines))


def _color_to_svg(color: Color | None, gradient_ids: dict) -> str | None:
    """Convert a color to SVG string."""
    if color is None:
        return None

    if isinstance(color, SolidColor):
        r = int(color.color[0].item() * 255)
        g = int(color.color[1].item() * 255)
        b = int(color.color[2].item() * 255)
        return f"rgb({r},{g},{b})"

    # Gradient reference
    if id(color) in gradient_ids:
        return f"url(#{gradient_ids[id(color)]})"

    return None


def _gradient_to_svg(grad_id: str, gradient: LinearGradient | RadialGradient) -> str:
    """Convert a gradient to SVG element string."""
    lines = []

    if isinstance(gradient, LinearGradient):
        x1 = float(gradient.begin[0].item())
        y1 = float(gradient.begin[1].item())
        x2 = float(gradient.end[0].item())
        y2 = float(gradient.end[1].item())
        lines.append(
            f'    <linearGradient id="{grad_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'gradientUnits="userSpaceOnUse">'
        )
    else:  # RadialGradient
        cx = float(gradient.center[0].item())
        cy = float(gradient.center[1].item())
        r = float(gradient.radius[0].item())  # Use rx as radius
        lines.append(
            f'    <radialGradient id="{grad_id}" cx="{cx}" cy="{cy}" r="{r}" '
            f'gradientUnits="userSpaceOnUse">'
        )

    # Add stops
    for i in range(gradient.offsets.shape[0]):
        offset = float(gradient.offsets[i].item())
        r = int(gradient.stop_colors[i, 0].item() * 255)
        g = int(gradient.stop_colors[i, 1].item() * 255)
        b = int(gradient.stop_colors[i, 2].item() * 255)
        a = float(gradient.stop_colors[i, 3].item())
        lines.append(
            f'      <stop offset="{offset}" stop-color="rgb({r},{g},{b})" stop-opacity="{a}"/>'
        )

    if isinstance(gradient, LinearGradient):
        lines.append("    </linearGradient>")
    else:
        lines.append("    </radialGradient>")

    return "\n".join(lines)


def _transform_to_svg(transform: torch.Tensor) -> str | None:
    """Convert a 3x3 transform matrix to SVG transform string."""
    # Check if identity
    identity = torch.eye(3, device=transform.device, dtype=transform.dtype)
    if torch.allclose(transform, identity, atol=1e-6):
        return None

    # Convert to SVG matrix(a, b, c, d, e, f) format
    # SVG matrix is: [a c e; b d f; 0 0 1]
    # Our matrix is: [a c e; b d f; 0 0 1] (same layout)
    a = float(transform[0, 0].item())
    b = float(transform[1, 0].item())
    c = float(transform[0, 1].item())
    d = float(transform[1, 1].item())
    e = float(transform[0, 2].item())
    f = float(transform[1, 2].item())

    return f"matrix({a},{b},{c},{d},{e},{f})"


def _path_to_d(path: Path) -> str:
    """Convert a Path shape to SVG path d attribute."""
    parts = []
    points = path.points
    num_control = path.num_control_points

    if len(points) == 0:
        return ""

    # Move to first point
    x = float(points[0, 0].item())
    y = float(points[0, 1].item())
    parts.append(f"M {x} {y}")

    idx = 1
    for i in range(len(num_control)):
        n_ctrl = int(num_control[i].item())

        if n_ctrl == 0:
            # Line segment
            if idx < len(points):
                x = float(points[idx, 0].item())
                y = float(points[idx, 1].item())
                parts.append(f"L {x} {y}")
                idx += 1

        elif n_ctrl == 1:
            # Quadratic bezier
            if idx + 1 < len(points):
                x1 = float(points[idx, 0].item())
                y1 = float(points[idx, 1].item())
                x = float(points[idx + 1, 0].item())
                y = float(points[idx + 1, 1].item())
                parts.append(f"Q {x1} {y1} {x} {y}")
                idx += 2

        elif n_ctrl == 2:
            # Cubic bezier
            if idx + 2 < len(points):
                x1 = float(points[idx, 0].item())
                y1 = float(points[idx, 1].item())
                x2 = float(points[idx + 1, 0].item())
                y2 = float(points[idx + 1, 1].item())
                x = float(points[idx + 2, 0].item())
                y = float(points[idx + 2, 1].item())
                parts.append(f"C {x1} {y1} {x2} {y2} {x} {y}")
                idx += 3

    if path.is_closed:
        parts.append("Z")

    return " ".join(parts)
