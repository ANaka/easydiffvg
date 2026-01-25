"""SVG parsing for easydiffvg."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from easydiffvg.shapes import Circle, Ellipse, Rect, Polygon, Path as PathShape
from easydiffvg.groups import ShapeGroup
from easydiffvg.color import SolidColor, LinearGradient, RadialGradient


def parse_svg(
    filename: str,
    device: torch.device | str = "cpu",
) -> tuple[int, int, list, list]:
    """Parse an SVG file into easydiffvg primitives.

    Args:
        filename: Path to SVG file
        device: Torch device for tensors

    Returns:
        Tuple of (canvas_width, canvas_height, shapes, shape_groups)
    """
    if isinstance(device, str):
        device = torch.device(device)

    tree = ET.parse(filename)
    root = tree.getroot()

    # Extract namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Get canvas dimensions
    width = _parse_length(root.get("width", "100"))
    height = _parse_length(root.get("height", "100"))

    # Also check viewBox
    viewbox = root.get("viewBox")
    if viewbox:
        parts = viewbox.split()
        if len(parts) >= 4:
            width = float(parts[2])
            height = float(parts[3])

    # Parse definitions (gradients, etc.)
    defs = {}
    for defs_elem in root.iter("{http://www.w3.org/2000/svg}defs"):
        for child in defs_elem:
            elem_id = child.get("id")
            if elem_id:
                defs[elem_id] = child

    # Also check for defs without namespace
    for defs_elem in root.iter("defs"):
        for child in defs_elem:
            elem_id = child.get("id")
            if elem_id:
                defs[elem_id] = child

    shapes: list = []
    shape_groups: list = []

    # Parse shapes
    _parse_element(root, shapes, shape_groups, defs, device)

    return int(width), int(height), shapes, shape_groups


def _parse_length(value: str) -> float:
    """Parse an SVG length value."""
    if not value:
        return 0.0
    # Remove units
    value = re.sub(r"[a-zA-Z%]+$", "", value.strip())
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_color(
    color_str: str | None, opacity: float = 1.0, device: torch.device = torch.device("cpu")
) -> SolidColor | None:
    """Parse a CSS color string to SolidColor."""
    if color_str is None or color_str.lower() == "none":
        return None

    color_str = color_str.strip()

    # Handle url() references - return None for now (handled separately)
    if color_str.startswith("url("):
        return None

    # Parse hex colors
    if color_str.startswith("#"):
        hex_str = color_str[1:]
        if len(hex_str) == 3:
            r = int(hex_str[0] * 2, 16) / 255.0
            g = int(hex_str[1] * 2, 16) / 255.0
            b = int(hex_str[2] * 2, 16) / 255.0
        elif len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
        else:
            r, g, b = 0.0, 0.0, 0.0

        return SolidColor(
            color=torch.tensor([r, g, b, opacity], device=device)
        )

    # Parse rgb()
    if color_str.startswith("rgb("):
        match = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
        if match:
            r = int(match.group(1)) / 255.0
            g = int(match.group(2)) / 255.0
            b = int(match.group(3)) / 255.0
            return SolidColor(
                color=torch.tensor([r, g, b, opacity], device=device)
            )

    # Parse named colors (common ones)
    named_colors = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (1, 0, 0),
        "green": (0, 0.5, 0),
        "blue": (0, 0, 1),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "orange": (1, 0.647, 0),
        "purple": (0.5, 0, 0.5),
        "pink": (1, 0.753, 0.796),
    }

    color_lower = color_str.lower()
    if color_lower in named_colors:
        r, g, b = named_colors[color_lower]
        return SolidColor(
            color=torch.tensor([r, g, b, opacity], device=device)
        )

    # Default to black
    return SolidColor(color=torch.tensor([0.0, 0.0, 0.0, opacity], device=device))


def _parse_gradient(
    elem: ET.Element, defs: dict, device: torch.device
) -> LinearGradient | RadialGradient | None:
    """Parse a gradient element."""
    tag = elem.tag.split("}")[-1]  # Remove namespace

    stops = []
    for stop in elem:
        stop_tag = stop.tag.split("}")[-1]
        if stop_tag == "stop":
            offset = float(stop.get("offset", "0").rstrip("%")) / 100.0 if "%" in stop.get("offset", "0") else float(stop.get("offset", "0"))
            stop_color = stop.get("stop-color", "#000000")
            stop_opacity = float(stop.get("stop-opacity", "1"))

            color = _parse_color(stop_color, stop_opacity, device)
            if color:
                stops.append((offset, color.color))

    if not stops:
        return None

    offsets = torch.tensor([s[0] for s in stops], device=device)
    stop_colors = torch.stack([s[1] for s in stops])

    if tag == "linearGradient":
        x1 = float(elem.get("x1", "0").rstrip("%")) / 100.0 if "%" in elem.get("x1", "0") else float(elem.get("x1", "0"))
        y1 = float(elem.get("y1", "0").rstrip("%")) / 100.0 if "%" in elem.get("y1", "0") else float(elem.get("y1", "0"))
        x2 = float(elem.get("x2", "1").rstrip("%")) / 100.0 if "%" in elem.get("x2", "1") else float(elem.get("x2", "1"))
        y2 = float(elem.get("y2", "0").rstrip("%")) / 100.0 if "%" in elem.get("y2", "0") else float(elem.get("y2", "0"))

        return LinearGradient(
            begin=torch.tensor([x1, y1], device=device),
            end=torch.tensor([x2, y2], device=device),
            offsets=offsets,
            stop_colors=stop_colors,
        )

    elif tag == "radialGradient":
        cx = float(elem.get("cx", "0.5").rstrip("%")) / 100.0 if "%" in elem.get("cx", "0.5") else float(elem.get("cx", "0.5"))
        cy = float(elem.get("cy", "0.5").rstrip("%")) / 100.0 if "%" in elem.get("cy", "0.5") else float(elem.get("cy", "0.5"))
        r = float(elem.get("r", "0.5").rstrip("%")) / 100.0 if "%" in elem.get("r", "0.5") else float(elem.get("r", "0.5"))

        return RadialGradient(
            center=torch.tensor([cx, cy], device=device),
            radius=torch.tensor([r, r], device=device),
            offsets=offsets,
            stop_colors=stop_colors,
        )

    return None


def _resolve_color(
    color_str: str | None,
    opacity: float,
    defs: dict,
    device: torch.device,
) -> SolidColor | LinearGradient | RadialGradient | None:
    """Resolve a color, which may be a url() reference."""
    if color_str is None or color_str.lower() == "none":
        return None

    if color_str.startswith("url("):
        # Extract ID from url(#id)
        match = re.match(r"url\(#([^)]+)\)", color_str)
        if match:
            ref_id = match.group(1)
            if ref_id in defs:
                return _parse_gradient(defs[ref_id], defs, device)
        return None

    return _parse_color(color_str, opacity, device)


def _parse_element(
    elem: ET.Element,
    shapes: list,
    shape_groups: list,
    defs: dict,
    device: torch.device,
    transform: torch.Tensor | None = None,
) -> None:
    """Recursively parse an SVG element and its children."""
    tag = elem.tag.split("}")[-1]  # Remove namespace

    # Skip definitions
    if tag == "defs":
        return

    # Handle groups
    if tag == "g":
        # Parse group transform if any
        group_transform = _parse_transform(elem.get("transform"), device)
        if transform is not None and group_transform is not None:
            combined = transform @ group_transform
        elif group_transform is not None:
            combined = group_transform
        else:
            combined = transform

        for child in elem:
            _parse_element(child, shapes, shape_groups, defs, device, combined)
        return

    # Parse shape attributes
    fill_str = elem.get("fill", "black")
    stroke_str = elem.get("stroke")
    fill_opacity = float(elem.get("fill-opacity", "1"))
    stroke_opacity = float(elem.get("stroke-opacity", "1"))
    opacity = float(elem.get("opacity", "1"))
    stroke_width = float(elem.get("stroke-width", "1"))

    fill_opacity *= opacity
    stroke_opacity *= opacity

    fill_color = _resolve_color(fill_str, fill_opacity, defs, device)
    stroke_color = _resolve_color(stroke_str, stroke_opacity, defs, device)

    # Parse element transform
    elem_transform = _parse_transform(elem.get("transform"), device)
    if transform is not None and elem_transform is not None:
        final_transform = transform @ elem_transform
    elif elem_transform is not None:
        final_transform = elem_transform
    elif transform is not None:
        final_transform = transform
    else:
        final_transform = torch.eye(3, device=device)

    shape = None

    if tag == "circle":
        cx = float(elem.get("cx", "0"))
        cy = float(elem.get("cy", "0"))
        r = float(elem.get("r", "0"))

        shape = Circle(
            center=torch.tensor([cx, cy], device=device),
            radius=torch.tensor(r, device=device),
            stroke_width=torch.tensor(stroke_width, device=device),
        )

    elif tag == "ellipse":
        cx = float(elem.get("cx", "0"))
        cy = float(elem.get("cy", "0"))
        rx = float(elem.get("rx", "0"))
        ry = float(elem.get("ry", "0"))

        shape = Ellipse(
            center=torch.tensor([cx, cy], device=device),
            radius=torch.tensor([rx, ry], device=device),
            stroke_width=torch.tensor(stroke_width, device=device),
        )

    elif tag == "rect":
        x = float(elem.get("x", "0"))
        y = float(elem.get("y", "0"))
        w = float(elem.get("width", "0"))
        h = float(elem.get("height", "0"))

        shape = Rect(
            p_min=torch.tensor([x, y], device=device),
            p_max=torch.tensor([x + w, y + h], device=device),
            stroke_width=torch.tensor(stroke_width, device=device),
        )

    elif tag == "polygon":
        points_str = elem.get("points", "")
        points = _parse_points(points_str)

        if len(points) > 0:
            shape = Polygon(
                points=torch.tensor(points, device=device),
                is_closed=True,
                stroke_width=torch.tensor(stroke_width, device=device),
            )

    elif tag == "polyline":
        points_str = elem.get("points", "")
        points = _parse_points(points_str)

        if len(points) > 0:
            shape = Polygon(
                points=torch.tensor(points, device=device),
                is_closed=False,
                stroke_width=torch.tensor(stroke_width, device=device),
            )

    elif tag == "line":
        x1 = float(elem.get("x1", "0"))
        y1 = float(elem.get("y1", "0"))
        x2 = float(elem.get("x2", "0"))
        y2 = float(elem.get("y2", "0"))

        shape = Polygon(
            points=torch.tensor([[x1, y1], [x2, y2]], device=device),
            is_closed=False,
            stroke_width=torch.tensor(stroke_width, device=device),
        )

    elif tag == "path":
        d = elem.get("d", "")
        path_shape = _parse_path_d(d, device, stroke_width)
        if path_shape:
            shape = path_shape

    if shape is not None:
        shape_idx = len(shapes)
        shapes.append(shape)

        group = ShapeGroup(
            shape_ids=torch.tensor([shape_idx], dtype=torch.int32, device=device),
            fill_color=fill_color,
            stroke_color=stroke_color,
            shape_to_canvas=final_transform,
        )
        shape_groups.append(group)

    # Recurse into children
    for child in elem:
        _parse_element(child, shapes, shape_groups, defs, device, transform)


def _parse_points(points_str: str) -> list[list[float]]:
    """Parse SVG points attribute (polygon/polyline)."""
    points = []
    # Split by whitespace and/or commas
    nums = re.findall(r"-?[\d.]+", points_str)
    for i in range(0, len(nums) - 1, 2):
        x = float(nums[i])
        y = float(nums[i + 1])
        points.append([x, y])
    return points


def _parse_transform(transform_str: str | None, device: torch.device) -> torch.Tensor | None:
    """Parse SVG transform attribute."""
    if not transform_str:
        return None

    result = torch.eye(3, device=device)

    # Parse transform functions
    transforms = re.findall(r"(\w+)\s*\(([^)]+)\)", transform_str)

    for func, args in transforms:
        nums = [float(x) for x in re.findall(r"-?[\d.]+", args)]

        if func == "translate":
            tx = nums[0] if len(nums) > 0 else 0
            ty = nums[1] if len(nums) > 1 else 0
            mat = torch.tensor(
                [[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=torch.float32, device=device
            )
            result = result @ mat

        elif func == "scale":
            sx = nums[0] if len(nums) > 0 else 1
            sy = nums[1] if len(nums) > 1 else sx
            mat = torch.tensor(
                [[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=torch.float32, device=device
            )
            result = result @ mat

        elif func == "rotate":
            angle = nums[0] if len(nums) > 0 else 0
            angle_rad = angle * 3.141592653589793 / 180.0
            cos_a = torch.cos(torch.tensor(angle_rad))
            sin_a = torch.sin(torch.tensor(angle_rad))
            mat = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                dtype=torch.float32,
                device=device,
            )

            # Handle rotation center
            if len(nums) >= 3:
                cx, cy = nums[1], nums[2]
                pre = torch.tensor(
                    [[1, 0, cx], [0, 1, cy], [0, 0, 1]],
                    dtype=torch.float32,
                    device=device,
                )
                post = torch.tensor(
                    [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]],
                    dtype=torch.float32,
                    device=device,
                )
                mat = pre @ mat @ post

            result = result @ mat

        elif func == "matrix":
            if len(nums) >= 6:
                a, b, c, d, e, f = nums[:6]
                mat = torch.tensor(
                    [[a, c, e], [b, d, f], [0, 0, 1]], dtype=torch.float32, device=device
                )
                result = result @ mat

    return result


def _parse_path_d(d: str, device: torch.device, stroke_width: float) -> PathShape | None:
    """Parse SVG path d attribute into a Path shape."""
    if not d:
        return None

    # Tokenize path data
    tokens = re.findall(r"[MmZzLlHhVvCcSsQqTtAa]|-?[\d.]+", d)

    points = []
    num_control_points = []

    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    prev_control_x, prev_control_y = None, None
    prev_cmd = None

    i = 0
    while i < len(tokens):
        cmd = tokens[i]

        if cmd in "MmZzLlHhVvCcSsQqTtAa":
            i += 1
        else:
            # Implicit command (repeat previous)
            if prev_cmd:
                cmd = prev_cmd
                if cmd == "M":
                    cmd = "L"
                elif cmd == "m":
                    cmd = "l"
            else:
                i += 1
                continue

        if cmd == "M":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            current_x, current_y = x, y
            start_x, start_y = x, y
            if points:
                # Start new subpath
                pass
            points.append([current_x, current_y])

        elif cmd == "m":
            dx = float(tokens[i])
            dy = float(tokens[i + 1])
            i += 2
            current_x += dx
            current_y += dy
            start_x, start_y = current_x, current_y
            points.append([current_x, current_y])

        elif cmd == "L":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            current_x, current_y = x, y
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "l":
            dx = float(tokens[i])
            dy = float(tokens[i + 1])
            i += 2
            current_x += dx
            current_y += dy
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "H":
            x = float(tokens[i])
            i += 1
            current_x = x
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "h":
            dx = float(tokens[i])
            i += 1
            current_x += dx
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "V":
            y = float(tokens[i])
            i += 1
            current_y = y
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "v":
            dy = float(tokens[i])
            i += 1
            current_y += dy
            points.append([current_x, current_y])
            num_control_points.append(0)

        elif cmd == "C":
            x1 = float(tokens[i])
            y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2])
            y2 = float(tokens[i + 3])
            x = float(tokens[i + 4])
            y = float(tokens[i + 5])
            i += 6
            points.append([x1, y1])
            points.append([x2, y2])
            points.append([x, y])
            num_control_points.append(2)
            current_x, current_y = x, y
            prev_control_x, prev_control_y = x2, y2

        elif cmd == "c":
            dx1 = float(tokens[i])
            dy1 = float(tokens[i + 1])
            dx2 = float(tokens[i + 2])
            dy2 = float(tokens[i + 3])
            dx = float(tokens[i + 4])
            dy = float(tokens[i + 5])
            i += 6
            x1, y1 = current_x + dx1, current_y + dy1
            x2, y2 = current_x + dx2, current_y + dy2
            x, y = current_x + dx, current_y + dy
            points.append([x1, y1])
            points.append([x2, y2])
            points.append([x, y])
            num_control_points.append(2)
            prev_control_x, prev_control_y = x2, y2
            current_x, current_y = x, y

        elif cmd == "Q":
            x1 = float(tokens[i])
            y1 = float(tokens[i + 1])
            x = float(tokens[i + 2])
            y = float(tokens[i + 3])
            i += 4
            points.append([x1, y1])
            points.append([x, y])
            num_control_points.append(1)
            current_x, current_y = x, y
            prev_control_x, prev_control_y = x1, y1

        elif cmd == "q":
            dx1 = float(tokens[i])
            dy1 = float(tokens[i + 1])
            dx = float(tokens[i + 2])
            dy = float(tokens[i + 3])
            i += 4
            x1, y1 = current_x + dx1, current_y + dy1
            x, y = current_x + dx, current_y + dy
            points.append([x1, y1])
            points.append([x, y])
            num_control_points.append(1)
            prev_control_x, prev_control_y = x1, y1
            current_x, current_y = x, y

        elif cmd in "Zz":
            # Close path
            if (current_x, current_y) != (start_x, start_y):
                points.append([start_x, start_y])
                num_control_points.append(0)
            current_x, current_y = start_x, start_y

        else:
            # Skip unsupported commands
            i += 1
            continue

        prev_cmd = cmd

    if len(points) < 2:
        return None

    # Determine if closed
    is_closed = (
        len(points) > 2
        and abs(points[-1][0] - points[0][0]) < 1e-6
        and abs(points[-1][1] - points[0][1]) < 1e-6
    )

    # Remove duplicate closing point if path is closed
    if is_closed and len(points) > 1:
        points = points[:-1]
        if num_control_points:
            num_control_points = num_control_points[:-1]

    return PathShape(
        num_control_points=torch.tensor(num_control_points, dtype=torch.int32, device=device),
        points=torch.tensor(points, dtype=torch.float32, device=device),
        is_closed=is_closed,
        stroke_width=torch.tensor(stroke_width, device=device),
    )
