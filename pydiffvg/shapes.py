"""Shape primitives for pydiffvg."""

import math

import svgpathtools
import torch


class Circle:
    """A circle shape defined by center point and radius."""

    def __init__(self, radius, center, stroke_width=torch.tensor(1.0), id=""):
        self.radius = radius
        self.center = center
        self.stroke_width = stroke_width
        self.id = id


class Ellipse:
    """An ellipse shape defined by center and radii (rx, ry)."""

    def __init__(self, radius, center, stroke_width=torch.tensor(1.0), id=""):
        self.radius = radius  # [2] rx, ry
        self.center = center  # [2] x, y
        self.stroke_width = stroke_width
        self.id = id


class Path:
    """A path composed of line and bezier curve segments.

    Each segment is defined by num_control_points:
    - 0: line segment (2 points: start, end)
    - 1: quadratic bezier (3 points: start, control, end)
    - 2: cubic bezier (4 points: start, ctrl1, ctrl2, end)

    Points are shared between segments (end of one = start of next).
    """

    def __init__(
        self,
        num_control_points,
        points,
        is_closed,
        stroke_width=torch.tensor(1.0),
        id="",
        use_distance_approx=False,
    ):
        self.num_control_points = num_control_points  # [M] per-segment control point count
        self.points = points  # [N, 2] all control points
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.id = id
        self.use_distance_approx = use_distance_approx


class Polygon:
    """A polygon or polyline defined by a sequence of points."""

    def __init__(self, points, is_closed, stroke_width=torch.tensor(1.0), id=""):
        self.points = points  # [N, 2] vertices
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.id = id


class Rect:
    """A rectangle defined by min and max corners."""

    def __init__(self, p_min, p_max, stroke_width=torch.tensor(1.0), id=""):
        self.p_min = p_min  # [2] top-left corner
        self.p_max = p_max  # [2] bottom-right corner
        self.stroke_width = stroke_width
        self.id = id


# Type alias for any shape
Shape = Circle | Ellipse | Rect | Polygon | Path


def from_svg_path(path_str, shape_to_canvas=torch.eye(3), force_close=False):
    """Convert SVG path string to list of Path objects.

    Args:
        path_str: SVG path data string (e.g., "M 0 0 L 100 100")
        shape_to_canvas: 3x3 transformation matrix to apply to points
        force_close: If True, force open paths to be closed

    Returns:
        List of Path objects
    """
    path = svgpathtools.parse_path(path_str)
    if len(path) == 0:
        return []
    ret_paths = []
    subpaths = path.continuous_subpaths()
    for subpath in subpaths:
        if subpath.isclosed():
            if (
                len(subpath) > 1
                and isinstance(subpath[-1], svgpathtools.Line)
                and subpath[-1].length() < 1e-5
            ):
                subpath.remove(subpath[-1])
                subpath[-1].end = subpath[0].start  # Force closing the path
                subpath.end = subpath[-1].end
                assert subpath.isclosed()
        else:
            beg = subpath[0].start
            end = subpath[-1].end
            if abs(end - beg) < 1e-5:
                subpath[-1].end = beg  # Force closing the path
                subpath.end = subpath[-1].end
                assert subpath.isclosed()
            elif force_close:
                subpath.append(svgpathtools.Line(end, beg))
                subpath.end = subpath[-1].end
                assert subpath.isclosed()

        num_control_points = []
        points = []

        for i, e in enumerate(subpath):
            if i == 0:
                points.append((e.start.real, e.start.imag))
            else:
                # Must begin from the end of previous segment
                assert e.start.real == points[-1][0]
                assert e.start.imag == points[-1][1]
            if isinstance(e, svgpathtools.Line):
                num_control_points.append(0)
            elif isinstance(e, svgpathtools.QuadraticBezier):
                num_control_points.append(1)
                points.append((e.control.real, e.control.imag))
            elif isinstance(e, svgpathtools.CubicBezier):
                num_control_points.append(2)
                points.append((e.control1.real, e.control1.imag))
                points.append((e.control2.real, e.control2.imag))
            elif isinstance(e, svgpathtools.Arc):
                # Convert to Cubic curves
                # https://www.joecridge.me/content/pdf/bezier-arcs.pdf
                start = e.theta * math.pi / 180.0
                stop = (e.theta + e.delta) * math.pi / 180.0

                sign = 1.0
                if stop < start:
                    sign = -1.0

                epsilon = 0.00001
                while sign * (stop - start) > epsilon:
                    arc_to_draw = stop - start
                    if arc_to_draw > 0.0:
                        arc_to_draw = min(arc_to_draw, 0.5 * math.pi)
                    else:
                        arc_to_draw = max(arc_to_draw, -0.5 * math.pi)
                    alpha = arc_to_draw / 2.0
                    cos_alpha = math.cos(alpha)
                    sin_alpha = math.sin(alpha)
                    cot_alpha = 1.0 / math.tan(alpha)
                    phi = start + alpha
                    cos_phi = math.cos(phi)
                    sin_phi = math.sin(phi)
                    lambda_ = (4.0 - cos_alpha) / 3.0
                    mu = sin_alpha + (cos_alpha - lambda_) * cot_alpha
                    last = sign * (stop - (start + arc_to_draw)) <= epsilon
                    num_control_points.append(2)
                    rx = e.radius.real
                    ry = e.radius.imag
                    cx = e.center.real
                    cy = e.center.imag
                    rot = e.phi * math.pi / 180.0
                    cos_rot = math.cos(rot)
                    sin_rot = math.sin(rot)
                    x = lambda_ * cos_phi + mu * sin_phi
                    y = lambda_ * sin_phi - mu * cos_phi
                    xx = x * cos_rot - y * sin_rot
                    yy = x * sin_rot + y * cos_rot
                    points.append((cx + rx * xx, cy + ry * yy))
                    x = lambda_ * cos_phi - mu * sin_phi
                    y = lambda_ * sin_phi + mu * cos_phi
                    xx = x * cos_rot - y * sin_rot
                    yy = x * sin_rot + y * cos_rot
                    points.append((cx + rx * xx, cy + ry * yy))
                    if not last:
                        points.append(
                            (
                                cx + rx * math.cos(rot + start + arc_to_draw),
                                cy + ry * math.sin(rot + start + arc_to_draw),
                            )
                        )
                    start += arc_to_draw
            if i != len(subpath) - 1:
                points.append((e.end.real, e.end.imag))
            else:
                if subpath.isclosed():
                    # Must end at the beginning of first segment
                    assert e.end.real == points[0][0]
                    assert e.end.imag == points[0][1]
                else:
                    points.append((e.end.real, e.end.imag))
        points = torch.tensor(points, dtype=torch.float)
        points = torch.cat((points, torch.ones([points.shape[0], 1])), dim=1) @ torch.transpose(
            shape_to_canvas, 0, 1
        )
        points = points / points[:, 2:3]
        points = points[:, :2].contiguous()
        ret_paths.append(Path(torch.tensor(num_control_points), points, subpath.isclosed()))
    return ret_paths
