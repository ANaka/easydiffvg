"""Utility modules for pydiffvg."""

from pydiffvg.utils.bezier import (
    evaluate_quadratic,
    evaluate_cubic,
    quadratic_to_cubic,
    subdivide_cubic,
    cubic_bounding_box,
)
from pydiffvg.utils.winding import (
    winding_number_line,
    winding_number_quadratic,
    winding_number_cubic,
)
from pydiffvg.utils.distance import (
    distance_to_line_segment,
    distance_to_quadratic_bezier,
    distance_to_cubic_bezier,
)

__all__ = [
    "evaluate_quadratic",
    "evaluate_cubic",
    "quadratic_to_cubic",
    "subdivide_cubic",
    "cubic_bounding_box",
    "winding_number_line",
    "winding_number_quadratic",
    "winding_number_cubic",
    "distance_to_line_segment",
    "distance_to_quadratic_bezier",
    "distance_to_cubic_bezier",
]
