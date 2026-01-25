"""pydiffvg: Pure PyTorch differentiable vector graphics.

A drop-in replacement for pydiffvg that requires no native compilation.
Simply `pip install pydiffvg` and it just works.

Example:
    >>> import torch
    >>> import pydiffvg
    >>>
    >>> # Create a red circle
    >>> circle = pydiffvg.Circle(
    ...     radius=torch.tensor(20.0),
    ...     center=torch.tensor([32.0, 32.0]),
    ... )
    >>> group = pydiffvg.ShapeGroup(
    ...     shape_ids=torch.tensor([0]),
    ...     fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]),
    ... )
    >>>
    >>> # Render to image
    >>> image = pydiffvg.render(64, 64, [circle], [group])
    >>> print(image.shape)
    torch.Size([64, 64, 4])
"""

from pydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect, Shape, from_svg_path
from pydiffvg.groups import ShapeGroup
from pydiffvg.color import Color, LinearGradient, RadialGradient, SolidColor
from pydiffvg.render import (
    RenderFunction,
    render,
    OutputType,
    print_timing,
    set_print_timing,
)
from pydiffvg.render_diff import render_differentiable
from pydiffvg.rasterize import PixelFilter, FilterType
from pydiffvg.svg import parse_svg, save_svg
from pydiffvg.device import get_device, get_use_gpu, set_device, set_use_gpu
from pydiffvg.image import imwrite

__version__ = "0.1.0"

__all__ = [
    # Shapes
    "Circle",
    "Ellipse",
    "Path",
    "Polygon",
    "Rect",
    "Shape",
    "from_svg_path",
    # Groups
    "ShapeGroup",
    # Colors
    "Color",
    "LinearGradient",
    "RadialGradient",
    "SolidColor",
    # Rendering
    "RenderFunction",
    "render",
    "render_differentiable",
    "OutputType",
    "PixelFilter",
    "FilterType",
    # Timing
    "print_timing",
    "set_print_timing",
    # SVG I/O
    "parse_svg",
    "save_svg",
    # Device management
    "get_device",
    "get_use_gpu",
    "set_device",
    "set_use_gpu",
    # Image I/O
    "imwrite",
]
