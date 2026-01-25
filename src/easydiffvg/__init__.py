"""easydiffvg: Pure PyTorch differentiable vector graphics.

A drop-in replacement for pydiffvg that requires no native compilation.
Simply `pip install easydiffvg` and it just works.

Example:
    >>> import torch
    >>> import easydiffvg
    >>>
    >>> # Create a red circle
    >>> circle = easydiffvg.Circle(
    ...     center=torch.tensor([32.0, 32.0]),
    ...     radius=torch.tensor(20.0),
    ... )
    >>> group = easydiffvg.ShapeGroup(
    ...     shape_ids=torch.tensor([0]),
    ...     fill_color=easydiffvg.SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
    ... )
    >>>
    >>> # Render to image
    >>> image = easydiffvg.render(64, 64, [circle], [group])
    >>> print(image.shape)
    torch.Size([64, 64, 4])
"""

from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect, Shape
from easydiffvg.groups import ShapeGroup
from easydiffvg.color import Color, LinearGradient, RadialGradient, SolidColor
from easydiffvg.render import RenderFunction, render
from easydiffvg.render_diff import render_differentiable
from easydiffvg.rasterize import PixelFilter
from easydiffvg.svg import parse_svg, save_svg
from easydiffvg.device import get_device, get_use_gpu, set_device, set_use_gpu
from easydiffvg.image import imwrite

__version__ = "0.1.0"

__all__ = [
    # Shapes
    "Circle",
    "Ellipse",
    "Path",
    "Polygon",
    "Rect",
    "Shape",
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
    "PixelFilter",
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
