"""Shape grouping with appearance properties."""

import torch

from pydiffvg.color import LinearGradient, RadialGradient, SolidColor


class ShapeGroup:
    """Groups shapes together with fill/stroke colors and transform.

    Colors can be specified as:
    - Raw torch.Tensor [4] RGBA - will be wrapped in SolidColor internally
    - SolidColor, LinearGradient, or RadialGradient objects
    - None for no fill/stroke
    """

    def __init__(
        self,
        shape_ids,
        fill_color,
        use_even_odd_rule=True,
        stroke_color=None,
        shape_to_canvas=torch.eye(3),
        id="",
    ):
        self.shape_ids = shape_ids  # [N] indices into shapes list
        self.fill_color = self._normalize_color(fill_color)
        self.use_even_odd_rule = use_even_odd_rule
        self.stroke_color = self._normalize_color(stroke_color)
        self.shape_to_canvas = shape_to_canvas
        self.id = id

    def _normalize_color(self, color):
        """Convert raw tensor to SolidColor, pass through gradients."""
        if color is None:
            return None
        if isinstance(color, torch.Tensor):
            return SolidColor(color)  # internal wrapper
        if isinstance(color, (SolidColor, LinearGradient, RadialGradient)):
            return color
        raise TypeError(f"Invalid color type: {type(color)}")
