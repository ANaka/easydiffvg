"""Shape grouping with appearance properties."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pydiffvg.color import Color


@dataclass
class ShapeGroup:
    """Groups shapes together with fill/stroke colors and transform."""

    shape_ids: torch.Tensor  # [N] indices into shapes list
    fill_color: "Color | None"
    stroke_color: "Color | None" = None
    use_even_odd_rule: bool = True
    shape_to_canvas: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    id: str = ""
