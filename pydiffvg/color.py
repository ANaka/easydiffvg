"""Color types for pydiffvg."""

from dataclasses import dataclass

import torch


@dataclass
class SolidColor:
    """A solid RGBA color."""

    color: torch.Tensor  # [4] RGBA, values in [0, 1]


@dataclass
class LinearGradient:
    """A linear gradient color."""

    begin: torch.Tensor  # [2] start point
    end: torch.Tensor  # [2] end point
    offsets: torch.Tensor  # [S] stop positions in [0, 1]
    stop_colors: torch.Tensor  # [S, 4] RGBA at each stop


@dataclass
class RadialGradient:
    """A radial gradient color."""

    center: torch.Tensor  # [2] center point
    radius: torch.Tensor  # [2] rx, ry (can be elliptical)
    offsets: torch.Tensor  # [S] stop positions in [0, 1]
    stop_colors: torch.Tensor  # [S, 4] RGBA at each stop


# Type alias for any color
Color = SolidColor | LinearGradient | RadialGradient
