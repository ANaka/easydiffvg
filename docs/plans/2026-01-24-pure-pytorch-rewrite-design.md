# easydiffvg: Pure PyTorch Rewrite Design

## Overview

Reimplement diffvg as a pure PyTorch library for easy `pip install` without native compilation.

**Goals:**
- `pip install easydiffvg` just works (no CMake, no C++ compiler, no CUDA toolkit)
- Drop-in API replacement for original pydiffvg
- Full feature parity: all shapes, gradients, transforms, SVG I/O
- Device-agnostic: runs on CPU, CUDA, MPS, or any PyTorch-supported device

## Architecture

```
easydiffvg/
├── __init__.py          # Public API exports
├── shapes.py            # Shape primitives (Circle, Ellipse, Path, Polygon, Rect)
├── groups.py            # ShapeGroup with fill/stroke/transform
├── color.py             # SolidColor, LinearGradient, RadialGradient
├── render.py            # RenderFunction (torch.autograd.Function)
├── rasterize.py         # Core rasterization logic (forward pass)
├── gradients.py         # Boundary sampling for backward pass
├── bvh.py               # Bounding volume hierarchy for acceleration
├── svg/
│   ├── parse.py         # SVG → shapes
│   └── save.py          # shapes → SVG
└── utils/
    ├── bezier.py        # Bezier curve math
    ├── winding.py       # Winding number computation
    └── distance.py      # Distance field utilities
```

Each module uses pure PyTorch tensors. No numpy required at runtime.

## Core Rendering Pipeline

### Forward Pass (rasterize.py)

```python
def render(
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
    samples: int = 2,  # antialiasing samples per pixel
    filter: PixelFilter = PixelFilter.BOX,
) -> torch.Tensor:  # [H, W, 4] RGBA
```

For each pixel:
1. Generate sample points (stratified or random based on `samples`)
2. Query BVH to find candidate shapes
3. For each sample: compute winding number → inside/outside → accumulate color
4. Average samples → final pixel color

### Backward Pass (gradients.py)

Standard autodiff fails at shape boundaries (discontinuous). Instead, use **boundary sampling**:

1. Find pixels near shape edges
2. Sample points along the boundary
3. Compute how moving the boundary affects pixel coverage
4. Chain rule back to shape parameters (control points, positions, etc.)

This implements the Reynolds transport theorem for boundary integrals - the same approach as the original diffvg.

## Shape Primitives

All shapes store parameters as `torch.Tensor` for differentiability:

```python
@dataclass
class Circle:
    center: torch.Tensor      # [2] x, y
    radius: torch.Tensor      # [1]

@dataclass
class Ellipse:
    center: torch.Tensor      # [2]
    radius: torch.Tensor      # [2] rx, ry

@dataclass
class Rect:
    p_min: torch.Tensor       # [2] top-left
    p_max: torch.Tensor       # [2] bottom-right

@dataclass
class Polygon:
    points: torch.Tensor      # [N, 2]

@dataclass
class Path:
    points: torch.Tensor      # [N, 2] control points
    num_control_points: torch.Tensor  # [M] per-segment (0=line, 1=quadratic, 2=cubic)
    is_closed: bool
```

## ShapeGroup

Bundles shapes with appearance:

```python
@dataclass
class ShapeGroup:
    shape_ids: list[int]              # indices into shapes list
    fill_color: Color | None          # SolidColor, LinearGradient, or RadialGradient
    stroke_color: Color | None
    stroke_width: torch.Tensor | None
    shape_to_canvas: torch.Tensor     # [3, 3] transform matrix
    use_even_odd_rule: bool = False
```

## Colors and Gradients

```python
@dataclass
class SolidColor:
    color: torch.Tensor  # [4] RGBA, values in [0, 1]

@dataclass
class LinearGradient:
    begin: torch.Tensor           # [2] start point
    end: torch.Tensor             # [2] end point
    offsets: torch.Tensor         # [S] stop positions in [0, 1]
    stop_colors: torch.Tensor     # [S, 4] RGBA at each stop

@dataclass
class RadialGradient:
    center: torch.Tensor          # [2]
    radius: torch.Tensor          # [2] rx, ry
    offsets: torch.Tensor         # [S]
    stop_colors: torch.Tensor     # [S, 4]

Color = SolidColor | LinearGradient | RadialGradient
```

## BVH Acceleration

Without acceleration, rendering is O(pixels × shapes). BVH makes it O(pixels × log(shapes)).

```python
@dataclass
class BVHNode:
    bbox: torch.Tensor      # [4] min_x, min_y, max_x, max_y
    left: int | None        # child index or None if leaf
    right: int | None
    shape_id: int | None    # only set for leaf nodes

def build_bvh(shapes: list[Shape]) -> list[BVHNode]:
    """Build BVH using surface area heuristic."""

def query_bvh(nodes: list[BVHNode], point: torch.Tensor) -> list[int]:
    """Return shape IDs whose bboxes contain point."""
```

BVH rebuilt per-frame during optimization. Fast enough for typical shape counts (<1000).

Paths get internal BVH over segments since a single path can have hundreds of bezier curves.

## SVG Import/Export

```python
def parse_svg(filename: str, device: torch.device = 'cpu') -> tuple[
    int,                    # canvas_width
    int,                    # canvas_height
    list[Shape],            # shapes
    list[ShapeGroup],       # shape_groups
]:
    """Parse SVG file into easydiffvg primitives."""

def save_svg(
    filename: str,
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
):
    """Export shapes to SVG file."""
```

Uses `svgpathtools` dependency for parsing SVG path syntax.

## Public API

```python
from .shapes import Circle, Ellipse, Path, Polygon, Rect
from .groups import ShapeGroup
from .color import SolidColor, LinearGradient, RadialGradient
from .render import RenderFunction
from .svg.parse import parse_svg
from .svg.save import save_svg

def render(
    canvas_width: int,
    canvas_height: int,
    shapes: list,
    shape_groups: list,
    samples: int = 2,
    filter = None,
) -> torch.Tensor:
    """Convenience wrapper around RenderFunction."""
    return RenderFunction.apply(
        canvas_width, canvas_height, shapes, shape_groups, samples, filter
    )
```

## Migration Path

```python
# Before
import pydiffvg
img = pydiffvg.render(...)

# After
import easydiffvg
img = easydiffvg.render(...)
```

## Dependencies

- `torch` - core tensor operations and autograd
- `svgpathtools` - SVG parsing (existing diffvg dependency)

## Non-Goals

- CUDA kernels / native code (pure PyTorch only)
- TensorFlow support (PyTorch only)
- Backward compatibility with Python < 3.12
