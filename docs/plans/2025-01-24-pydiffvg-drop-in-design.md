# pydiffvg Drop-in Replacement Design

**Date:** 2025-01-24
**Goal:** Make easydiffvg a drop-in replacement for pydiffvg

## Overview

Transform easydiffvg into a package that provides the `pydiffvg` module directly, so users can:

1. Add the fork to `requirements.txt` or `pyproject.toml`
2. `import pydiffvg` works without any code changes
3. No C++ compilation, no CUDA toolkit - just `pip install`

## Approach

- Fork `BachiLi/diffvg`, replace C++/CUDA with pure PyTorch
- Keep original code in `.original_diffvg/` for reference
- Match original pydiffvg public API exactly
- Internal structure can differ - only public API matters

## Package Structure

```
diffvg/                    # repo root (forked from BachiLi/diffvg)
├── .original_diffvg/      # original C++/CUDA code for reference
├── pydiffvg/              # the pure PyTorch package
│   ├── __init__.py        # public API exports
│   ├── shapes.py          # Circle, Ellipse, Path, Polygon, Rect
│   ├── groups.py          # ShapeGroup
│   ├── color.py           # SolidColor, LinearGradient, RadialGradient
│   ├── render.py          # RenderFunction, render()
│   ├── rasterize.py       # core rasterization
│   ├── gradients.py       # boundary sampling for backward pass
│   ├── bvh.py             # acceleration structure
│   ├── device.py          # get_device, set_device, get_use_gpu, set_use_gpu
│   ├── image.py           # imwrite
│   ├── pixel_filter.py    # PixelFilter class
│   ├── optimize_svg.py    # SVG optimization utilities
│   ├── svg/
│   │   ├── parse.py       # parse_svg
│   │   └── save.py        # save_svg
│   └── utils/
│       ├── bezier.py
│       ├── winding.py
│       └── distance.py
├── pyproject.toml
└── README.md
```

The `__init__.py` uses `from .module import *` pattern to match original's flat namespace.

## Shape API

Shape classes match original signatures exactly:

```python
class Circle:
    def __init__(self, radius, center, stroke_width=torch.tensor(1.0)):
        self.radius = radius          # scalar tensor
        self.center = center          # [2] tensor
        self.stroke_width = stroke_width

class Ellipse:
    def __init__(self, radius, center, stroke_width=torch.tensor(1.0)):
        self.radius = radius          # [2] tensor (rx, ry)
        self.center = center          # [2]
        self.stroke_width = stroke_width

class Path:
    def __init__(self, num_control_points, points, is_closed,
                 stroke_width=torch.tensor(1.0), use_distance_approx=False):
        self.num_control_points = num_control_points  # [N] int tensor
        self.points = points                          # [M, 2] tensor
        self.is_closed = is_closed                    # bool
        self.stroke_width = stroke_width
        self.use_distance_approx = use_distance_approx

class Polygon:
    def __init__(self, points, is_closed=True, stroke_width=torch.tensor(1.0)):
        self.points = points          # [N, 2]
        self.is_closed = is_closed
        self.stroke_width = stroke_width

class Rect:
    def __init__(self, p_min, p_max, stroke_width=torch.tensor(1.0)):
        self.p_min = p_min            # [2]
        self.p_max = p_max            # [2]
        self.stroke_width = stroke_width
```

**Utility function:**
```python
def from_svg_path(path_str: str) -> Path:
    """Convert SVG path string to Path object."""
```

## ShapeGroup and Colors

ShapeGroup accepts both raw tensors and color objects for backward compatibility:

```python
class ShapeGroup:
    def __init__(
        self,
        shape_ids,                    # [N] int tensor
        fill_color=None,              # Tensor | LinearGradient | RadialGradient | None
        stroke_color=None,            # same
        use_even_odd_rule=False,
        shape_to_canvas=torch.eye(3), # [3, 3] transform matrix
    ):
        self.shape_ids = shape_ids
        self.fill_color = self._normalize_color(fill_color)
        self.stroke_color = self._normalize_color(stroke_color)
        self.use_even_odd_rule = use_even_odd_rule
        self.shape_to_canvas = shape_to_canvas

    def _normalize_color(self, color):
        """Convert raw tensor to SolidColor, pass through gradients."""
        if color is None:
            return None
        if isinstance(color, torch.Tensor):
            return SolidColor(color)  # internal wrapper
        return color  # LinearGradient or RadialGradient
```

Color classes match original:

```python
class LinearGradient:
    def __init__(self, begin, end, offsets, stop_colors):
        self.begin = begin            # [2]
        self.end = end                # [2]
        self.offsets = offsets        # [S]
        self.stop_colors = stop_colors # [S, 4]

class RadialGradient:
    def __init__(self, center, radius, offsets, stop_colors):
        self.center = center          # [2]
        self.radius = radius          # [2] (rx, ry)
        self.offsets = offsets        # [S]
        self.stop_colors = stop_colors # [S, 4]
```

`SolidColor` is internal - users never need to import it directly.

## Render API

RenderFunction provides both the original serialize_scene pattern and a clean convenience API:

```python
class OutputType(IntEnum):
    color = 1
    sdf = 2

class RenderFunction(torch.autograd.Function):
    @staticmethod
    def serialize_scene(canvas_width, canvas_height, shapes, shape_groups,
                        filter=PixelFilter(type=FilterType.box, radius=0.5),
                        output_type=OutputType.color,
                        use_prefiltering=False,
                        eval_positions=torch.tensor([])):
        """Serialize scene to flat args list for apply()."""
        # Returns list of args matching original format
    
    @staticmethod
    def forward(ctx, width, height, num_samples_x, num_samples_y, 
                seed, background_image, *args):
        """Forward rendering pass."""
    
    @staticmethod
    def backward(ctx, grad_img):
        """Backward pass with boundary sampling."""

# Clean API (bonus, not in original)
def render(canvas_width, canvas_height, shapes, shape_groups,
           num_samples_x=2, num_samples_y=2, seed=0,
           background_image=None, filter=None,
           output_type=OutputType.color) -> torch.Tensor:
    """Convenience wrapper - handles serialization internally."""
```

**Both usage patterns work:**
```python
# Original pattern
args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, groups)
img = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *args)

# Clean pattern  
img = pydiffvg.render(w, h, shapes, groups)
```

## Utilities and Extras

**PixelFilter:**
```python
class FilterType(IntEnum):
    box = 0
    tent = 1
    radial_paraboloid = 2
    hann = 3

class PixelFilter:
    def __init__(self, type=FilterType.box, radius=torch.tensor(0.5)):
        self.type = type
        self.radius = radius
```

**Device management:**
```python
_device = torch.device('cpu')
_use_gpu = False

def get_device() -> torch.device: ...
def set_device(device: torch.device): ...
def get_use_gpu() -> bool: ...
def set_use_gpu(use_gpu: bool): ...
```

**Timing debug:**
```python
print_timing = False
def set_print_timing(val: bool): ...
```

**SVG utilities:**
```python
def parse_svg(filename, device='cpu') -> tuple[int, int, list, list]:
    """Returns (width, height, shapes, shape_groups)"""

def save_svg(filename, width, height, shapes, shape_groups): ...

# optimize_svg.py
def optimize_svg(shapes, shape_groups, ...) -> tuple[list, list]:
    """Simplify/optimize SVG shapes."""
```

**Image I/O:**
```python
def imwrite(img: torch.Tensor, filename: str): ...
```

## SDF Output Mode

The original supports rendering signed distance fields instead of color images:

```python
# SDF output mode
args = pydiffvg.RenderFunction.serialize_scene(
    w, h, shapes, groups,
    output_type=pydiffvg.OutputType.sdf
)
sdf = pydiffvg.RenderFunction.apply(w, h, 2, 2, 0, None, *args)
# Returns [H, W, 1] - distance to nearest shape boundary

# Can also evaluate at arbitrary positions
args = pydiffvg.RenderFunction.serialize_scene(
    w, h, shapes, groups,
    output_type=pydiffvg.OutputType.sdf,
    eval_positions=torch.tensor([[10.0, 20.0], [30.0, 40.0]])
)
sdf = pydiffvg.RenderFunction.apply(...)
# Returns [N, 1] - SDF at each eval position
```

**Implementation approach:**
- Reuse distance computation from `utils/distance.py`
- For each pixel/position, compute minimum signed distance to all shapes
- Negative inside, positive outside
- Differentiable w.r.t. shape parameters

This is useful for loss functions that care about shape boundaries rather than filled regions.

## Migration Steps

### Step 1: Reorganize files
- Move `diffvg/` → `.original_diffvg/`
- Move `src/easydiffvg/` → `pydiffvg/`
- Update `pyproject.toml` to package `pydiffvg`

### Step 2: Update shape classes
- Change from dataclasses to regular classes
- Add `stroke_width`, `use_distance_approx` params
- Match original constructor signatures

### Step 3: Update ShapeGroup
- Accept raw tensors for colors (auto-wrap to SolidColor)
- Match original param names/order

### Step 4: Add RenderFunction.serialize_scene()
- Implement scene serialization matching original format
- Update `apply()` signature to match

### Step 5: Add missing utilities
- `from_svg_path()` 
- `PixelFilter` class with FilterType enum
- `OutputType` enum
- `optimize_svg` module
- `set_print_timing()`

### Step 6: Implement SDF output
- Add output_type support to render pipeline
- Support eval_positions for arbitrary point queries

### Step 7: Update tests
- Rename imports from easydiffvg to pydiffvg
- Add compatibility tests against original API

## Non-Goals

- CUDA kernels / native code (pure PyTorch only)
- TensorFlow support (PyTorch only)
- Backward compatibility with Python < 3.12

## Dependencies

- `torch` - core tensor operations and autograd
- `svgpathtools` - SVG parsing
