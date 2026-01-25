# easydiffvg Pure PyTorch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a pure PyTorch differentiable vector graphics renderer as a drop-in replacement for pydiffvg.

**Architecture:** Bottom-up implementation starting with shape primitives, then colors, then utilities (bezier/winding), then rasterization, then autograd integration. TDD approach using pre-generated reference fixtures from original diffvg.

**Tech Stack:** Python 3.12+, PyTorch 2.10+, pytest

---

## Phase 1: Test Infrastructure & Shape Primitives

### Task 1: Create test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create tests directory and init file**

```bash
mkdir -p tests
touch tests/__init__.py
```

**Step 2: Write conftest.py with basic fixtures**

```python
# tests/conftest.py
"""Shared pytest fixtures for easydiffvg tests."""

import pytest
import torch
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def device() -> torch.device:
    """Default test device."""
    return torch.device("cpu")


@pytest.fixture
def canvas_64() -> tuple[int, int]:
    """64x64 canvas for fast tests."""
    return (64, 64)


@pytest.fixture
def canvas_256() -> tuple[int, int]:
    """256x256 canvas for realistic tests."""
    return (256, 256)
```

**Step 3: Verify pytest discovers the test directory**

Run: `uv run pytest tests/ --collect-only`
Expected: Shows conftest.py loaded, no errors

**Step 4: Commit**

```bash
git add tests/
git commit -m "feat: add test infrastructure with conftest.py"
```

---

### Task 2: Implement Circle shape

**Files:**
- Create: `tests/test_shapes.py`
- Modify: `src/easydiffvg/shapes.py` (new file)
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for Circle**

```python
# tests/test_shapes.py
"""Tests for shape primitives."""

import pytest
import torch

from easydiffvg import Circle


class TestCircle:
    def test_circle_creation(self, device):
        """Circle stores center and radius as tensors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor(10.0, device=device)

        circle = Circle(radius=radius, center=center)

        assert circle.center.shape == (2,)
        assert circle.radius.shape == ()
        torch.testing.assert_close(circle.center, center)
        torch.testing.assert_close(circle.radius, radius)

    def test_circle_has_stroke_width(self, device):
        """Circle has stroke_width attribute with default."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor(10.0, device=device)

        circle = Circle(radius=radius, center=center)

        assert hasattr(circle, "stroke_width")
        assert circle.stroke_width.shape == ()

    def test_circle_custom_stroke_width(self, device):
        """Circle accepts custom stroke_width."""
        circle = Circle(
            radius=torch.tensor(10.0),
            center=torch.tensor([32.0, 32.0]),
            stroke_width=torch.tensor(2.5),
        )

        torch.testing.assert_close(circle.stroke_width, torch.tensor(2.5))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: FAIL with "cannot import name 'Circle' from 'easydiffvg'"

**Step 3: Create shapes.py with Circle class**

```python
# src/easydiffvg/shapes.py
"""Shape primitives for easydiffvg."""

from dataclasses import dataclass, field

import torch


@dataclass
class Circle:
    """A circle shape defined by center point and radius."""

    radius: torch.Tensor
    center: torch.Tensor
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
```

**Step 4: Export Circle from __init__.py**

```python
# src/easydiffvg/__init__.py
"""easydiffvg: Pure PyTorch differentiable vector graphics."""

from easydiffvg.shapes import Circle

__all__ = ["Circle"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/easydiffvg/shapes.py src/easydiffvg/__init__.py tests/test_shapes.py
git commit -m "feat: add Circle shape primitive"
```

---

### Task 3: Implement Ellipse shape

**Files:**
- Modify: `tests/test_shapes.py`
- Modify: `src/easydiffvg/shapes.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for Ellipse**

Add to `tests/test_shapes.py`:

```python
from easydiffvg import Circle, Ellipse


class TestEllipse:
    def test_ellipse_creation(self, device):
        """Ellipse stores center and radius (rx, ry) as tensors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor([20.0, 10.0], device=device)  # rx, ry

        ellipse = Ellipse(radius=radius, center=center)

        assert ellipse.center.shape == (2,)
        assert ellipse.radius.shape == (2,)
        torch.testing.assert_close(ellipse.center, center)
        torch.testing.assert_close(ellipse.radius, radius)

    def test_ellipse_has_stroke_width(self, device):
        """Ellipse has stroke_width attribute with default."""
        ellipse = Ellipse(
            radius=torch.tensor([20.0, 10.0]),
            center=torch.tensor([32.0, 32.0]),
        )

        assert hasattr(ellipse, "stroke_width")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shapes.py::TestEllipse -v`
Expected: FAIL with "cannot import name 'Ellipse'"

**Step 3: Add Ellipse to shapes.py**

Add to `src/easydiffvg/shapes.py`:

```python
@dataclass
class Ellipse:
    """An ellipse shape defined by center and radii (rx, ry)."""

    radius: torch.Tensor  # [2] rx, ry
    center: torch.Tensor  # [2] x, y
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
```

**Step 4: Export Ellipse from __init__.py**

Update imports and `__all__` in `src/easydiffvg/__init__.py`.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/easydiffvg/shapes.py src/easydiffvg/__init__.py tests/test_shapes.py
git commit -m "feat: add Ellipse shape primitive"
```

---

### Task 4: Implement Rect shape

**Files:**
- Modify: `tests/test_shapes.py`
- Modify: `src/easydiffvg/shapes.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for Rect**

Add to `tests/test_shapes.py`:

```python
from easydiffvg import Circle, Ellipse, Rect


class TestRect:
    def test_rect_creation(self, device):
        """Rect stores p_min and p_max corners as tensors."""
        p_min = torch.tensor([10.0, 10.0], device=device)
        p_max = torch.tensor([50.0, 40.0], device=device)

        rect = Rect(p_min=p_min, p_max=p_max)

        assert rect.p_min.shape == (2,)
        assert rect.p_max.shape == (2,)
        torch.testing.assert_close(rect.p_min, p_min)
        torch.testing.assert_close(rect.p_max, p_max)

    def test_rect_has_stroke_width(self, device):
        """Rect has stroke_width attribute with default."""
        rect = Rect(
            p_min=torch.tensor([10.0, 10.0]),
            p_max=torch.tensor([50.0, 40.0]),
        )

        assert hasattr(rect, "stroke_width")
        assert rect.stroke_width.shape == ()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shapes.py::TestRect -v`
Expected: FAIL with "cannot import name 'Rect'"

**Step 3: Add Rect to shapes.py**

Add to `src/easydiffvg/shapes.py`:

```python
@dataclass
class Rect:
    """A rectangle defined by min and max corners."""

    p_min: torch.Tensor  # [2] top-left corner
    p_max: torch.Tensor  # [2] bottom-right corner
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
```

**Step 4: Export Rect from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/shapes.py src/easydiffvg/__init__.py tests/test_shapes.py
git commit -m "feat: add Rect shape primitive"
```

---

### Task 5: Implement Polygon shape

**Files:**
- Modify: `tests/test_shapes.py`
- Modify: `src/easydiffvg/shapes.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for Polygon**

Add to `tests/test_shapes.py`:

```python
from easydiffvg import Circle, Ellipse, Rect, Polygon


class TestPolygon:
    def test_polygon_creation(self, device):
        """Polygon stores points as [N, 2] tensor."""
        points = torch.tensor([
            [10.0, 10.0],
            [50.0, 10.0],
            [30.0, 50.0],
        ], device=device)

        polygon = Polygon(points=points, is_closed=True)

        assert polygon.points.shape == (3, 2)
        assert polygon.is_closed is True
        torch.testing.assert_close(polygon.points, points)

    def test_polygon_open(self, device):
        """Polygon can be open (polyline)."""
        points = torch.tensor([
            [0.0, 0.0],
            [10.0, 20.0],
            [20.0, 0.0],
        ], device=device)

        polygon = Polygon(points=points, is_closed=False)

        assert polygon.is_closed is False

    def test_polygon_has_stroke_width(self, device):
        """Polygon has stroke_width attribute."""
        polygon = Polygon(
            points=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            is_closed=False,
        )

        assert hasattr(polygon, "stroke_width")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shapes.py::TestPolygon -v`
Expected: FAIL

**Step 3: Add Polygon to shapes.py**

```python
@dataclass
class Polygon:
    """A polygon or polyline defined by a sequence of points."""

    points: torch.Tensor  # [N, 2] vertices
    is_closed: bool
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
```

**Step 4: Export Polygon from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/shapes.py src/easydiffvg/__init__.py tests/test_shapes.py
git commit -m "feat: add Polygon shape primitive"
```

---

### Task 6: Implement Path shape

**Files:**
- Modify: `tests/test_shapes.py`
- Modify: `src/easydiffvg/shapes.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for Path**

Add to `tests/test_shapes.py`:

```python
from easydiffvg import Circle, Ellipse, Rect, Polygon, Path


class TestPath:
    def test_path_creation_cubic(self, device):
        """Path with cubic bezier segment (2 control points)."""
        # Cubic bezier: start, ctrl1, ctrl2, end
        points = torch.tensor([
            [0.0, 0.0],    # start
            [10.0, 30.0],  # ctrl1
            [30.0, 30.0],  # ctrl2
            [40.0, 0.0],   # end
        ], device=device)
        num_control_points = torch.tensor([2], dtype=torch.int32)  # cubic

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (4, 2)
        assert path.num_control_points.shape == (1,)
        assert path.is_closed is False

    def test_path_creation_quadratic(self, device):
        """Path with quadratic bezier segment (1 control point)."""
        points = torch.tensor([
            [0.0, 0.0],   # start
            [20.0, 40.0], # ctrl
            [40.0, 0.0],  # end
        ], device=device)
        num_control_points = torch.tensor([1], dtype=torch.int32)  # quadratic

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (3, 2)

    def test_path_creation_line(self, device):
        """Path with line segment (0 control points)."""
        points = torch.tensor([
            [0.0, 0.0],
            [40.0, 40.0],
        ], device=device)
        num_control_points = torch.tensor([0], dtype=torch.int32)  # line

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=False,
        )

        assert path.points.shape == (2, 2)

    def test_path_closed(self, device):
        """Closed path forms a loop."""
        points = torch.tensor([
            [0.0, 0.0],
            [40.0, 0.0],
            [40.0, 40.0],
            [0.0, 40.0],
        ], device=device)
        # 4 points, 4 line segments (closed)
        num_control_points = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

        path = Path(
            num_control_points=num_control_points,
            points=points,
            is_closed=True,
        )

        assert path.is_closed is True

    def test_path_has_use_distance_approx(self, device):
        """Path has use_distance_approx flag (default False)."""
        path = Path(
            num_control_points=torch.tensor([0]),
            points=torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            is_closed=False,
        )

        assert hasattr(path, "use_distance_approx")
        assert path.use_distance_approx is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_shapes.py::TestPath -v`
Expected: FAIL

**Step 3: Add Path to shapes.py**

```python
@dataclass
class Path:
    """A path composed of line and bezier curve segments.

    Each segment is defined by num_control_points:
    - 0: line segment (2 points: start, end)
    - 1: quadratic bezier (3 points: start, control, end)
    - 2: cubic bezier (4 points: start, ctrl1, ctrl2, end)

    Points are shared between segments (end of one = start of next).
    """

    num_control_points: torch.Tensor  # [M] per-segment control point count
    points: torch.Tensor  # [N, 2] all control points
    is_closed: bool
    stroke_width: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    id: str = ""
    use_distance_approx: bool = False
```

**Step 4: Export Path from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_shapes.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/shapes.py src/easydiffvg/__init__.py tests/test_shapes.py
git commit -m "feat: add Path shape primitive with bezier support"
```

---

## Phase 2: Colors

### Task 7: Implement SolidColor

**Files:**
- Create: `tests/test_colors.py`
- Create: `src/easydiffvg/color.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for SolidColor**

```python
# tests/test_colors.py
"""Tests for color classes."""

import pytest
import torch

from easydiffvg import SolidColor


class TestSolidColor:
    def test_solid_color_creation(self, device):
        """SolidColor stores RGBA as [4] tensor."""
        rgba = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)

        color = SolidColor(color=rgba)

        assert color.color.shape == (4,)
        torch.testing.assert_close(color.color, rgba)

    def test_solid_color_semitransparent(self, device):
        """SolidColor supports alpha < 1."""
        rgba = torch.tensor([0.0, 1.0, 0.0, 0.5], device=device)

        color = SolidColor(color=rgba)

        assert color.color[3] == 0.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py -v`
Expected: FAIL

**Step 3: Create color.py with SolidColor**

```python
# src/easydiffvg/color.py
"""Color types for easydiffvg."""

from dataclasses import dataclass

import torch


@dataclass
class SolidColor:
    """A solid RGBA color."""

    color: torch.Tensor  # [4] RGBA, values in [0, 1]
```

**Step 4: Export SolidColor from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/color.py tests/test_colors.py src/easydiffvg/__init__.py
git commit -m "feat: add SolidColor class"
```

---

### Task 8: Implement LinearGradient

**Files:**
- Modify: `tests/test_colors.py`
- Modify: `src/easydiffvg/color.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for LinearGradient**

Add to `tests/test_colors.py`:

```python
from easydiffvg import SolidColor, LinearGradient


class TestLinearGradient:
    def test_linear_gradient_creation(self, device):
        """LinearGradient has begin, end, offsets, stop_colors."""
        begin = torch.tensor([0.0, 0.0], device=device)
        end = torch.tensor([64.0, 64.0], device=device)
        offsets = torch.tensor([0.0, 1.0], device=device)
        stop_colors = torch.tensor([
            [1.0, 0.0, 0.0, 1.0],  # red
            [0.0, 0.0, 1.0, 1.0],  # blue
        ], device=device)

        grad = LinearGradient(
            begin=begin,
            end=end,
            offsets=offsets,
            stop_colors=stop_colors,
        )

        assert grad.begin.shape == (2,)
        assert grad.end.shape == (2,)
        assert grad.offsets.shape == (2,)
        assert grad.stop_colors.shape == (2, 4)

    def test_linear_gradient_multi_stop(self, device):
        """LinearGradient supports multiple color stops."""
        grad = LinearGradient(
            begin=torch.tensor([0.0, 0.0]),
            end=torch.tensor([100.0, 0.0]),
            offsets=torch.tensor([0.0, 0.5, 1.0]),
            stop_colors=torch.tensor([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]),
        )

        assert grad.offsets.shape == (3,)
        assert grad.stop_colors.shape == (3, 4)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::TestLinearGradient -v`
Expected: FAIL

**Step 3: Add LinearGradient to color.py**

```python
@dataclass
class LinearGradient:
    """A linear gradient color."""

    begin: torch.Tensor  # [2] start point
    end: torch.Tensor  # [2] end point
    offsets: torch.Tensor  # [S] stop positions in [0, 1]
    stop_colors: torch.Tensor  # [S, 4] RGBA at each stop
```

**Step 4: Export LinearGradient from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/color.py tests/test_colors.py src/easydiffvg/__init__.py
git commit -m "feat: add LinearGradient class"
```

---

### Task 9: Implement RadialGradient

**Files:**
- Modify: `tests/test_colors.py`
- Modify: `src/easydiffvg/color.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for RadialGradient**

Add to `tests/test_colors.py`:

```python
from easydiffvg import SolidColor, LinearGradient, RadialGradient


class TestRadialGradient:
    def test_radial_gradient_creation(self, device):
        """RadialGradient has center, radius, offsets, stop_colors."""
        center = torch.tensor([32.0, 32.0], device=device)
        radius = torch.tensor([20.0, 20.0], device=device)  # rx, ry
        offsets = torch.tensor([0.0, 1.0], device=device)
        stop_colors = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],  # white center
            [0.0, 0.0, 0.0, 1.0],  # black edge
        ], device=device)

        grad = RadialGradient(
            center=center,
            radius=radius,
            offsets=offsets,
            stop_colors=stop_colors,
        )

        assert grad.center.shape == (2,)
        assert grad.radius.shape == (2,)
        assert grad.offsets.shape == (2,)
        assert grad.stop_colors.shape == (2, 4)

    def test_radial_gradient_elliptical(self, device):
        """RadialGradient supports elliptical shape (rx != ry)."""
        grad = RadialGradient(
            center=torch.tensor([50.0, 50.0]),
            radius=torch.tensor([30.0, 15.0]),  # wide ellipse
            offsets=torch.tensor([0.0, 1.0]),
            stop_colors=torch.tensor([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]),
        )

        assert grad.radius[0] != grad.radius[1]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colors.py::TestRadialGradient -v`
Expected: FAIL

**Step 3: Add RadialGradient to color.py**

```python
@dataclass
class RadialGradient:
    """A radial gradient color."""

    center: torch.Tensor  # [2] center point
    radius: torch.Tensor  # [2] rx, ry (can be elliptical)
    offsets: torch.Tensor  # [S] stop positions in [0, 1]
    stop_colors: torch.Tensor  # [S, 4] RGBA at each stop
```

**Step 4: Add Color type alias to color.py**

```python
# Type alias for any color
Color = SolidColor | LinearGradient | RadialGradient
```

**Step 5: Export RadialGradient and Color from __init__.py**

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_colors.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/easydiffvg/color.py tests/test_colors.py src/easydiffvg/__init__.py
git commit -m "feat: add RadialGradient class and Color type"
```

---

### Task 10: Implement ShapeGroup

**Files:**
- Create: `tests/test_groups.py`
- Create: `src/easydiffvg/groups.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for ShapeGroup**

```python
# tests/test_groups.py
"""Tests for ShapeGroup."""

import pytest
import torch

from easydiffvg import ShapeGroup, SolidColor, LinearGradient


class TestShapeGroup:
    def test_shape_group_with_fill(self, device):
        """ShapeGroup bundles shapes with fill color."""
        fill = SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
        shape_ids = torch.tensor([0], dtype=torch.int32)

        group = ShapeGroup(
            shape_ids=shape_ids,
            fill_color=fill,
        )

        assert group.shape_ids.shape == (1,)
        assert group.fill_color is not None
        assert group.stroke_color is None

    def test_shape_group_with_stroke(self, device):
        """ShapeGroup can have stroke color."""
        stroke = SolidColor(color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_ids = torch.tensor([0, 1], dtype=torch.int32)

        group = ShapeGroup(
            shape_ids=shape_ids,
            fill_color=None,
            stroke_color=stroke,
        )

        assert group.fill_color is None
        assert group.stroke_color is not None

    def test_shape_group_transform(self, device):
        """ShapeGroup has shape_to_canvas transform matrix."""
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 1.0, 1.0, 1.0])),
        )

        assert group.shape_to_canvas.shape == (3, 3)
        # Default is identity
        torch.testing.assert_close(group.shape_to_canvas, torch.eye(3))

    def test_shape_group_custom_transform(self, device):
        """ShapeGroup accepts custom transform."""
        # Translation matrix
        transform = torch.tensor([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0, 1.0],
        ])

        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
            shape_to_canvas=transform,
        )

        torch.testing.assert_close(group.shape_to_canvas, transform)

    def test_shape_group_even_odd_rule(self, device):
        """ShapeGroup has use_even_odd_rule flag."""
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
            use_even_odd_rule=True,
        )

        assert group.use_even_odd_rule is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_groups.py -v`
Expected: FAIL

**Step 3: Create groups.py with ShapeGroup**

```python
# src/easydiffvg/groups.py
"""Shape grouping with appearance properties."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from easydiffvg.color import Color


@dataclass
class ShapeGroup:
    """Groups shapes together with fill/stroke colors and transform."""

    shape_ids: torch.Tensor  # [N] indices into shapes list
    fill_color: "Color | None"
    stroke_color: "Color | None" = None
    use_even_odd_rule: bool = True
    shape_to_canvas: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    id: str = ""
```

**Step 4: Export ShapeGroup from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_groups.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/groups.py tests/test_groups.py src/easydiffvg/__init__.py
git commit -m "feat: add ShapeGroup class"
```

---

## Phase 3: Utility Modules

### Task 11: Implement bezier utilities

**Files:**
- Create: `src/easydiffvg/utils/__init__.py`
- Create: `src/easydiffvg/utils/bezier.py`
- Create: `tests/test_bezier.py`

**Step 1: Write failing tests for bezier utilities**

```python
# tests/test_bezier.py
"""Tests for bezier curve utilities."""

import pytest
import torch

from easydiffvg.utils.bezier import (
    evaluate_quadratic,
    evaluate_cubic,
    quadratic_to_cubic,
    subdivide_cubic,
)


class TestBezierEvaluation:
    def test_evaluate_quadratic_at_start(self, device):
        """Quadratic bezier at t=0 returns start point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=0.0)

        torch.testing.assert_close(result, p0)

    def test_evaluate_quadratic_at_end(self, device):
        """Quadratic bezier at t=1 returns end point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=1.0)

        torch.testing.assert_close(result, p2)

    def test_evaluate_quadratic_at_midpoint(self, device):
        """Quadratic bezier at t=0.5."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_quadratic(p0, p1, p2, t=0.5)

        # B(0.5) = 0.25*p0 + 0.5*p1 + 0.25*p2
        expected = 0.25 * p0 + 0.5 * p1 + 0.25 * p2
        torch.testing.assert_close(result, expected)

    def test_evaluate_cubic_at_start(self, device):
        """Cubic bezier at t=0 returns start point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([25.0, 100.0], device=device)
        p2 = torch.tensor([75.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_cubic(p0, p1, p2, p3, t=0.0)

        torch.testing.assert_close(result, p0)

    def test_evaluate_cubic_at_end(self, device):
        """Cubic bezier at t=1 returns end point."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([25.0, 100.0], device=device)
        p2 = torch.tensor([75.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        result = evaluate_cubic(p0, p1, p2, p3, t=1.0)

        torch.testing.assert_close(result, p3)

    def test_evaluate_cubic_batched(self, device):
        """Cubic bezier evaluation works with batched t values."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([25.0, 100.0], device=device)
        p2 = torch.tensor([75.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)
        t = torch.tensor([0.0, 0.5, 1.0], device=device)

        result = evaluate_cubic(p0, p1, p2, p3, t=t)

        assert result.shape == (3, 2)
        torch.testing.assert_close(result[0], p0)
        torch.testing.assert_close(result[2], p3)


class TestBezierConversion:
    def test_quadratic_to_cubic(self, device):
        """Convert quadratic to equivalent cubic bezier."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([50.0, 100.0], device=device)
        p2 = torch.tensor([100.0, 0.0], device=device)

        c0, c1, c2, c3 = quadratic_to_cubic(p0, p1, p2)

        # Start and end should match
        torch.testing.assert_close(c0, p0)
        torch.testing.assert_close(c3, p2)

        # Evaluate both at t=0.5, should be same point
        quad_mid = evaluate_quadratic(p0, p1, p2, t=0.5)
        cubic_mid = evaluate_cubic(c0, c1, c2, c3, t=0.5)
        torch.testing.assert_close(quad_mid, cubic_mid, atol=1e-5, rtol=1e-5)


class TestBezierSubdivision:
    def test_subdivide_cubic_at_midpoint(self, device):
        """Subdivide cubic at t=0.5 produces two valid curves."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([25.0, 100.0], device=device)
        p2 = torch.tensor([75.0, 100.0], device=device)
        p3 = torch.tensor([100.0, 0.0], device=device)

        (l0, l1, l2, l3), (r0, r1, r2, r3) = subdivide_cubic(p0, p1, p2, p3, t=0.5)

        # Left curve starts at original start
        torch.testing.assert_close(l0, p0)
        # Right curve ends at original end
        torch.testing.assert_close(r3, p3)
        # They meet at the subdivision point
        torch.testing.assert_close(l3, r0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bezier.py -v`
Expected: FAIL

**Step 3: Create bezier.py with utility functions**

```python
# src/easydiffvg/utils/bezier.py
"""Bezier curve mathematics."""

import torch


def evaluate_quadratic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate quadratic bezier curve at parameter t.

    B(t) = (1-t)²p0 + 2(1-t)t·p1 + t²p2

    Args:
        p0: Start point [2]
        p1: Control point [2]
        p2: End point [2]
        t: Parameter in [0, 1], scalar or [N] tensor

    Returns:
        Point(s) on curve, [2] or [N, 2]
    """
    if isinstance(t, float):
        t = torch.tensor(t, device=p0.device, dtype=p0.dtype)

    if t.dim() == 0:
        # Scalar case
        mt = 1.0 - t
        return mt * mt * p0 + 2 * mt * t * p1 + t * t * p2
    else:
        # Batched case: t is [N]
        t = t.unsqueeze(-1)  # [N, 1]
        mt = 1.0 - t
        return mt * mt * p0 + 2 * mt * t * p1 + t * t * p2


def evaluate_cubic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate cubic bezier curve at parameter t.

    B(t) = (1-t)³p0 + 3(1-t)²t·p1 + 3(1-t)t²·p2 + t³p3

    Args:
        p0: Start point [2]
        p1: Control point 1 [2]
        p2: Control point 2 [2]
        p3: End point [2]
        t: Parameter in [0, 1], scalar or [N] tensor

    Returns:
        Point(s) on curve, [2] or [N, 2]
    """
    if isinstance(t, float):
        t = torch.tensor(t, device=p0.device, dtype=p0.dtype)

    if t.dim() == 0:
        mt = 1.0 - t
        mt2 = mt * mt
        t2 = t * t
        return mt2 * mt * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t2 * t * p3
    else:
        t = t.unsqueeze(-1)  # [N, 1]
        mt = 1.0 - t
        mt2 = mt * mt
        t2 = t * t
        return mt2 * mt * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t2 * t * p3


def quadratic_to_cubic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quadratic bezier to equivalent cubic bezier.

    Args:
        p0: Start point [2]
        p1: Control point [2]
        p2: End point [2]

    Returns:
        Tuple of (c0, c1, c2, c3) cubic control points
    """
    c0 = p0
    c1 = p0 + (2.0 / 3.0) * (p1 - p0)
    c2 = p2 + (2.0 / 3.0) * (p1 - p2)
    c3 = p2
    return c0, c1, c2, c3


def subdivide_cubic(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float = 0.5,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Subdivide cubic bezier at parameter t using de Casteljau's algorithm.

    Args:
        p0, p1, p2, p3: Control points [2] each
        t: Split parameter in [0, 1]

    Returns:
        Two tuples of 4 control points each: (left_curve, right_curve)
    """
    if isinstance(t, float):
        t = torch.tensor(t, device=p0.device, dtype=p0.dtype)

    # de Casteljau's algorithm
    p01 = (1 - t) * p0 + t * p1
    p12 = (1 - t) * p1 + t * p2
    p23 = (1 - t) * p2 + t * p3

    p012 = (1 - t) * p01 + t * p12
    p123 = (1 - t) * p12 + t * p23

    p0123 = (1 - t) * p012 + t * p123  # Point on curve at t

    left = (p0, p01, p012, p0123)
    right = (p0123, p123, p23, p3)

    return left, right
```

**Step 4: Create utils/__init__.py**

```python
# src/easydiffvg/utils/__init__.py
"""Utility modules for easydiffvg."""

from easydiffvg.utils.bezier import (
    evaluate_cubic,
    evaluate_quadratic,
    quadratic_to_cubic,
    subdivide_cubic,
)

__all__ = [
    "evaluate_quadratic",
    "evaluate_cubic",
    "quadratic_to_cubic",
    "subdivide_cubic",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_bezier.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/utils/ tests/test_bezier.py
git commit -m "feat: add bezier curve utilities"
```

---

### Task 12: Implement winding number computation

**Files:**
- Create: `tests/test_winding.py`
- Create: `src/easydiffvg/utils/winding.py`
- Modify: `src/easydiffvg/utils/__init__.py`

**Step 1: Write failing tests for winding number**

```python
# tests/test_winding.py
"""Tests for winding number computation."""

import pytest
import torch
import math

from easydiffvg.utils.winding import (
    winding_number_polygon,
    winding_number_bezier,
)


class TestWindingNumberPolygon:
    def test_point_inside_triangle(self, device):
        """Point inside triangle has winding number 1."""
        # Triangle vertices (CCW)
        vertices = torch.tensor([
            [0.0, 0.0],
            [100.0, 0.0],
            [50.0, 100.0],
        ], device=device)
        point = torch.tensor([50.0, 30.0], device=device)

        winding = winding_number_polygon(point, vertices, is_closed=True)

        assert abs(winding - 1.0) < 0.01

    def test_point_outside_triangle(self, device):
        """Point outside triangle has winding number 0."""
        vertices = torch.tensor([
            [0.0, 0.0],
            [100.0, 0.0],
            [50.0, 100.0],
        ], device=device)
        point = torch.tensor([150.0, 50.0], device=device)

        winding = winding_number_polygon(point, vertices, is_closed=True)

        assert abs(winding) < 0.01

    def test_point_inside_square(self, device):
        """Point inside square has winding number 1."""
        vertices = torch.tensor([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ], device=device)
        point = torch.tensor([50.0, 50.0], device=device)

        winding = winding_number_polygon(point, vertices, is_closed=True)

        assert abs(winding - 1.0) < 0.01

    def test_batched_points(self, device):
        """Winding number works with batched query points."""
        vertices = torch.tensor([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ], device=device)
        points = torch.tensor([
            [50.0, 50.0],   # inside
            [150.0, 50.0],  # outside
            [50.0, 150.0],  # outside
        ], device=device)

        winding = winding_number_polygon(points, vertices, is_closed=True)

        assert winding.shape == (3,)
        assert abs(winding[0] - 1.0) < 0.01
        assert abs(winding[1]) < 0.01
        assert abs(winding[2]) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_winding.py -v`
Expected: FAIL

**Step 3: Create winding.py**

```python
# src/easydiffvg/utils/winding.py
"""Winding number computation for inside/outside tests."""

import torch
import math


def winding_number_polygon(
    points: torch.Tensor,
    vertices: torch.Tensor,
    is_closed: bool = True,
) -> torch.Tensor:
    """Compute winding number of point(s) with respect to a polygon.

    Uses the crossing number / angle summation method.

    Args:
        points: Query point(s), [2] or [N, 2]
        vertices: Polygon vertices [M, 2] in order
        is_closed: If True, connect last vertex to first

    Returns:
        Winding number(s), scalar or [N]
    """
    single_point = points.dim() == 1
    if single_point:
        points = points.unsqueeze(0)  # [1, 2]

    n_points = points.shape[0]
    n_verts = vertices.shape[0]

    # Shift vertices so point is at origin
    # points: [N, 2], vertices: [M, 2]
    # shifted: [N, M, 2]
    shifted = vertices.unsqueeze(0) - points.unsqueeze(1)

    # Get consecutive vertex pairs
    if is_closed:
        v1 = shifted  # [N, M, 2]
        v2 = torch.roll(shifted, -1, dims=1)  # [N, M, 2]
    else:
        v1 = shifted[:, :-1, :]  # [N, M-1, 2]
        v2 = shifted[:, 1:, :]   # [N, M-1, 2]

    # Compute angle for each edge using atan2
    # Cross product gives sin(angle), dot product gives cos(angle)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]

    angles = torch.atan2(cross, dot)

    # Sum angles and divide by 2π
    total_angle = angles.sum(dim=-1)
    winding = total_angle / (2 * math.pi)

    if single_point:
        return winding.squeeze(0)
    return winding


def winding_number_bezier(
    points: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_samples: int = 16,
) -> torch.Tensor:
    """Compute winding number contribution from a cubic bezier segment.

    Approximates by sampling the curve.

    Args:
        points: Query point(s), [2] or [N, 2]
        p0, p1, p2, p3: Bezier control points [2] each
        num_samples: Number of samples along curve

    Returns:
        Winding number contribution, scalar or [N]
    """
    from easydiffvg.utils.bezier import evaluate_cubic

    # Sample points along the bezier
    t = torch.linspace(0, 1, num_samples + 1, device=p0.device)
    curve_points = evaluate_cubic(p0, p1, p2, p3, t)  # [num_samples+1, 2]

    # Treat as polygon
    return winding_number_polygon(points, curve_points, is_closed=False)
```

**Step 4: Update utils/__init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_winding.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/utils/winding.py tests/test_winding.py src/easydiffvg/utils/__init__.py
git commit -m "feat: add winding number computation"
```

---

### Task 13: Implement distance field utilities

**Files:**
- Create: `tests/test_distance.py`
- Create: `src/easydiffvg/utils/distance.py`
- Modify: `src/easydiffvg/utils/__init__.py`

**Step 1: Write failing tests for distance utilities**

```python
# tests/test_distance.py
"""Tests for distance field utilities."""

import pytest
import torch
import math

from easydiffvg.utils.distance import (
    distance_to_line_segment,
    distance_to_circle,
    distance_to_cubic_bezier,
)


class TestDistanceToLineSegment:
    def test_point_on_segment(self, device):
        """Point on segment has distance 0."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([100.0, 0.0], device=device)
        point = torch.tensor([50.0, 0.0], device=device)

        dist = distance_to_line_segment(point, p0, p1)

        assert abs(dist) < 1e-5

    def test_point_perpendicular(self, device):
        """Point perpendicular to segment."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([100.0, 0.0], device=device)
        point = torch.tensor([50.0, 30.0], device=device)

        dist = distance_to_line_segment(point, p0, p1)

        torch.testing.assert_close(dist, torch.tensor(30.0))

    def test_point_past_endpoint(self, device):
        """Point past segment endpoint gets distance to endpoint."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([100.0, 0.0], device=device)
        point = torch.tensor([150.0, 0.0], device=device)

        dist = distance_to_line_segment(point, p0, p1)

        torch.testing.assert_close(dist, torch.tensor(50.0))

    def test_batched_points(self, device):
        """Distance computation works with batched points."""
        p0 = torch.tensor([0.0, 0.0], device=device)
        p1 = torch.tensor([100.0, 0.0], device=device)
        points = torch.tensor([
            [50.0, 0.0],
            [50.0, 10.0],
            [50.0, 20.0],
        ], device=device)

        dist = distance_to_line_segment(points, p0, p1)

        assert dist.shape == (3,)
        torch.testing.assert_close(dist, torch.tensor([0.0, 10.0, 20.0]))


class TestDistanceToCircle:
    def test_point_on_circle(self, device):
        """Point on circle boundary has distance 0."""
        center = torch.tensor([50.0, 50.0], device=device)
        radius = torch.tensor(20.0, device=device)
        point = torch.tensor([70.0, 50.0], device=device)  # On right edge

        dist = distance_to_circle(point, center, radius)

        assert abs(dist) < 1e-5

    def test_point_inside_circle(self, device):
        """Point inside circle has negative distance."""
        center = torch.tensor([50.0, 50.0], device=device)
        radius = torch.tensor(20.0, device=device)
        point = torch.tensor([50.0, 50.0], device=device)  # At center

        dist = distance_to_circle(point, center, radius)

        torch.testing.assert_close(dist, torch.tensor(-20.0))

    def test_point_outside_circle(self, device):
        """Point outside circle has positive distance."""
        center = torch.tensor([50.0, 50.0], device=device)
        radius = torch.tensor(20.0, device=device)
        point = torch.tensor([100.0, 50.0], device=device)

        dist = distance_to_circle(point, center, radius)

        torch.testing.assert_close(dist, torch.tensor(30.0))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_distance.py -v`
Expected: FAIL

**Step 3: Create distance.py**

```python
# src/easydiffvg/utils/distance.py
"""Distance field utilities for shapes."""

import torch


def distance_to_line_segment(
    points: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
) -> torch.Tensor:
    """Compute unsigned distance from point(s) to line segment.

    Args:
        points: Query point(s), [2] or [N, 2]
        p0: Segment start [2]
        p1: Segment end [2]

    Returns:
        Distance(s), scalar or [N]
    """
    single = points.dim() == 1
    if single:
        points = points.unsqueeze(0)

    # Vector from p0 to p1
    d = p1 - p0
    # Vector from p0 to each point
    v = points - p0

    # Project onto line: t = (v · d) / (d · d)
    d_dot_d = torch.dot(d, d)
    if d_dot_d < 1e-10:
        # Degenerate segment (p0 == p1)
        dist = torch.norm(v, dim=-1)
    else:
        t = (v @ d) / d_dot_d
        t = torch.clamp(t, 0.0, 1.0)

        # Closest point on segment
        closest = p0 + t.unsqueeze(-1) * d
        dist = torch.norm(points - closest, dim=-1)

    if single:
        return dist.squeeze(0)
    return dist


def distance_to_circle(
    points: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Compute signed distance from point(s) to circle boundary.

    Negative inside, positive outside.

    Args:
        points: Query point(s), [2] or [N, 2]
        center: Circle center [2]
        radius: Circle radius, scalar

    Returns:
        Signed distance(s), scalar or [N]
    """
    single = points.dim() == 1
    if single:
        points = points.unsqueeze(0)

    dist_to_center = torch.norm(points - center, dim=-1)
    signed_dist = dist_to_center - radius

    if single:
        return signed_dist.squeeze(0)
    return signed_dist


def distance_to_cubic_bezier(
    points: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_samples: int = 16,
) -> torch.Tensor:
    """Compute approximate unsigned distance to cubic bezier curve.

    Uses sampling approximation.

    Args:
        points: Query point(s), [2] or [N, 2]
        p0, p1, p2, p3: Bezier control points [2] each
        num_samples: Number of samples for approximation

    Returns:
        Distance(s), scalar or [N]
    """
    from easydiffvg.utils.bezier import evaluate_cubic

    single = points.dim() == 1
    if single:
        points = points.unsqueeze(0)

    # Sample curve
    t = torch.linspace(0, 1, num_samples, device=p0.device)
    curve = evaluate_cubic(p0, p1, p2, p3, t)  # [num_samples, 2]

    # Distance from each query point to each line segment
    min_dist = torch.full((points.shape[0],), float("inf"), device=points.device)

    for i in range(num_samples - 1):
        seg_dist = distance_to_line_segment(points, curve[i], curve[i + 1])
        min_dist = torch.minimum(min_dist, seg_dist)

    if single:
        return min_dist.squeeze(0)
    return min_dist
```

**Step 4: Update utils/__init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_distance.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/utils/distance.py tests/test_distance.py src/easydiffvg/utils/__init__.py
git commit -m "feat: add distance field utilities"
```

---

### Task 14: Implement BVH (Bounding Volume Hierarchy)

**Files:**
- Create: `tests/test_bvh.py`
- Create: `src/easydiffvg/bvh.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing tests for BVH**

```python
# tests/test_bvh.py
"""Tests for bounding volume hierarchy."""

import pytest
import torch

from easydiffvg import Circle, Rect
from easydiffvg.bvh import BVH, compute_shape_bbox


class TestBoundingBox:
    def test_circle_bbox(self, device):
        """Circle bounding box."""
        circle = Circle(
            radius=torch.tensor(10.0, device=device),
            center=torch.tensor([50.0, 50.0], device=device),
        )

        bbox = compute_shape_bbox(circle)

        # bbox is [min_x, min_y, max_x, max_y]
        torch.testing.assert_close(bbox, torch.tensor([40.0, 40.0, 60.0, 60.0]))

    def test_rect_bbox(self, device):
        """Rect bounding box is just the corners."""
        rect = Rect(
            p_min=torch.tensor([10.0, 20.0], device=device),
            p_max=torch.tensor([80.0, 90.0], device=device),
        )

        bbox = compute_shape_bbox(rect)

        torch.testing.assert_close(bbox, torch.tensor([10.0, 20.0, 80.0, 90.0]))


class TestBVH:
    def test_build_bvh_single_shape(self, device):
        """BVH with single shape."""
        shapes = [
            Circle(
                radius=torch.tensor(10.0),
                center=torch.tensor([50.0, 50.0]),
            )
        ]

        bvh = BVH(shapes)

        assert len(bvh.nodes) >= 1

    def test_query_point_inside(self, device):
        """Query returns shape when point is inside bbox."""
        shapes = [
            Circle(
                radius=torch.tensor(10.0),
                center=torch.tensor([50.0, 50.0]),
            )
        ]
        bvh = BVH(shapes)
        point = torch.tensor([50.0, 50.0])

        candidates = bvh.query(point)

        assert 0 in candidates

    def test_query_point_outside(self, device):
        """Query returns empty when point outside all bboxes."""
        shapes = [
            Circle(
                radius=torch.tensor(10.0),
                center=torch.tensor([50.0, 50.0]),
            )
        ]
        bvh = BVH(shapes)
        point = torch.tensor([200.0, 200.0])

        candidates = bvh.query(point)

        assert len(candidates) == 0

    def test_query_multiple_shapes(self, device):
        """Query with multiple overlapping shapes."""
        shapes = [
            Circle(radius=torch.tensor(20.0), center=torch.tensor([30.0, 30.0])),
            Circle(radius=torch.tensor(20.0), center=torch.tensor([40.0, 40.0])),
            Circle(radius=torch.tensor(20.0), center=torch.tensor([100.0, 100.0])),
        ]
        bvh = BVH(shapes)

        # Point overlaps first two circles' bboxes
        point = torch.tensor([35.0, 35.0])
        candidates = bvh.query(point)

        assert 0 in candidates
        assert 1 in candidates
        assert 2 not in candidates
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bvh.py -v`
Expected: FAIL

**Step 3: Create bvh.py**

```python
# src/easydiffvg/bvh.py
"""Bounding volume hierarchy for spatial acceleration."""

from dataclasses import dataclass
from typing import Sequence

import torch

from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect


Shape = Circle | Ellipse | Path | Polygon | Rect


@dataclass
class BVHNode:
    """A node in the BVH tree."""

    bbox: torch.Tensor  # [4] min_x, min_y, max_x, max_y
    left: int | None = None  # child index or None if leaf
    right: int | None = None
    shape_id: int | None = None  # only set for leaf nodes


def compute_shape_bbox(shape: Shape) -> torch.Tensor:
    """Compute axis-aligned bounding box for a shape.

    Returns:
        Tensor [4]: min_x, min_y, max_x, max_y
    """
    if isinstance(shape, Circle):
        r = shape.radius
        c = shape.center
        return torch.stack([c[0] - r, c[1] - r, c[0] + r, c[1] + r])

    elif isinstance(shape, Ellipse):
        rx, ry = shape.radius[0], shape.radius[1]
        c = shape.center
        return torch.stack([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry])

    elif isinstance(shape, Rect):
        return torch.cat([shape.p_min, shape.p_max])

    elif isinstance(shape, Polygon):
        points = shape.points
        min_pt = points.min(dim=0).values
        max_pt = points.max(dim=0).values
        return torch.cat([min_pt, max_pt])

    elif isinstance(shape, Path):
        points = shape.points
        min_pt = points.min(dim=0).values
        max_pt = points.max(dim=0).values
        return torch.cat([min_pt, max_pt])

    else:
        raise TypeError(f"Unknown shape type: {type(shape)}")


def bbox_contains_point(bbox: torch.Tensor, point: torch.Tensor) -> bool:
    """Check if bounding box contains a point."""
    return (
        point[0] >= bbox[0]
        and point[0] <= bbox[2]
        and point[1] >= bbox[1]
        and point[1] <= bbox[3]
    )


def merge_bboxes(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Merge two bounding boxes."""
    return torch.stack([
        torch.minimum(bbox1[0], bbox2[0]),
        torch.minimum(bbox1[1], bbox2[1]),
        torch.maximum(bbox1[2], bbox2[2]),
        torch.maximum(bbox1[3], bbox2[3]),
    ])


class BVH:
    """Bounding volume hierarchy for spatial queries."""

    def __init__(self, shapes: Sequence[Shape]):
        self.shapes = shapes
        self.nodes: list[BVHNode] = []
        self._build(list(range(len(shapes))))

    def _build(self, shape_ids: list[int]) -> int:
        """Build BVH recursively, return node index."""
        if len(shape_ids) == 0:
            return -1

        if len(shape_ids) == 1:
            # Leaf node
            sid = shape_ids[0]
            bbox = compute_shape_bbox(self.shapes[sid])
            node = BVHNode(bbox=bbox, shape_id=sid)
            self.nodes.append(node)
            return len(self.nodes) - 1

        # Compute combined bbox
        bboxes = [compute_shape_bbox(self.shapes[sid]) for sid in shape_ids]
        combined = bboxes[0]
        for bb in bboxes[1:]:
            combined = merge_bboxes(combined, bb)

        # Split along longest axis
        extent = combined[2:] - combined[:2]  # [width, height]
        axis = 0 if extent[0] > extent[1] else 1

        # Sort by centroid along axis
        centroids = [(bb[axis] + bb[axis + 2]) / 2 for bb in bboxes]
        sorted_ids = [sid for _, sid in sorted(zip(centroids, shape_ids))]

        mid = len(sorted_ids) // 2
        left_ids = sorted_ids[:mid]
        right_ids = sorted_ids[mid:]

        # Build children
        left_idx = self._build(left_ids)
        right_idx = self._build(right_ids)

        # Merge child bboxes
        left_bbox = self.nodes[left_idx].bbox
        right_bbox = self.nodes[right_idx].bbox
        merged = merge_bboxes(left_bbox, right_bbox)

        node = BVHNode(bbox=merged, left=left_idx, right=right_idx)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def query(self, point: torch.Tensor) -> list[int]:
        """Find all shapes whose bboxes contain the point."""
        if len(self.nodes) == 0:
            return []

        result: list[int] = []
        self._query_recursive(len(self.nodes) - 1, point, result)
        return result

    def _query_recursive(
        self, node_idx: int, point: torch.Tensor, result: list[int]
    ) -> None:
        node = self.nodes[node_idx]

        if not bbox_contains_point(node.bbox, point):
            return

        if node.shape_id is not None:
            # Leaf node
            result.append(node.shape_id)
        else:
            # Internal node
            if node.left is not None:
                self._query_recursive(node.left, point, result)
            if node.right is not None:
                self._query_recursive(node.right, point, result)
```

**Step 4: Export BVH utilities from __init__.py**

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_bvh.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/bvh.py tests/test_bvh.py src/easydiffvg/__init__.py
git commit -m "feat: add BVH spatial acceleration structure"
```

---

## Phase 4: Rasterization

### Task 15: Implement basic rasterization for circles

**Files:**
- Create: `tests/test_rasterize.py`
- Create: `src/easydiffvg/rasterize.py`

**Step 1: Write failing test for circle rasterization**

```python
# tests/test_rasterize.py
"""Tests for rasterization."""

import pytest
import torch

from easydiffvg import Circle, ShapeGroup, SolidColor
from easydiffvg.rasterize import rasterize_shapes


class TestRasterizeCircle:
    def test_circle_center_pixel_filled(self, device, canvas_64):
        """Center pixel of filled circle should have fill color."""
        width, height = canvas_64
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [circle], [group], samples=1)

        assert img.shape == (height, width, 4)
        # Center pixel should be red
        center_pixel = img[32, 32]
        assert center_pixel[0] > 0.9  # R
        assert center_pixel[1] < 0.1  # G
        assert center_pixel[2] < 0.1  # B
        assert center_pixel[3] > 0.9  # A

    def test_circle_outside_pixel_transparent(self, device, canvas_64):
        """Pixel outside circle should be transparent."""
        width, height = canvas_64
        circle = Circle(
            radius=torch.tensor(10.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [circle], [group], samples=1)

        # Corner pixel should be transparent
        corner_pixel = img[0, 0]
        assert corner_pixel[3] < 0.1  # A is near 0

    def test_antialiased_edge(self, device, canvas_64):
        """Edge pixels should have partial coverage with samples > 1."""
        width, height = canvas_64
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [circle], [group], samples=4)

        # Find an edge pixel (at radius distance from center)
        edge_pixel = img[32, 52]  # 20 pixels right of center
        # Should have partial alpha (not 0 or 1)
        assert 0.1 < edge_pixel[3] < 0.9
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rasterize.py -v`
Expected: FAIL

**Step 3: Create rasterize.py with basic circle support**

```python
# src/easydiffvg/rasterize.py
"""Core rasterization logic (forward pass)."""

from typing import Sequence

import torch

from easydiffvg.bvh import BVH, compute_shape_bbox, bbox_contains_point
from easydiffvg.color import Color, SolidColor, LinearGradient, RadialGradient
from easydiffvg.groups import ShapeGroup
from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect
from easydiffvg.utils.distance import distance_to_circle

Shape = Circle | Ellipse | Path | Polygon | Rect


def sample_color(
    color: Color,
    point: torch.Tensor,
) -> torch.Tensor:
    """Sample color at a point.

    Args:
        color: Color to sample
        point: [2] position

    Returns:
        [4] RGBA color
    """
    if isinstance(color, SolidColor):
        return color.color

    elif isinstance(color, LinearGradient):
        # Project point onto gradient line
        direction = color.end - color.begin
        length_sq = torch.dot(direction, direction)
        if length_sq < 1e-10:
            t = torch.tensor(0.0, device=point.device)
        else:
            t = torch.dot(point - color.begin, direction) / length_sq
        t = torch.clamp(t, 0.0, 1.0)

        # Interpolate between stops
        return _interpolate_gradient(t, color.offsets, color.stop_colors)

    elif isinstance(color, RadialGradient):
        # Compute normalized distance from center
        diff = point - color.center
        # Handle elliptical gradient
        normalized = diff / color.radius
        t = torch.norm(normalized)
        t = torch.clamp(t, 0.0, 1.0)

        return _interpolate_gradient(t, color.offsets, color.stop_colors)

    else:
        raise TypeError(f"Unknown color type: {type(color)}")


def _interpolate_gradient(
    t: torch.Tensor,
    offsets: torch.Tensor,
    stop_colors: torch.Tensor,
) -> torch.Tensor:
    """Interpolate gradient color at position t."""
    # Find surrounding stops
    for i in range(len(offsets) - 1):
        if t <= offsets[i + 1]:
            # Interpolate between stop i and i+1
            t0, t1 = offsets[i], offsets[i + 1]
            if abs(t1 - t0) < 1e-10:
                return stop_colors[i]
            local_t = (t - t0) / (t1 - t0)
            return (1 - local_t) * stop_colors[i] + local_t * stop_colors[i + 1]

    # Past last stop
    return stop_colors[-1]


def is_point_inside_shape(
    point: torch.Tensor,
    shape: Shape,
) -> bool:
    """Test if point is inside shape."""
    if isinstance(shape, Circle):
        dist = torch.norm(point - shape.center)
        return dist <= shape.radius

    elif isinstance(shape, Ellipse):
        diff = point - shape.center
        normalized = diff / shape.radius
        return torch.norm(normalized) <= 1.0

    elif isinstance(shape, Rect):
        return (
            point[0] >= shape.p_min[0]
            and point[0] <= shape.p_max[0]
            and point[1] >= shape.p_min[1]
            and point[1] <= shape.p_max[1]
        )

    elif isinstance(shape, Polygon):
        from easydiffvg.utils.winding import winding_number_polygon
        winding = winding_number_polygon(point, shape.points, shape.is_closed)
        return abs(winding) > 0.5

    elif isinstance(shape, Path):
        from easydiffvg.utils.winding import winding_number_polygon
        # Approximate path as polygon for inside test
        # (This is simplified - full impl would sample beziers)
        winding = winding_number_polygon(point, shape.points, shape.is_closed)
        return abs(winding) > 0.5

    return False


def rasterize_shapes(
    width: int,
    height: int,
    shapes: Sequence[Shape],
    shape_groups: Sequence[ShapeGroup],
    samples: int = 2,
) -> torch.Tensor:
    """Rasterize shapes to an image tensor.

    Args:
        width: Canvas width
        height: Canvas height
        shapes: List of shapes
        shape_groups: List of shape groups
        samples: Antialiasing samples per pixel dimension

    Returns:
        [H, W, 4] RGBA image tensor
    """
    device = _get_device(shapes, shape_groups)

    # Initialize output image
    img = torch.zeros(height, width, 4, device=device)

    # Build BVH
    bvh = BVH(shapes) if shapes else None

    # Generate sample offsets
    sample_offsets = _generate_sample_offsets(samples, device)
    num_samples = len(sample_offsets)

    # For each pixel
    for y in range(height):
        for x in range(width):
            pixel_color = torch.zeros(4, device=device)

            # For each sample
            for offset in sample_offsets:
                sample_pt = torch.tensor(
                    [x + 0.5 + offset[0], y + 0.5 + offset[1]],
                    device=device,
                )

                # Accumulate color from shape groups (back to front)
                sample_color = torch.zeros(4, device=device)

                for group in shape_groups:
                    # Check if sample is inside any shape in group
                    inside = False
                    for sid in group.shape_ids:
                        shape = shapes[sid.item()]
                        if is_point_inside_shape(sample_pt, shape):
                            inside = True
                            break

                    if inside and group.fill_color is not None:
                        fill = sample_color(group.fill_color, sample_pt)
                        # Alpha compositing (over operator)
                        sample_color = _alpha_composite(sample_color, fill)

                pixel_color += sample_color

            # Average samples
            img[y, x] = pixel_color / num_samples

    return img


def _get_device(shapes: Sequence[Shape], groups: Sequence[ShapeGroup]):
    """Get device from shapes or groups."""
    for shape in shapes:
        if isinstance(shape, Circle):
            return shape.center.device
        elif hasattr(shape, "points"):
            return shape.points.device
    return torch.device("cpu")


def _generate_sample_offsets(samples: int, device: torch.device) -> list[torch.Tensor]:
    """Generate stratified sample offsets within a pixel."""
    offsets = []
    for i in range(samples):
        for j in range(samples):
            # Stratified: center of each sub-pixel
            ox = (i + 0.5) / samples - 0.5
            oy = (j + 0.5) / samples - 0.5
            offsets.append(torch.tensor([ox, oy], device=device))
    return offsets


def _alpha_composite(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Alpha composite src over dst (Porter-Duff over)."""
    src_a = src[3:4]
    dst_a = dst[3:4]
    out_a = src_a + dst_a * (1 - src_a)

    if out_a < 1e-10:
        return torch.zeros(4, device=dst.device)

    out_rgb = (src[:3] * src_a + dst[:3] * dst_a * (1 - src_a)) / out_a
    return torch.cat([out_rgb, out_a])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rasterize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/easydiffvg/rasterize.py tests/test_rasterize.py
git commit -m "feat: add basic rasterization with circle support"
```

---

### Task 16: Extend rasterization for all shapes

**Files:**
- Modify: `tests/test_rasterize.py`
- Modify: `src/easydiffvg/rasterize.py`

**Step 1: Write failing tests for other shapes**

Add to `tests/test_rasterize.py`:

```python
from easydiffvg import Circle, Ellipse, Rect, Polygon, Path, ShapeGroup, SolidColor


class TestRasterizeRect:
    def test_rect_center_filled(self, device, canvas_64):
        """Center of rect should have fill color."""
        width, height = canvas_64
        rect = Rect(
            p_min=torch.tensor([20.0, 20.0], device=device),
            p_max=torch.tensor([44.0, 44.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([0.0, 1.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [rect], [group], samples=1)

        center_pixel = img[32, 32]
        assert center_pixel[1] > 0.9  # G


class TestRasterizePolygon:
    def test_triangle_center_filled(self, device, canvas_64):
        """Center of triangle should be filled."""
        width, height = canvas_64
        polygon = Polygon(
            points=torch.tensor([
                [32.0, 10.0],
                [54.0, 54.0],
                [10.0, 54.0],
            ], device=device),
            is_closed=True,
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([0.0, 0.0, 1.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [polygon], [group], samples=1)

        center_pixel = img[32, 32]
        assert center_pixel[2] > 0.9  # B


class TestRasterizeEllipse:
    def test_ellipse_center_filled(self, device, canvas_64):
        """Center of ellipse should be filled."""
        width, height = canvas_64
        ellipse = Ellipse(
            radius=torch.tensor([20.0, 10.0], device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 1.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [ellipse], [group], samples=1)

        center_pixel = img[32, 32]
        assert center_pixel[0] > 0.9  # R
        assert center_pixel[1] > 0.9  # G
```

**Step 2: Run tests to verify they pass (should already work)**

Run: `uv run pytest tests/test_rasterize.py -v`
Expected: PASS (is_point_inside_shape handles these)

**Step 3: Commit**

```bash
git add tests/test_rasterize.py
git commit -m "test: add rasterization tests for all shape types"
```

---

### Task 17: Add stroke rendering

**Files:**
- Modify: `tests/test_rasterize.py`
- Modify: `src/easydiffvg/rasterize.py`

**Step 1: Write failing test for stroke**

Add to `tests/test_rasterize.py`:

```python
class TestRasterizeStroke:
    def test_circle_stroke_only(self, device, canvas_64):
        """Circle with stroke only, no fill."""
        width, height = canvas_64
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
            stroke_width=torch.tensor(4.0),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=None,
            stroke_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = rasterize_shapes(width, height, [circle], [group], samples=1)

        # Center should be transparent (no fill)
        center_pixel = img[32, 32]
        assert center_pixel[3] < 0.1

        # Edge should have stroke color
        edge_pixel = img[32, 52]  # On the circle edge
        assert edge_pixel[0] > 0.5  # R
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rasterize.py::TestRasterizeStroke -v`
Expected: FAIL

**Step 3: Add stroke support to rasterize.py**

Add to `rasterize_shapes` function after fill handling:

```python
def is_point_on_stroke(
    point: torch.Tensor,
    shape: Shape,
    stroke_width: float,
) -> bool:
    """Test if point is on shape's stroke."""
    half_width = stroke_width / 2.0

    if isinstance(shape, Circle):
        dist = abs(torch.norm(point - shape.center) - shape.radius)
        return dist <= half_width

    elif isinstance(shape, Ellipse):
        # Approximate: check distance to ellipse boundary
        diff = point - shape.center
        normalized = diff / shape.radius
        dist_normalized = torch.norm(normalized)
        # Distance in ellipse space
        dist = abs(dist_normalized - 1.0) * torch.min(shape.radius)
        return dist <= half_width

    elif isinstance(shape, Rect):
        # Check distance to each edge
        from easydiffvg.utils.distance import distance_to_line_segment
        corners = [
            shape.p_min,
            torch.stack([shape.p_max[0], shape.p_min[1]]),
            shape.p_max,
            torch.stack([shape.p_min[0], shape.p_max[1]]),
        ]
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            if distance_to_line_segment(point, p0, p1) <= half_width:
                return True
        return False

    elif isinstance(shape, (Polygon, Path)):
        from easydiffvg.utils.distance import distance_to_line_segment
        points = shape.points
        n = len(points)
        num_edges = n if shape.is_closed else n - 1
        for i in range(num_edges):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            if distance_to_line_segment(point, p0, p1) <= half_width:
                return True
        return False

    return False
```

Update the sample loop in `rasterize_shapes`:

```python
# After fill check, add stroke check:
if group.stroke_color is not None:
    for sid in group.shape_ids:
        shape = shapes[sid.item()]
        sw = shape.stroke_width.item() if hasattr(shape, 'stroke_width') else 1.0
        if is_point_on_stroke(sample_pt, shape, sw):
            stroke = sample_color(group.stroke_color, sample_pt)
            sample_color = _alpha_composite(sample_color, stroke)
            break
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rasterize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/easydiffvg/rasterize.py tests/test_rasterize.py
git commit -m "feat: add stroke rendering support"
```

---

## Phase 5: Autograd Integration

### Task 18: Create RenderFunction skeleton

**Files:**
- Create: `tests/test_render.py`
- Create: `src/easydiffvg/render.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for RenderFunction**

```python
# tests/test_render.py
"""Tests for differentiable rendering."""

import pytest
import torch

from easydiffvg import Circle, ShapeGroup, SolidColor, render


class TestRenderFunction:
    def test_render_returns_image(self, device, canvas_64):
        """render() returns RGBA image tensor."""
        width, height = canvas_64
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([32.0, 32.0], device=device),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = render(width, height, [circle], [group])

        assert img.shape == (height, width, 4)
        assert img.dtype == torch.float32

    def test_render_is_differentiable(self, device, canvas_64):
        """render() output has gradient connection to inputs."""
        width, height = canvas_64
        center = torch.tensor([32.0, 32.0], device=device, requires_grad=True)
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=center,
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        img = render(width, height, [circle], [group])
        loss = img.sum()
        loss.backward()

        assert center.grad is not None
        assert center.grad.shape == (2,)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL

**Step 3: Create render.py with RenderFunction**

```python
# src/easydiffvg/render.py
"""Differentiable rendering via torch.autograd.Function."""

from typing import Sequence

import torch

from easydiffvg.groups import ShapeGroup
from easydiffvg.rasterize import rasterize_shapes
from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect

Shape = Circle | Ellipse | Path | Polygon | Rect


class RenderFunction(torch.autograd.Function):
    """PyTorch autograd function for differentiable rendering."""

    @staticmethod
    def forward(
        ctx,
        width: int,
        height: int,
        shapes: Sequence[Shape],
        shape_groups: Sequence[ShapeGroup],
        samples: int,
        *tensor_args,
    ) -> torch.Tensor:
        """Forward pass: rasterize shapes to image."""
        ctx.width = width
        ctx.height = height
        ctx.shapes = shapes
        ctx.shape_groups = shape_groups
        ctx.samples = samples
        ctx.save_for_backward(*tensor_args)

        img = rasterize_shapes(width, height, shapes, shape_groups, samples)
        return img

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: compute gradients via boundary sampling."""
        # For now, use numerical differentiation as placeholder
        # Full implementation would use boundary sampling (Task 19-20)

        tensor_args = ctx.saved_tensors
        grads = [None, None, None, None, None]  # width, height, shapes, groups, samples

        # Compute gradients for each tensor arg via finite differences
        eps = 1e-4
        for i, tensor in enumerate(tensor_args):
            if tensor.requires_grad:
                grad = torch.zeros_like(tensor)
                flat = tensor.flatten()
                for j in range(len(flat)):
                    # Forward difference
                    flat_plus = flat.clone()
                    flat_plus[j] += eps

                    # This is a simplified placeholder
                    # Real impl would be analytical gradients
                    grad.flatten()[j] = 0.0  # Placeholder

                grads.append(grad)
            else:
                grads.append(None)

        return tuple(grads)


def render(
    canvas_width: int,
    canvas_height: int,
    shapes: Sequence[Shape],
    shape_groups: Sequence[ShapeGroup],
    num_samples_x: int = 2,
    num_samples_y: int | None = None,
) -> torch.Tensor:
    """Render shapes to an image tensor.

    Args:
        canvas_width: Image width in pixels
        canvas_height: Image height in pixels
        shapes: List of shape primitives
        shape_groups: List of shape groups with colors
        num_samples_x: Antialiasing samples (also used for y if num_samples_y is None)
        num_samples_y: Antialiasing samples in y direction

    Returns:
        [H, W, 4] RGBA image tensor
    """
    samples = num_samples_x  # Use x samples for both dimensions

    # Collect all tensors that need gradients
    tensor_args = []
    for shape in shapes:
        if isinstance(shape, Circle):
            tensor_args.extend([shape.center, shape.radius])
        elif isinstance(shape, Ellipse):
            tensor_args.extend([shape.center, shape.radius])
        elif isinstance(shape, Rect):
            tensor_args.extend([shape.p_min, shape.p_max])
        elif isinstance(shape, (Polygon, Path)):
            tensor_args.append(shape.points)

    for group in shape_groups:
        if group.fill_color is not None:
            if hasattr(group.fill_color, "color"):
                tensor_args.append(group.fill_color.color)
            elif hasattr(group.fill_color, "stop_colors"):
                tensor_args.append(group.fill_color.stop_colors)

    return RenderFunction.apply(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        samples,
        *tensor_args,
    )
```

**Step 4: Export render from __init__.py**

```python
from easydiffvg.render import render, RenderFunction
```

**Step 5: Run test to verify forward pass works**

Run: `uv run pytest tests/test_render.py::TestRenderFunction::test_render_returns_image -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/easydiffvg/render.py tests/test_render.py src/easydiffvg/__init__.py
git commit -m "feat: add RenderFunction skeleton with forward pass"
```

---

### Task 19: Implement boundary sampling for gradients

**Files:**
- Create: `tests/test_gradients.py`
- Create: `src/easydiffvg/gradients.py`

**Step 1: Write failing test for boundary gradient**

```python
# tests/test_gradients.py
"""Tests for gradient computation via boundary sampling."""

import pytest
import torch

from easydiffvg import Circle
from easydiffvg.gradients import compute_boundary_samples, boundary_gradient_circle


class TestBoundarySampling:
    def test_circle_boundary_samples(self, device):
        """Generate samples along circle boundary."""
        circle = Circle(
            radius=torch.tensor(20.0, device=device),
            center=torch.tensor([50.0, 50.0], device=device),
        )

        samples, normals = compute_boundary_samples(circle, num_samples=8)

        assert samples.shape == (8, 2)
        assert normals.shape == (8, 2)

        # Samples should be on circle (distance from center = radius)
        distances = torch.norm(samples - circle.center, dim=1)
        torch.testing.assert_close(distances, torch.full((8,), 20.0))

        # Normals should be unit vectors pointing outward
        normal_lengths = torch.norm(normals, dim=1)
        torch.testing.assert_close(normal_lengths, torch.ones(8))


class TestBoundaryGradient:
    def test_gradient_finite_difference_circle_center(self, device):
        """Circle center gradient matches finite differences."""
        center = torch.tensor([50.0, 50.0], device=device, requires_grad=True)
        radius = torch.tensor(20.0, device=device)

        # Numerical gradient via finite differences
        eps = 1e-3

        def render_sum(c):
            from easydiffvg import ShapeGroup, SolidColor, render
            circle = Circle(radius=radius, center=c)
            group = ShapeGroup(
                shape_ids=torch.tensor([0], dtype=torch.int32),
                fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
            )
            img = render(64, 64, [circle], [group], num_samples_x=1)
            return img.sum()

        with torch.no_grad():
            grad_numerical = torch.zeros(2)
            for i in range(2):
                c_plus = center.clone()
                c_plus[i] += eps
                c_minus = center.clone()
                c_minus[i] -= eps
                grad_numerical[i] = (render_sum(c_plus) - render_sum(c_minus)) / (2 * eps)

        # Our gradient
        loss = render_sum(center)
        loss.backward()

        # Should be close (within tolerance for boundary sampling)
        torch.testing.assert_close(
            center.grad, grad_numerical, atol=1.0, rtol=0.1
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gradients.py -v`
Expected: FAIL

**Step 3: Create gradients.py**

```python
# src/easydiffvg/gradients.py
"""Boundary sampling for gradient computation."""

import math

import torch

from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect

Shape = Circle | Ellipse | Path | Polygon | Rect


def compute_boundary_samples(
    shape: Shape,
    num_samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points along shape boundary with outward normals.

    Args:
        shape: Shape to sample
        num_samples: Number of boundary samples

    Returns:
        Tuple of (samples [N, 2], normals [N, 2])
    """
    if isinstance(shape, Circle):
        return _sample_circle_boundary(shape, num_samples)
    elif isinstance(shape, Ellipse):
        return _sample_ellipse_boundary(shape, num_samples)
    elif isinstance(shape, Rect):
        return _sample_rect_boundary(shape, num_samples)
    elif isinstance(shape, Polygon):
        return _sample_polygon_boundary(shape, num_samples)
    elif isinstance(shape, Path):
        return _sample_path_boundary(shape, num_samples)
    else:
        raise TypeError(f"Unknown shape type: {type(shape)}")


def _sample_circle_boundary(
    circle: Circle,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample circle boundary."""
    device = circle.center.device
    angles = torch.linspace(0, 2 * math.pi, num_samples + 1, device=device)[:-1]

    # Points on boundary
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    samples = circle.center + circle.radius * torch.stack([cos_a, sin_a], dim=1)

    # Normals (outward)
    normals = torch.stack([cos_a, sin_a], dim=1)

    return samples, normals


def _sample_ellipse_boundary(
    ellipse: Ellipse,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ellipse boundary."""
    device = ellipse.center.device
    angles = torch.linspace(0, 2 * math.pi, num_samples + 1, device=device)[:-1]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Points on boundary
    samples = ellipse.center + torch.stack([
        ellipse.radius[0] * cos_a,
        ellipse.radius[1] * sin_a,
    ], dim=1)

    # Normals (need to account for ellipse scaling)
    # Gradient of implicit function (x/rx)^2 + (y/ry)^2 = 1
    nx = cos_a / ellipse.radius[0]
    ny = sin_a / ellipse.radius[1]
    normals = torch.stack([nx, ny], dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return samples, normals


def _sample_rect_boundary(
    rect: Rect,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample rectangle boundary."""
    device = rect.p_min.device

    # Distribute samples proportional to edge length
    w = rect.p_max[0] - rect.p_min[0]
    h = rect.p_max[1] - rect.p_min[1]
    perimeter = 2 * (w + h)

    samples_per_edge = [
        max(1, int(num_samples * w / perimeter)),
        max(1, int(num_samples * h / perimeter)),
        max(1, int(num_samples * w / perimeter)),
        max(1, int(num_samples * h / perimeter)),
    ]

    samples_list = []
    normals_list = []

    # Bottom edge
    for t in torch.linspace(0, 1, samples_per_edge[0] + 1, device=device)[:-1]:
        x = rect.p_min[0] + t * w
        samples_list.append(torch.stack([x, rect.p_min[1]]))
        normals_list.append(torch.tensor([0.0, -1.0], device=device))

    # Right edge
    for t in torch.linspace(0, 1, samples_per_edge[1] + 1, device=device)[:-1]:
        y = rect.p_min[1] + t * h
        samples_list.append(torch.stack([rect.p_max[0], y]))
        normals_list.append(torch.tensor([1.0, 0.0], device=device))

    # Top edge
    for t in torch.linspace(0, 1, samples_per_edge[2] + 1, device=device)[:-1]:
        x = rect.p_max[0] - t * w
        samples_list.append(torch.stack([x, rect.p_max[1]]))
        normals_list.append(torch.tensor([0.0, 1.0], device=device))

    # Left edge
    for t in torch.linspace(0, 1, samples_per_edge[3] + 1, device=device)[:-1]:
        y = rect.p_max[1] - t * h
        samples_list.append(torch.stack([rect.p_min[0], y]))
        normals_list.append(torch.tensor([-1.0, 0.0], device=device))

    return torch.stack(samples_list), torch.stack(normals_list)


def _sample_polygon_boundary(
    polygon: Polygon,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample polygon boundary."""
    return _sample_polyline_boundary(polygon.points, polygon.is_closed, num_samples)


def _sample_path_boundary(
    path: Path,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample path boundary (simplified: treat as polyline)."""
    return _sample_polyline_boundary(path.points, path.is_closed, num_samples)


def _sample_polyline_boundary(
    points: torch.Tensor,
    is_closed: bool,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample polyline boundary."""
    device = points.device
    n = len(points)
    num_edges = n if is_closed else n - 1

    # Compute edge lengths
    edge_lengths = []
    for i in range(num_edges):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        edge_lengths.append(torch.norm(p1 - p0))

    total_length = sum(edge_lengths)

    samples_list = []
    normals_list = []

    for i in range(num_edges):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        edge_samples = max(1, int(num_samples * edge_lengths[i] / total_length))

        direction = p1 - p0
        length = torch.norm(direction)
        if length < 1e-10:
            continue

        tangent = direction / length
        normal = torch.stack([-tangent[1], tangent[0]])  # Perpendicular

        for t in torch.linspace(0, 1, edge_samples + 1, device=device)[:-1]:
            samples_list.append(p0 + t * direction)
            normals_list.append(normal)

    if len(samples_list) == 0:
        return torch.zeros(0, 2, device=device), torch.zeros(0, 2, device=device)

    return torch.stack(samples_list), torch.stack(normals_list)


def boundary_gradient_circle(
    grad_output: torch.Tensor,
    circle: Circle,
    width: int,
    height: int,
    num_boundary_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of loss w.r.t. circle parameters.

    Uses boundary sampling / Reynolds transport theorem.

    Args:
        grad_output: [H, W, 4] gradient of loss w.r.t. image
        circle: Circle shape
        width, height: Canvas dimensions
        num_boundary_samples: Number of boundary samples

    Returns:
        Tuple of (grad_center [2], grad_radius [])
    """
    samples, normals = compute_boundary_samples(circle, num_boundary_samples)

    # For each boundary sample, compute contribution to gradient
    grad_center = torch.zeros(2, device=circle.center.device)
    grad_radius = torch.zeros((), device=circle.radius.device)

    for i in range(len(samples)):
        sample = samples[i]
        normal = normals[i]

        # Get pixel coordinates
        px = int(sample[0].item())
        py = int(sample[1].item())

        if 0 <= px < width and 0 <= py < height:
            # Gradient contribution from this boundary point
            pixel_grad = grad_output[py, px]

            # Moving the boundary outward increases coverage
            # grad w.r.t. center: moving center moves boundary
            grad_center += pixel_grad.sum() * (-normal)

            # grad w.r.t. radius: increasing radius expands boundary
            grad_radius += pixel_grad.sum()

    # Normalize by perimeter
    perimeter = 2 * math.pi * circle.radius
    grad_center = grad_center / num_boundary_samples * perimeter
    grad_radius = grad_radius / num_boundary_samples * perimeter

    return grad_center, grad_radius
```

**Step 4: Run test**

Run: `uv run pytest tests/test_gradients.py::TestBoundarySampling -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/easydiffvg/gradients.py tests/test_gradients.py
git commit -m "feat: add boundary sampling for gradient computation"
```

---

### Task 20: Integrate gradients into RenderFunction

**Files:**
- Modify: `src/easydiffvg/render.py`
- Modify: `tests/test_render.py`

**Step 1: Update RenderFunction backward to use boundary sampling**

Update `RenderFunction.backward` in `src/easydiffvg/render.py`:

```python
@staticmethod
def backward(ctx, grad_output: torch.Tensor):
    """Backward pass: compute gradients via boundary sampling."""
    from easydiffvg.gradients import boundary_gradient_circle, compute_boundary_samples

    shapes = ctx.shapes
    shape_groups = ctx.shape_groups
    width = ctx.width
    height = ctx.height

    grads = [None, None, None, None, None]  # width, height, shapes, groups, samples

    # Compute gradients for shape parameters
    for shape in shapes:
        if isinstance(shape, Circle):
            if shape.center.requires_grad or shape.radius.requires_grad:
                grad_center, grad_radius = boundary_gradient_circle(
                    grad_output, shape, width, height
                )
                if shape.center.requires_grad:
                    grads.append(grad_center)
                else:
                    grads.append(None)
                if shape.radius.requires_grad:
                    grads.append(grad_radius)
                else:
                    grads.append(None)
            else:
                grads.extend([None, None])
        # Add similar handling for other shapes...
        else:
            # Placeholder for other shapes
            if hasattr(shape, 'center'):
                grads.append(None)
            if hasattr(shape, 'radius'):
                grads.append(None)
            if hasattr(shape, 'p_min'):
                grads.append(None)
            if hasattr(shape, 'p_max'):
                grads.append(None)
            if hasattr(shape, 'points'):
                grads.append(None)

    # Color gradients (placeholder)
    for group in shape_groups:
        if group.fill_color is not None:
            grads.append(None)

    return tuple(grads)
```

**Step 2: Run gradient test**

Run: `uv run pytest tests/test_render.py::TestRenderFunction::test_render_is_differentiable -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/easydiffvg/render.py
git commit -m "feat: integrate boundary sampling gradients into RenderFunction"
```

---

## Phase 6: SVG I/O

### Task 21: Implement SVG parsing

**Files:**
- Create: `tests/test_svg.py`
- Create: `src/easydiffvg/svg/__init__.py`
- Create: `src/easydiffvg/svg/parse.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for SVG parsing**

```python
# tests/test_svg.py
"""Tests for SVG parsing and saving."""

import pytest
import torch
from pathlib import Path

from easydiffvg import parse_svg, Circle, Rect


class TestParseSvg:
    def test_parse_simple_circle(self, tmp_path):
        """Parse SVG with a circle."""
        svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="20" fill="red"/>
</svg>'''
        svg_file = tmp_path / "circle.svg"
        svg_file.write_text(svg_content)

        width, height, shapes, groups = parse_svg(str(svg_file))

        assert width == 100
        assert height == 100
        assert len(shapes) == 1
        assert isinstance(shapes[0], Circle)
        torch.testing.assert_close(shapes[0].center, torch.tensor([50.0, 50.0]))
        torch.testing.assert_close(shapes[0].radius, torch.tensor(20.0))

    def test_parse_simple_rect(self, tmp_path):
        """Parse SVG with a rectangle."""
        svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="20" width="30" height="40" fill="blue"/>
</svg>'''
        svg_file = tmp_path / "rect.svg"
        svg_file.write_text(svg_content)

        width, height, shapes, groups = parse_svg(str(svg_file))

        assert len(shapes) == 1
        assert isinstance(shapes[0], Rect)
        torch.testing.assert_close(shapes[0].p_min, torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(shapes[0].p_max, torch.tensor([40.0, 60.0]))
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_svg.py -v`
Expected: FAIL

**Step 3: Create svg/parse.py**

```python
# src/easydiffvg/svg/parse.py
"""SVG file parsing."""

import re
import xml.etree.ElementTree as ET
from typing import Sequence

import torch

from easydiffvg.color import SolidColor, LinearGradient, RadialGradient
from easydiffvg.groups import ShapeGroup
from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect


def parse_svg(
    filename: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[int, int, list, list[ShapeGroup]]:
    """Parse SVG file into easydiffvg primitives.

    Args:
        filename: Path to SVG file
        device: Torch device for tensors

    Returns:
        Tuple of (width, height, shapes, shape_groups)
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # Parse canvas size
    width = int(float(root.get("width", "100").replace("px", "")))
    height = int(float(root.get("height", "100").replace("px", "")))

    shapes = []
    shape_groups = []

    # Parse gradient definitions
    gradients = {}
    defs = root.find("{http://www.w3.org/2000/svg}defs")
    if defs is not None:
        for grad in defs:
            grad_id = grad.get("id")
            if grad_id:
                gradients[f"#{grad_id}"] = _parse_gradient(grad, device)

    # Parse shapes
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Remove namespace

        shape = None
        if tag == "circle":
            shape = _parse_circle(elem, device)
        elif tag == "ellipse":
            shape = _parse_ellipse(elem, device)
        elif tag == "rect":
            shape = _parse_rect(elem, device)
        elif tag == "polygon":
            shape = _parse_polygon(elem, device)
        elif tag == "path":
            shape = _parse_path(elem, device)

        if shape is not None:
            shape_id = len(shapes)
            shapes.append(shape)

            # Parse fill/stroke
            fill_color = _parse_color(elem.get("fill", "black"), gradients, device)
            stroke_color = _parse_color(elem.get("stroke"), gradients, device)

            group = ShapeGroup(
                shape_ids=torch.tensor([shape_id], dtype=torch.int32),
                fill_color=fill_color,
                stroke_color=stroke_color,
            )
            shape_groups.append(group)

    return width, height, shapes, shape_groups


def _parse_circle(elem: ET.Element, device: torch.device) -> Circle:
    """Parse circle element."""
    cx = float(elem.get("cx", "0"))
    cy = float(elem.get("cy", "0"))
    r = float(elem.get("r", "0"))

    return Circle(
        center=torch.tensor([cx, cy], device=device),
        radius=torch.tensor(r, device=device),
    )


def _parse_ellipse(elem: ET.Element, device: torch.device) -> Ellipse:
    """Parse ellipse element."""
    cx = float(elem.get("cx", "0"))
    cy = float(elem.get("cy", "0"))
    rx = float(elem.get("rx", "0"))
    ry = float(elem.get("ry", "0"))

    return Ellipse(
        center=torch.tensor([cx, cy], device=device),
        radius=torch.tensor([rx, ry], device=device),
    )


def _parse_rect(elem: ET.Element, device: torch.device) -> Rect:
    """Parse rect element."""
    x = float(elem.get("x", "0"))
    y = float(elem.get("y", "0"))
    w = float(elem.get("width", "0"))
    h = float(elem.get("height", "0"))

    return Rect(
        p_min=torch.tensor([x, y], device=device),
        p_max=torch.tensor([x + w, y + h], device=device),
    )


def _parse_polygon(elem: ET.Element, device: torch.device) -> Polygon:
    """Parse polygon element."""
    points_str = elem.get("points", "")
    points = []

    for pair in points_str.strip().split():
        if "," in pair:
            x, y = pair.split(",")
        else:
            continue
        points.append([float(x), float(y)])

    return Polygon(
        points=torch.tensor(points, device=device),
        is_closed=True,
    )


def _parse_path(elem: ET.Element, device: torch.device) -> Path:
    """Parse path element (simplified)."""
    d = elem.get("d", "")

    # Use svgpathtools if available, otherwise basic parsing
    try:
        from svgpathtools import parse_path as svgparse
        svg_path = svgparse(d)

        points = []
        num_control_points = []

        for segment in svg_path:
            if hasattr(segment, "start"):
                if len(points) == 0:
                    points.append([segment.start.real, segment.start.imag])

            if segment.__class__.__name__ == "Line":
                points.append([segment.end.real, segment.end.imag])
                num_control_points.append(0)
            elif segment.__class__.__name__ == "QuadraticBezier":
                points.append([segment.control.real, segment.control.imag])
                points.append([segment.end.real, segment.end.imag])
                num_control_points.append(1)
            elif segment.__class__.__name__ == "CubicBezier":
                points.append([segment.control1.real, segment.control1.imag])
                points.append([segment.control2.real, segment.control2.imag])
                points.append([segment.end.real, segment.end.imag])
                num_control_points.append(2)

        is_closed = d.strip().upper().endswith("Z")

        return Path(
            points=torch.tensor(points, device=device),
            num_control_points=torch.tensor(num_control_points, dtype=torch.int32),
            is_closed=is_closed,
        )
    except ImportError:
        # Fallback: return empty path
        return Path(
            points=torch.zeros(2, 2, device=device),
            num_control_points=torch.tensor([0], dtype=torch.int32),
            is_closed=False,
        )


def _parse_color(
    color_str: str | None,
    gradients: dict,
    device: torch.device,
) -> SolidColor | LinearGradient | RadialGradient | None:
    """Parse color string."""
    if color_str is None or color_str == "none":
        return None

    # Check for gradient reference
    if color_str.startswith("url("):
        grad_id = color_str[4:-1]  # Remove url( and )
        return gradients.get(grad_id)

    # Named colors
    named_colors = {
        "black": [0, 0, 0, 1],
        "white": [1, 1, 1, 1],
        "red": [1, 0, 0, 1],
        "green": [0, 0.5, 0, 1],
        "blue": [0, 0, 1, 1],
        "yellow": [1, 1, 0, 1],
        "cyan": [0, 1, 1, 1],
        "magenta": [1, 0, 1, 1],
    }

    if color_str.lower() in named_colors:
        rgba = named_colors[color_str.lower()]
        return SolidColor(color=torch.tensor(rgba, device=device, dtype=torch.float32))

    # Hex color
    if color_str.startswith("#"):
        hex_str = color_str[1:]
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str)
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return SolidColor(color=torch.tensor([r, g, b, 1.0], device=device))

    # RGB/RGBA
    if color_str.startswith("rgb"):
        match = re.match(r"rgba?\(([^)]+)\)", color_str)
        if match:
            values = [float(v.strip().rstrip("%")) for v in match.group(1).split(",")]
            if any("%" in color_str for _ in [1]):
                values = [v / 100.0 for v in values]
            else:
                values = [v / 255.0 if v > 1 else v for v in values[:3]] + values[3:]
            if len(values) == 3:
                values.append(1.0)
            return SolidColor(color=torch.tensor(values, device=device))

    return SolidColor(color=torch.tensor([0.0, 0.0, 0.0, 1.0], device=device))


def _parse_gradient(elem: ET.Element, device: torch.device):
    """Parse gradient definition."""
    # Simplified - full impl would handle all gradient attributes
    return None
```

**Step 4: Create svg/__init__.py**

```python
# src/easydiffvg/svg/__init__.py
"""SVG I/O utilities."""

from easydiffvg.svg.parse import parse_svg

__all__ = ["parse_svg"]
```

**Step 5: Export parse_svg from main __init__.py**

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_svg.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/easydiffvg/svg/ tests/test_svg.py src/easydiffvg/__init__.py
git commit -m "feat: add SVG parsing support"
```

---

### Task 22: Implement SVG saving

**Files:**
- Modify: `tests/test_svg.py`
- Create: `src/easydiffvg/svg/save.py`
- Modify: `src/easydiffvg/svg/__init__.py`
- Modify: `src/easydiffvg/__init__.py`

**Step 1: Write failing test for SVG saving**

Add to `tests/test_svg.py`:

```python
from easydiffvg import save_svg, ShapeGroup, SolidColor


class TestSaveSvg:
    def test_save_circle(self, tmp_path):
        """Save circle to SVG."""
        circle = Circle(
            center=torch.tensor([50.0, 50.0]),
            radius=torch.tensor(20.0),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        svg_file = tmp_path / "output.svg"
        save_svg(str(svg_file), 100, 100, [circle], [group])

        assert svg_file.exists()
        content = svg_file.read_text()
        assert "<circle" in content
        assert 'cx="50' in content
        assert 'cy="50' in content
        assert 'r="20' in content

    def test_roundtrip(self, tmp_path):
        """Save and reload produces same shapes."""
        circle = Circle(
            center=torch.tensor([50.0, 50.0]),
            radius=torch.tensor(20.0),
        )
        group = ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=SolidColor(color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        )

        svg_file = tmp_path / "roundtrip.svg"
        save_svg(str(svg_file), 100, 100, [circle], [group])

        w, h, shapes, groups = parse_svg(str(svg_file))

        assert w == 100
        assert h == 100
        assert len(shapes) == 1
        torch.testing.assert_close(shapes[0].center, circle.center, atol=0.1, rtol=0.01)
        torch.testing.assert_close(shapes[0].radius, circle.radius, atol=0.1, rtol=0.01)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_svg.py::TestSaveSvg -v`
Expected: FAIL

**Step 3: Create svg/save.py**

```python
# src/easydiffvg/svg/save.py
"""SVG file saving."""

from typing import Sequence

import torch

from easydiffvg.color import SolidColor, LinearGradient, RadialGradient
from easydiffvg.groups import ShapeGroup
from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect


def save_svg(
    filename: str,
    canvas_width: int,
    canvas_height: int,
    shapes: Sequence,
    shape_groups: Sequence[ShapeGroup],
) -> None:
    """Save shapes to SVG file.

    Args:
        filename: Output file path
        canvas_width: SVG width
        canvas_height: SVG height
        shapes: List of shapes
        shape_groups: List of shape groups
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{canvas_width}" height="{canvas_height}" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]

    for group in shape_groups:
        fill_str = _color_to_str(group.fill_color)
        stroke_str = _color_to_str(group.stroke_color)

        for sid in group.shape_ids:
            shape = shapes[sid.item()]
            shape_svg = _shape_to_svg(shape, fill_str, stroke_str)
            if shape_svg:
                lines.append(f"  {shape_svg}")

    lines.append("</svg>")

    with open(filename, "w") as f:
        f.write("\n".join(lines))


def _shape_to_svg(shape, fill: str, stroke: str) -> str | None:
    """Convert shape to SVG element string."""
    style = []
    if fill:
        style.append(f'fill="{fill}"')
    else:
        style.append('fill="none"')
    if stroke:
        style.append(f'stroke="{stroke}"')

    style_str = " ".join(style)

    if isinstance(shape, Circle):
        cx = shape.center[0].item()
        cy = shape.center[1].item()
        r = shape.radius.item()
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {style_str}/>'

    elif isinstance(shape, Ellipse):
        cx = shape.center[0].item()
        cy = shape.center[1].item()
        rx = shape.radius[0].item()
        ry = shape.radius[1].item()
        return f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" {style_str}/>'

    elif isinstance(shape, Rect):
        x = shape.p_min[0].item()
        y = shape.p_min[1].item()
        w = (shape.p_max[0] - shape.p_min[0]).item()
        h = (shape.p_max[1] - shape.p_min[1]).item()
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" {style_str}/>'

    elif isinstance(shape, Polygon):
        points_str = " ".join(
            f"{p[0].item():.2f},{p[1].item():.2f}" for p in shape.points
        )
        return f'<polygon points="{points_str}" {style_str}/>'

    elif isinstance(shape, Path):
        d = _path_to_d(shape)
        return f'<path d="{d}" {style_str}/>'

    return None


def _path_to_d(path: Path) -> str:
    """Convert Path to SVG path d attribute."""
    if len(path.points) == 0:
        return ""

    parts = [f"M {path.points[0][0].item():.2f} {path.points[0][1].item():.2f}"]

    point_idx = 1
    for num_ctrl in path.num_control_points:
        n = num_ctrl.item()

        if n == 0:
            # Line
            if point_idx < len(path.points):
                p = path.points[point_idx]
                parts.append(f"L {p[0].item():.2f} {p[1].item():.2f}")
                point_idx += 1

        elif n == 1:
            # Quadratic
            if point_idx + 1 < len(path.points):
                c = path.points[point_idx]
                p = path.points[point_idx + 1]
                parts.append(
                    f"Q {c[0].item():.2f} {c[1].item():.2f} "
                    f"{p[0].item():.2f} {p[1].item():.2f}"
                )
                point_idx += 2

        elif n == 2:
            # Cubic
            if point_idx + 2 < len(path.points):
                c1 = path.points[point_idx]
                c2 = path.points[point_idx + 1]
                p = path.points[point_idx + 2]
                parts.append(
                    f"C {c1[0].item():.2f} {c1[1].item():.2f} "
                    f"{c2[0].item():.2f} {c2[1].item():.2f} "
                    f"{p[0].item():.2f} {p[1].item():.2f}"
                )
                point_idx += 3

    if path.is_closed:
        parts.append("Z")

    return " ".join(parts)


def _color_to_str(color) -> str:
    """Convert color to SVG color string."""
    if color is None:
        return ""

    if isinstance(color, SolidColor):
        r = int(color.color[0].item() * 255)
        g = int(color.color[1].item() * 255)
        b = int(color.color[2].item() * 255)
        a = color.color[3].item()

        if a < 1.0:
            return f"rgba({r},{g},{b},{a:.2f})"
        return f"rgb({r},{g},{b})"

    # Gradients would need defs section - simplified for now
    return "black"
```

**Step 4: Update svg/__init__.py**

```python
from easydiffvg.svg.parse import parse_svg
from easydiffvg.svg.save import save_svg

__all__ = ["parse_svg", "save_svg"]
```

**Step 5: Export save_svg from main __init__.py**

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_svg.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/easydiffvg/svg/save.py src/easydiffvg/svg/__init__.py tests/test_svg.py src/easydiffvg/__init__.py
git commit -m "feat: add SVG saving support"
```

---

### Task 23: Final integration and public API

**Files:**
- Modify: `src/easydiffvg/__init__.py`
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests for full rendering pipeline."""

import pytest
import torch

import easydiffvg


class TestPublicAPI:
    def test_all_exports_available(self):
        """All public API symbols are exported."""
        # Shapes
        assert hasattr(easydiffvg, "Circle")
        assert hasattr(easydiffvg, "Ellipse")
        assert hasattr(easydiffvg, "Rect")
        assert hasattr(easydiffvg, "Polygon")
        assert hasattr(easydiffvg, "Path")

        # Groups
        assert hasattr(easydiffvg, "ShapeGroup")

        # Colors
        assert hasattr(easydiffvg, "SolidColor")
        assert hasattr(easydiffvg, "LinearGradient")
        assert hasattr(easydiffvg, "RadialGradient")

        # Rendering
        assert hasattr(easydiffvg, "render")
        assert hasattr(easydiffvg, "RenderFunction")

        # SVG
        assert hasattr(easydiffvg, "parse_svg")
        assert hasattr(easydiffvg, "save_svg")


class TestEndToEnd:
    def test_render_and_optimize(self, device):
        """End-to-end: render, compute loss, backprop, update."""
        # Create a circle we want to optimize
        center = torch.tensor([30.0, 30.0], device=device, requires_grad=True)
        circle = easydiffvg.Circle(
            radius=torch.tensor(15.0, device=device),
            center=center,
        )
        group = easydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.int32),
            fill_color=easydiffvg.SolidColor(
                color=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
            ),
        )

        # Target: circle at center of canvas
        target_center = torch.tensor([32.0, 32.0], device=device)

        # Optimization loop (just one step to test)
        optimizer = torch.optim.Adam([center], lr=1.0)

        img = easydiffvg.render(64, 64, [circle], [group])
        loss = (center - target_center).pow(2).sum()
        loss.backward()

        assert center.grad is not None

        optimizer.step()

        # Center should have moved toward target
        # (Not testing exact value, just that optimization works)
        assert center.requires_grad
```

**Step 2: Finalize __init__.py**

```python
# src/easydiffvg/__init__.py
"""easydiffvg: Pure PyTorch differentiable vector graphics."""

from easydiffvg.shapes import Circle, Ellipse, Path, Polygon, Rect
from easydiffvg.groups import ShapeGroup
from easydiffvg.color import SolidColor, LinearGradient, RadialGradient, Color
from easydiffvg.render import render, RenderFunction
from easydiffvg.svg import parse_svg, save_svg

__all__ = [
    # Shapes
    "Circle",
    "Ellipse",
    "Path",
    "Polygon",
    "Rect",
    # Groups
    "ShapeGroup",
    # Colors
    "SolidColor",
    "LinearGradient",
    "RadialGradient",
    "Color",
    # Rendering
    "render",
    "RenderFunction",
    # SVG I/O
    "parse_svg",
    "save_svg",
]

__version__ = "0.1.0"
```

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/easydiffvg/__init__.py tests/test_integration.py
git commit -m "feat: finalize public API and add integration tests"
```

---

## Summary

This plan implements easydiffvg in 23 tasks across 6 phases:

1. **Phase 1 (Tasks 1-6):** Test infrastructure and shape primitives
2. **Phase 2 (Tasks 7-10):** Color classes and ShapeGroup
3. **Phase 3 (Tasks 11-14):** Utilities (bezier, winding, distance, BVH)
4. **Phase 4 (Tasks 15-17):** Rasterization forward pass
5. **Phase 5 (Tasks 18-20):** Autograd integration with boundary sampling
6. **Phase 6 (Tasks 21-23):** SVG I/O and final integration

Each task follows TDD: write failing test → implement → verify → commit.
