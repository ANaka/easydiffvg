# easydiffvg Test Suite Design

## Overview

Comprehensive test suite for easydiffvg that benchmarks against original diffvg. Tests verify API compatibility, rendering correctness, and gradient accuracy.

**Goals:**
- Development guidance (TDD-style) - tests define expected behavior
- Numerical correctness - pixel-level output matches original diffvg
- Gradient correctness - backward pass produces matching gradients

**Approach:** Parameterized test matrix with pre-generated fixtures. Reference data generated once with original diffvg and committed to repo. No diffvg dependency needed at test time.

## Test Directory Structure

```
tests/
├── conftest.py              # Shared fixtures, tolerance helpers
├── fixtures/
│   ├── generate_references.py   # Script to run once with original diffvg
│   ├── shapes/              # Shape configuration JSON files
│   │   ├── circle_basic.json
│   │   ├── circle_variants.json
│   │   ├── path_cubic.json
│   │   └── ...
│   ├── images/              # Reference rendered images (.pt tensor files)
│   │   ├── circle_basic_64x64.pt
│   │   └── ...
│   └── gradients/           # Reference backward pass outputs (.pt files)
│       ├── circle_basic_grad.pt
│       └── ...
├── test_forward_exact.py    # Pixel-exact forward pass tests
├── test_forward_visual.py   # Visual similarity tests (RMSE threshold)
├── test_backward.py         # Gradient correctness tests
├── test_api_compat.py       # API shape/signature compatibility
└── test_svg_roundtrip.py    # SVG parse → render → save → parse cycle
```

## Test Matrix

### Shapes (5)
- Circle: varying center, radius
- Ellipse: varying center, rx/ry ratios
- Rect: varying positions, sizes
- Polygon: triangle, quad, complex (10+ points)
- Path: line segments, quadratic bezier, cubic bezier, mixed, closed/open

### Colors (3)
- SolidColor: opaque, semi-transparent
- LinearGradient: horizontal, vertical, diagonal, multi-stop
- RadialGradient: centered, off-center, elliptical

### Rendering Variations
- Canvas sizes: 64×64 (fast), 256×256 (realistic)
- Samples: 1 (no AA), 2 (default), 4 (high quality)
- Fill only, stroke only, fill + stroke

### Transforms
- Identity, translation, rotation, scale, combined

### Edge Cases
- Overlapping shapes (z-order)
- Shapes partially outside canvas
- Zero-width stroke, fully transparent
- Even-odd vs winding fill rule

**Total:** ~200-300 distinct test cases. Each case gets an exact test and a visual test. Gradient tests run on a representative subset (~50 cases) since backward pass is slower.

## Fixture Generation

`fixtures/generate_references.py` runs once with original diffvg installed:

```python
# fixtures/generate_references.py
"""
Run with original diffvg to generate reference data.
Usage: python generate_references.py
Requires: pydiffvg installed
"""

def generate_all():
    for config_path in Path("shapes").glob("*.json"):
        config = json.loads(config_path.read_text())

        # Build shapes using original pydiffvg
        shapes, groups = build_shapes_from_config(config)

        # Forward pass - render image
        img = pydiffvg.render(
            config["width"], config["height"],
            shapes, groups,
            num_samples_x=config["samples"],
            num_samples_y=config["samples"],
        )
        torch.save(img, f"images/{config['name']}.pt")

        # Backward pass - compute gradients
        img.sum().backward()
        grads = extract_gradients(shapes, groups)
        torch.save(grads, f"gradients/{config['name']}_grad.pt")
```

The script is idempotent. Adding a new test case = add JSON file + regenerate.

## Test Implementation

### Shared Fixtures (conftest.py)

```python
import pytest
import torch
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

def load_test_cases():
    """Discover all test cases from fixture files."""
    cases = []
    for config_path in (FIXTURES / "shapes").glob("*.json"):
        config = json.loads(config_path.read_text())
        cases.append(pytest.param(config, id=config["name"]))
    return cases

@pytest.fixture
def reference_image(request):
    name = request.param["name"]
    return torch.load(FIXTURES / "images" / f"{name}.pt")

@pytest.fixture
def reference_gradients(request):
    name = request.param["name"]
    return torch.load(FIXTURES / "gradients" / f"{name}_grad.pt")

def pytest_configure(config):
    config.addinivalue_line("markers", "exact: pixel-exact comparison tests")
    config.addinivalue_line("markers", "visual: visual similarity tests (RMSE)")
    config.addinivalue_line("markers", "gradients: backward pass tests")
    config.addinivalue_line("markers", "slow: tests that take >1s")
```

### Forward Pass Tests

```python
# test_forward_exact.py
@pytest.mark.exact
@pytest.mark.parametrize("config", load_test_cases())
def test_render_exact(config, reference_image):
    shapes, groups = build_shapes_from_config(config)
    result = easydiffvg.render(config["width"], config["height"], shapes, groups)

    torch.testing.assert_close(result, reference_image, rtol=1e-5, atol=1e-5)

# test_forward_visual.py
@pytest.mark.visual
@pytest.mark.parametrize("config", load_test_cases())
def test_render_visual(config, reference_image):
    shapes, groups = build_shapes_from_config(config)
    result = easydiffvg.render(config["width"], config["height"], shapes, groups)

    rmse = torch.sqrt(torch.mean((result - reference_image) ** 2))
    assert rmse < 0.01, f"RMSE {rmse:.4f} exceeds visual threshold"
```

### Gradient Tests

```python
# test_backward.py
GRADIENT_CASES = [c for c in load_test_cases() if c.values[0].get("test_gradients", True)]

@pytest.mark.gradients
@pytest.mark.slow
@pytest.mark.parametrize("config", GRADIENT_CASES)
def test_gradients(config, reference_gradients):
    shapes, groups = build_shapes_from_config(config, requires_grad=True)
    result = easydiffvg.render(config["width"], config["height"], shapes, groups)

    result.sum().backward()

    actual_grads = extract_gradients(shapes, groups)
    for name, expected in reference_gradients.items():
        actual = actual_grads[name]
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-4,
            msg=f"Gradient mismatch for {name}"
        )

@pytest.mark.gradients
@pytest.mark.slow
@pytest.mark.parametrize("config", GRADIENT_CASES)
def test_gradients_finite_diff(config):
    """Sanity check: verify gradients match finite differences."""
    shapes, groups = build_shapes_from_config(config, requires_grad=True)

    torch.autograd.gradcheck(
        lambda *params: render_with_params(config, params),
        extract_params(shapes, groups),
        eps=1e-4, atol=1e-3, rtol=1e-3,
    )
```

### API Compatibility Tests

```python
# test_api_compat.py
import easydiffvg

def test_render_function_signature():
    """Verify render() accepts same arguments as original."""
    import inspect
    easy_sig = inspect.signature(easydiffvg.render)
    # Compare against documented original signature
    expected_params = ["canvas_width", "canvas_height", "shapes", "shape_groups",
                       "num_samples_x", "num_samples_y", "filter"]
    actual_params = list(easy_sig.parameters.keys())
    assert actual_params == expected_params

def test_shape_classes_exist():
    """Verify all shape classes are exported."""
    for name in ["Circle", "Ellipse", "Path", "Polygon", "Rect"]:
        assert hasattr(easydiffvg, name)

def test_shape_constructors():
    """Verify shape constructors accept same arguments."""
    c = easydiffvg.Circle(
        center=torch.tensor([32.0, 32.0]),
        radius=torch.tensor(10.0),
    )
    assert c.center.shape == (2,)

    p = easydiffvg.Path(
        num_control_points=torch.tensor([2, 2]),
        points=torch.zeros(6, 2),
        is_closed=True,
    )
    assert p.points.shape == (6, 2)

def test_color_classes():
    """Verify gradient classes match original API."""
    lg = easydiffvg.LinearGradient(
        begin=torch.tensor([0.0, 0.0]),
        end=torch.tensor([64.0, 64.0]),
        offsets=torch.tensor([0.0, 1.0]),
        stop_colors=torch.tensor([[1, 0, 0, 1], [0, 0, 1, 1]]),
    )
    assert lg.stop_colors.shape == (2, 4)
```

### SVG Roundtrip Tests

```python
# test_svg_roundtrip.py
from pathlib import Path

SVG_FIXTURES = Path(__file__).parent / "fixtures" / "svgs"

@pytest.mark.parametrize("svg_file", SVG_FIXTURES.glob("*.svg"))
def test_parse_and_render(svg_file):
    """Parse SVG → render → compare to reference."""
    width, height, shapes, groups = easydiffvg.parse_svg(str(svg_file))
    result = easydiffvg.render(width, height, shapes, groups)

    ref = torch.load(svg_file.with_suffix(".pt"))
    torch.testing.assert_close(result, ref, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("svg_file", SVG_FIXTURES.glob("*.svg"))
def test_roundtrip_preserves_output(svg_file, tmp_path):
    """Parse → save → parse → render should match original render."""
    w, h, shapes1, groups1 = easydiffvg.parse_svg(str(svg_file))
    img1 = easydiffvg.render(w, h, shapes1, groups1)

    out_path = tmp_path / "roundtrip.svg"
    easydiffvg.save_svg(str(out_path), w, h, shapes1, groups1)

    w2, h2, shapes2, groups2 = easydiffvg.parse_svg(str(out_path))
    img2 = easydiffvg.render(w2, h2, shapes2, groups2)

    torch.testing.assert_close(img1, img2, rtol=1e-5, atol=1e-5)
```

## Shape Configuration Format

```json
// fixtures/shapes/circle_solid_fill.json
{
  "name": "circle_solid_fill",
  "width": 64,
  "height": 64,
  "samples": 2,
  "shapes": [
    {
      "type": "circle",
      "center": [32.0, 32.0],
      "radius": 20.0
    }
  ],
  "groups": [
    {
      "shape_ids": [0],
      "fill_color": {"type": "solid", "rgba": [1.0, 0.0, 0.0, 1.0]},
      "stroke_color": null
    }
  ],
  "test_gradients": true
}

// fixtures/shapes/path_cubic_gradient.json
{
  "name": "path_cubic_gradient",
  "width": 128,
  "height": 128,
  "samples": 2,
  "shapes": [
    {
      "type": "path",
      "points": [[10, 10], [40, 80], [80, 80], [120, 10]],
      "num_control_points": [2],
      "is_closed": false
    }
  ],
  "groups": [
    {
      "shape_ids": [0],
      "fill_color": null,
      "stroke_color": {
        "type": "linear_gradient",
        "begin": [0, 0],
        "end": [128, 128],
        "offsets": [0.0, 0.5, 1.0],
        "stop_colors": [[1,0,0,1], [0,1,0,1], [0,0,1,1]]
      },
      "stroke_width": 3.0
    }
  ]
}
```

## Development Workflow

```bash
# Run fast subset during development
pytest -m "not slow"

# Run only exact forward tests
pytest -m exact

# Run everything before committing
pytest

# Run specific shape type
pytest -k "circle"

# Regenerate fixtures after fixing generation bug
cd tests/fixtures && python generate_references.py
```

## Tolerances

| Test Type | rtol | atol | Notes |
|-----------|------|------|-------|
| Exact forward | 1e-5 | 1e-5 | Pixel-level match |
| Visual forward | - | RMSE < 0.01 | Perceptually identical |
| Gradients vs reference | 1e-4 | 1e-4 | Match original diffvg |
| Gradients vs finite diff | 1e-3 | 1e-3 | Mathematical correctness |
