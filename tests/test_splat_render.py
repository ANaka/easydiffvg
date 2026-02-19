import torch
import pytest


def test_evaluate_bezier_single_point():
    """Bezier at t=0 should return first control point, t=1 should return last."""
    from pydiffvg.splat_render import _evaluate_bezier

    # Single cubic: P0=(0,0), P1=(1,0), P2=(1,1), P3=(0,1)
    control_points = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]])
    # Shape: (B=1, num_strokes=1, 4, 2)

    t = torch.tensor([0.0, 1.0])
    result = _evaluate_bezier(control_points, t)
    # Shape: (B=1, num_strokes=1, K=2, 2)

    assert result.shape == (1, 1, 2, 2)
    assert torch.allclose(result[0, 0, 0], torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(result[0, 0, 1], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_evaluate_bezier_midpoint():
    """Bezier at t=0.5 for a straight line should be the midpoint."""
    from pydiffvg.splat_render import _evaluate_bezier

    # Straight line from (0,0) to (2,2) with control points on the line
    control_points = torch.tensor([[[[0.0, 0.0], [0.67, 0.67], [1.33, 1.33], [2.0, 2.0]]]])

    t = torch.tensor([0.5])
    result = _evaluate_bezier(control_points, t)

    assert torch.allclose(result[0, 0, 0], torch.tensor([1.0, 1.0]), atol=0.1)


def test_splat_render_basic_shape():
    """Render a single stroke, verify output shape and value range."""
    from pydiffvg.splat_render import splat_render_cubics

    # Single diagonal stroke from top-left to bottom-right
    cubics = torch.tensor([[
        [[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]]
    ]])  # (B=1, num_strokes=1, 4, 2)
    stroke_widths = torch.tensor([[2.0]])  # (B=1, num_strokes=1)

    result = splat_render_cubics(cubics, stroke_widths, canvas_size=64)

    assert result.shape == (1, 64, 64)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    # Background should be white (1.0), stroke should be darker
    assert result.mean() < 1.0  # Some pixels should be dark


def test_splat_render_gradient_flows():
    """Verify gradients flow through the renderer."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics = torch.tensor([[
        [[-0.5, -0.5], [0.0, -0.5], [0.0, 0.5], [0.5, 0.5]]
    ]], requires_grad=True)
    stroke_widths = torch.tensor([[2.0]])

    result = splat_render_cubics(cubics, stroke_widths, canvas_size=32)
    loss = result.mean()
    loss.backward()

    assert cubics.grad is not None
    assert cubics.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_split_path_to_cubics():
    """Split a multi-segment path into individual cubics."""
    from pydiffvg.splat_render import split_path_to_cubics

    # Path with 2 cubic segments: [P0, C1, C2, P1, C3, C4, P2]
    # num_control_points = [2, 2] means 2 control points per segment (cubic)
    points = torch.tensor([
        [0.0, 0.0],   # P0
        [1.0, 0.0],   # C1
        [2.0, 0.0],   # C2
        [3.0, 0.0],   # P1
        [4.0, 0.0],   # C3
        [5.0, 0.0],   # C4
        [6.0, 0.0],   # P2
    ])
    num_control_points = torch.tensor([2, 2])

    cubics = split_path_to_cubics(points, num_control_points)

    # Should produce 2 cubics, each with 4 control points
    assert cubics.shape == (2, 4, 2)
    # First cubic: P0, C1, C2, P1
    assert torch.allclose(cubics[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(cubics[0, 3], torch.tensor([3.0, 0.0]))
    # Second cubic: P1, C3, C4, P2
    assert torch.allclose(cubics[1, 0], torch.tensor([3.0, 0.0]))
    assert torch.allclose(cubics[1, 3], torch.tensor([6.0, 0.0]))


def test_split_path_with_lines():
    """Split a path with line segments (0 control points)."""
    from pydiffvg.splat_render import split_path_to_cubics

    # Path: line, cubic, line
    # Points: P0, P1, C1, C2, P2, P3
    points = torch.tensor([
        [0.0, 0.0],   # P0
        [1.0, 0.0],   # P1 (end of line)
        [1.5, 0.5],   # C1
        [2.5, 0.5],   # C2
        [3.0, 0.0],   # P2 (end of cubic)
        [4.0, 0.0],   # P3 (end of line)
    ])
    num_control_points = torch.tensor([0, 2, 0])

    cubics = split_path_to_cubics(points, num_control_points)

    # Lines become degenerate cubics (control points on the line)
    assert cubics.shape == (3, 4, 2)


def test_splat_render_function_interface():
    """SplatRenderFunction should match RenderFunction interface."""
    import pydiffvg
    from pydiffvg.splat_render import SplatRenderFunction

    # Create a simple path (single cubic stroke)
    points = torch.tensor([
        [50.0, 50.0],
        [100.0, 50.0],
        [100.0, 150.0],
        [150.0, 150.0],
    ])
    path = pydiffvg.Path(
        num_control_points=torch.tensor([2]),
        points=points,
        stroke_width=torch.tensor(3.0),
        is_closed=False,
    )
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    canvas_width, canvas_height = 224, 224

    # Serialize and render
    scene_args = SplatRenderFunction.serialize_scene(
        canvas_width, canvas_height, [path], [shape_group]
    )
    img = SplatRenderFunction.apply(
        canvas_width, canvas_height,
        2, 2,  # num_samples_x, num_samples_y (ignored for splat)
        0,     # seed (ignored)
        None,  # background
        *scene_args,
    )

    assert img.shape == (224, 224, 4)  # RGBA output
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_splat_render_function_gradients():
    """Gradients should flow through SplatRenderFunction."""
    import pydiffvg
    from pydiffvg.splat_render import SplatRenderFunction

    points = torch.tensor([
        [50.0, 50.0],
        [100.0, 50.0],
        [100.0, 150.0],
        [150.0, 150.0],
    ], requires_grad=True)

    path = pydiffvg.Path(
        num_control_points=torch.tensor([2]),
        points=points,
        stroke_width=torch.tensor(3.0),
        is_closed=False,
    )
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    scene_args = SplatRenderFunction.serialize_scene(
        128, 128, [path], [shape_group]
    )
    img = SplatRenderFunction.apply(128, 128, 2, 2, 0, None, *scene_args)

    loss = img.mean()
    loss.backward()

    assert points.grad is not None
    assert points.grad.abs().sum() > 0


def test_import_from_pydiffvg():
    """SplatRenderFunction should be importable from pydiffvg."""
    from pydiffvg import SplatRenderFunction
    assert SplatRenderFunction is not None
