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
