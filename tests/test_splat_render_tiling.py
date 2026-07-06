"""Tests for the tile-culled splatting path of splat_render_cubics.

Covers:
- tiling="tiles": tile-culled rendering must match the dense default
  numerically (NOT bitwise -- different op order and log-space compositing)
  in both the forward pass and the gradients, across gaussian counts,
  tile sizes, adversarial scenes, pixel_box windows, and batches.
- tiling="auto": threshold-based dispatch between the dense and tiled paths.
- tiling="none": bitwise identical to omitting the kwarg entirely.
- Validation of the tiling / tile_size kwargs.

The project gate is <= 1e-5 agreement between tiled and dense; measured
margins are far below it (forward ~2-5e-7, gradients ~2e-9 to 3e-8).

Note on CUDA determinism: the tiled path accumulates with scatter_add,
whose CUDA kernel uses atomics, so tiled outputs vary run-to-run by ~1 ulp
(~1.2e-7 measured on an RTX 5090). Assertions that are bitwise on CPU
therefore relax to small tolerances on CUDA where the tiled path is involved.
The dense path has no scatter_add and stays deterministic per-process on
both devices.
"""

import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Project gate for tiled-vs-dense agreement (forward and gradients).
FWD_TOL = 1e-5
GRAD_TOL = 1e-5

NUM_SAMPLES = 16  # gaussians per stroke: G = num_strokes * NUM_SAMPLES


def _make_scene(device, num_strokes, seed=0, batch=1):
    """Random cubics in [-1, 1], widths, and opacities on the given device."""
    torch.manual_seed(seed)
    cubics = torch.rand(batch, num_strokes, 4, 2) * 2.0 - 1.0
    stroke_widths = torch.rand(batch, num_strokes) * 2.0 + 0.5
    opacities = torch.rand(batch, num_strokes) * 0.5 + 0.5
    return cubics.to(device), stroke_widths.to(device), opacities.to(device)


def _render_with_grads(cubics, stroke_widths, opacities, **kwargs):
    """Forward image plus MSE-loss gradients w.r.t. cubics/widths/opacities.

    The MSE target is drawn from a fixed seed on CPU so dense and tiled
    renders of the same scene are compared against the identical loss.
    """
    from pydiffvg.splat_render import splat_render_cubics

    c = cubics.clone().requires_grad_(True)
    s = stroke_widths.clone().requires_grad_(True)
    o = opacities.clone().requires_grad_(True)
    out = splat_render_cubics(c, s, opacities=o, **kwargs)

    torch.manual_seed(123)
    target = torch.rand(out.shape).to(out.device)
    loss = ((out - target) ** 2).mean()
    grads = torch.autograd.grad(loss, [c, s, o])
    return out.detach(), grads


def _assert_tiled_matches_dense(cubics, stroke_widths, opacities, tile_size, **kwargs):
    """Tiled forward and gradients agree with dense within the 1e-5 gate."""
    out_dense, grads_dense = _render_with_grads(
        cubics, stroke_widths, opacities, **kwargs
    )
    out_tiled, grads_tiled = _render_with_grads(
        cubics, stroke_widths, opacities,
        tiling="tiles", tile_size=tile_size, **kwargs,
    )

    assert out_tiled.shape == out_dense.shape
    assert torch.isfinite(out_tiled).all()
    assert (out_tiled - out_dense).abs().max().item() <= FWD_TOL
    for g_tiled, g_dense in zip(grads_tiled, grads_dense):
        assert torch.isfinite(g_tiled).all()
        assert (g_tiled - g_dense).abs().max().item() <= GRAD_TOL

    return out_tiled, grads_tiled


# ---------------------------------------------------------------------------
# 1. Randomized-scene exactness across gaussian counts and tile sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tile_size", [16, 32])
@pytest.mark.parametrize("G", [96, 2048, 10240])
def test_random_scene_matches_dense(device, tile_size, G):
    """Tiled forward and grads match dense <= 1e-5 at three gaussian counts.

    G = num_strokes * 16 samples. The G=2048 case drops to canvas 192 on CPU
    to keep the suite fast; G=10240 is CUDA-only. Measured max abs diffs:
    forward ~2-5e-7, gradients ~1e-10 to 6e-9.
    """
    if G == 10240 and device == "cpu":
        pytest.skip("too slow on CPU")

    num_strokes = G // NUM_SAMPLES
    assert num_strokes * NUM_SAMPLES == G

    canvas = 192 if (G == 2048 and device == "cpu") else 384
    cubics, stroke_widths, opacities = _make_scene(device, num_strokes, seed=G)
    _assert_tiled_matches_dense(
        cubics, stroke_widths, opacities, tile_size,
        canvas_size=canvas, num_samples=NUM_SAMPLES,
    )


# ---------------------------------------------------------------------------
# 2. Adversarial scenes
# ---------------------------------------------------------------------------

def _horizontal_stroke(y_px, canvas):
    """A horizontal cubic whose center line sits exactly at pixel row y_px.

    Control points are constant in y, so every sampled gaussian mean lands
    at y = y_px in canvas coordinates ((y_norm + 1) / 2 * canvas == y_px).
    """
    y = y_px / (canvas / 2.0) - 1.0
    return torch.tensor([[-0.8, y], [-0.3, y], [0.3, y], [0.8, y]])


@pytest.mark.parametrize("device", DEVICES)
def test_strokes_straddling_tile_borders(device):
    """Strokes centered exactly on and just off a tile boundary match dense.

    Canvas 64 with tile_size 16: one horizontal stroke at y = 16.0 (exactly
    on the border between tile rows 0 and 1) and one at y = 15.5 (a pixel
    center on the border's near side), plus 20 random strokes.
    """
    canvas = 64
    torch.manual_seed(42)
    random_cubics = torch.rand(20, 4, 2) * 2.0 - 1.0
    border_cubics = torch.stack(
        [_horizontal_stroke(16.0, canvas), _horizontal_stroke(15.5, canvas)]
    )
    cubics = torch.cat([border_cubics, random_cubics]).unsqueeze(0).to(device)

    num_strokes = cubics.shape[1]
    torch.manual_seed(43)
    stroke_widths = (torch.rand(1, num_strokes) * 2.0 + 0.5).to(device)
    opacities = (torch.rand(1, num_strokes) * 0.5 + 0.5).to(device)

    _assert_tiled_matches_dense(
        cubics, stroke_widths, opacities, 16,
        canvas_size=canvas, num_samples=NUM_SAMPLES,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_off_canvas_all_white_and_zero_grads(device):
    """Fully off-canvas gaussians: all-white output, exactly-zero gradients.

    With every gaussian culled the tiled path has zero (gaussian, tile)
    pairs; a graph-connection stub must keep backward() working and yield
    gradients that are exactly zero (matching the dense path) rather than
    raising "unused parameter".
    """
    from pydiffvg.splat_render import splat_render_cubics

    torch.manual_seed(0)
    # Control points around +5.0 map far outside the canvas.
    cubics = (torch.rand(1, 4, 4, 2) * 0.2 + 5.0).to(device).requires_grad_(True)
    stroke_widths = (torch.rand(1, 4) + 0.5).to(device).requires_grad_(True)
    opacities = torch.full((1, 4), 0.8, device=device, requires_grad=True)

    out = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=64, num_samples=NUM_SAMPLES,
        opacities=opacities, tiling="tiles", tile_size=16,
    )

    assert out.shape == (1, 64, 64)
    assert (out == 1.0).all()

    out.sum().backward()
    for param in (cubics, stroke_widths, opacities):
        assert param.grad is not None
        assert (param.grad == 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_on_and_off_canvas_matches_dense(device):
    """A scene mixing off-canvas and on-canvas strokes still matches dense."""
    torch.manual_seed(7)
    on_canvas = torch.rand(1, 5, 4, 2) * 2.0 - 1.0
    off_canvas = torch.rand(1, 5, 4, 2) * 0.2 + 5.0
    cubics = torch.cat([on_canvas, off_canvas], dim=1).to(device)

    torch.manual_seed(8)
    stroke_widths = (torch.rand(1, 10) * 2.0 + 0.5).to(device)
    opacities = (torch.rand(1, 10) * 0.5 + 0.5).to(device)

    for tile_size in (16, 32):
        _assert_tiled_matches_dense(
            cubics, stroke_widths, opacities, tile_size,
            canvas_size=64, num_samples=NUM_SAMPLES,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_near_zero_stroke_width_matches_dense(device):
    """Stroke width 1e-6: output and grads stay finite and match dense."""
    torch.manual_seed(11)
    cubics = (torch.rand(1, 6, 4, 2) * 2.0 - 1.0).to(device)
    stroke_widths = torch.full((1, 6), 1e-6, device=device)
    opacities = (torch.rand(1, 6) * 0.5 + 0.5).to(device)

    # _assert_tiled_matches_dense also asserts finiteness of output and grads.
    _assert_tiled_matches_dense(
        cubics, stroke_widths, opacities, 16,
        canvas_size=64, num_samples=NUM_SAMPLES,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_zero_opacity_strokes_match_dense(device):
    """Opacity exactly 0 for half the strokes: forward and grads match dense,
    including the gradient entries for the zero-opacity strokes."""
    torch.manual_seed(21)
    cubics = (torch.rand(1, 8, 4, 2) * 2.0 - 1.0).to(device)
    stroke_widths = (torch.rand(1, 8) * 2.0 + 0.5).to(device)
    opacities = torch.rand(1, 8) * 0.5 + 0.5
    opacities[:, :4] = 0.0
    opacities = opacities.to(device)

    out_tiled, grads_tiled = _assert_tiled_matches_dense(
        cubics, stroke_widths, opacities, 16,
        canvas_size=64, num_samples=NUM_SAMPLES,
    )
    # The zero-opacity entries still receive (finite) gradients.
    opacity_grad = grads_tiled[2]
    assert torch.isfinite(opacity_grad[:, :4]).all()


# ---------------------------------------------------------------------------
# 3. pixel_box composition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_pixel_box_tiled_matches_dense(device):
    """tiling="tiles" with a non-tile-aligned pixel_box matches dense.

    Box (17, 9, 24, 20) at canvas 64 with tile_size 16 is deliberately not
    aligned to the tile grid, so the padded-tile crop inside _splat_tiled is
    exercised. Output shape must be (B, 24, 20).
    """
    box = (17, 9, 24, 20)
    cubics, stroke_widths, opacities = _make_scene(device, num_strokes=6, seed=31)

    out_tiled, _ = _assert_tiled_matches_dense(
        cubics, stroke_widths, opacities, 16,
        canvas_size=64, num_samples=NUM_SAMPLES, pixel_box=box,
    )
    assert out_tiled.shape == (1, 24, 20)


# ---------------------------------------------------------------------------
# 4. Batched scenes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_batch_matches_dense(device):
    """B=3 batch with a different random scene per batch element."""
    cubics, stroke_widths, opacities = _make_scene(
        device, num_strokes=10, seed=51, batch=3
    )
    # Random draws differ across the batch dimension by construction.
    assert not torch.equal(cubics[0], cubics[1])
    assert not torch.equal(cubics[1], cubics[2])

    for tile_size in (16, 32):
        out_tiled, _ = _assert_tiled_matches_dense(
            cubics, stroke_widths, opacities, tile_size,
            canvas_size=128, num_samples=NUM_SAMPLES,
        )
        assert out_tiled.shape == (3, 128, 128)


# ---------------------------------------------------------------------------
# 5. Checkpointing on the tiled path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_tiled_checkpoint_matches_no_checkpoint(device):
    """Tiled use_checkpoint=True and False compute the same thing.

    Checkpointing only changes when the forward is recomputed, not what is
    computed, so on CPU the forward is bitwise identical and the gradients
    were measured bitwise identical. On CUDA the tiled path's scatter_add
    uses atomics whose accumulation order varies between calls, so bitwise
    equality does not hold there (measured forward diff ~1.2e-7, one ulp;
    gradient diff ~2e-9); assert <= 1e-6 instead.
    """
    cubics, stroke_widths, opacities = _make_scene(device, num_strokes=20, seed=61)

    results = {}
    for use_checkpoint in (True, False):
        results[use_checkpoint] = _render_with_grads(
            cubics, stroke_widths, opacities,
            canvas_size=96, num_samples=NUM_SAMPLES,
            tiling="tiles", tile_size=16, use_checkpoint=use_checkpoint,
        )

    out_ckpt, grads_ckpt = results[True]
    out_direct, grads_direct = results[False]

    if device == "cpu":
        assert torch.equal(out_ckpt, out_direct)
    else:
        assert (out_ckpt - out_direct).abs().max().item() <= 1e-6
    for g_ckpt, g_direct in zip(grads_ckpt, grads_direct):
        assert (g_ckpt - g_direct).abs().max().item() <= 1e-6


# ---------------------------------------------------------------------------
# 6. tiling="auto" dispatch
# ---------------------------------------------------------------------------

def _spy_on_tiled_path(monkeypatch):
    """Wrap _splat_tiled so calls through the module global are recorded."""
    from pydiffvg import splat_render

    calls = []
    original = splat_render._splat_tiled

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(splat_render, "_splat_tiled", spy)
    return calls


@pytest.mark.parametrize("device", DEVICES)
def test_auto_below_threshold_takes_dense_path(device, monkeypatch):
    """Below the threshold, "auto" is the dense path: bitwise equal output
    and _splat_tiled never invoked."""
    from pydiffvg import splat_render
    from pydiffvg.splat_render import splat_render_cubics

    monkeypatch.setattr(splat_render, "_TILING_AUTO_THRESHOLD_G", 1000)
    calls = _spy_on_tiled_path(monkeypatch)

    # G = 6 * 16 = 96 < 1000.
    cubics, stroke_widths, opacities = _make_scene(device, num_strokes=6, seed=71)
    kwargs = dict(canvas_size=64, num_samples=NUM_SAMPLES, opacities=opacities)

    out_none = splat_render_cubics(cubics, stroke_widths, tiling="none", **kwargs)
    out_auto = splat_render_cubics(cubics, stroke_widths, tiling="auto", **kwargs)

    assert calls == []
    assert torch.equal(out_auto, out_none)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_at_threshold_takes_tiled_path(device, monkeypatch):
    """At/above the threshold, "auto" is the tiled path: _splat_tiled invoked
    and the output equals an explicit tiling="tiles" render.

    On CPU the tiled path is deterministic per-process, so the comparison is
    bitwise. On CUDA scatter_add atomics reorder fp32 accumulation between
    runs (measured ~1.2e-7 run-to-run on the same inputs), so assert <= 1e-5.
    Note <= 1e-5 alone could not distinguish tiled from dense output (they
    agree to ~3e-7); path selection is proven by the _splat_tiled spy.
    """
    from pydiffvg import splat_render
    from pydiffvg.splat_render import splat_render_cubics

    monkeypatch.setattr(splat_render, "_TILING_AUTO_THRESHOLD_G", 1000)
    calls = _spy_on_tiled_path(monkeypatch)

    # G = 128 * 16 = 2048 >= 1000. Small canvas keeps the CPU run fast.
    cubics, stroke_widths, opacities = _make_scene(device, num_strokes=128, seed=72)
    kwargs = dict(
        canvas_size=96, num_samples=NUM_SAMPLES,
        opacities=opacities, tile_size=16,
    )

    out_tiles = splat_render_cubics(cubics, stroke_widths, tiling="tiles", **kwargs)
    assert calls == [1]
    out_auto = splat_render_cubics(cubics, stroke_widths, tiling="auto", **kwargs)
    assert calls == [1, 1]

    if device == "cpu":
        assert torch.equal(out_auto, out_tiles)
    else:
        assert (out_auto - out_tiles).abs().max().item() <= 1e-5


# ---------------------------------------------------------------------------
# 7. Validation
# ---------------------------------------------------------------------------

def test_tiling_kwarg_validation():
    """Invalid tiling / tile_size values raise ValueError."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, _ = _make_scene("cpu", num_strokes=1)

    with pytest.raises(ValueError):
        splat_render_cubics(cubics, stroke_widths, canvas_size=32, tiling="bogus")

    for bad_tile_size in (0, -8, 16.0, "16"):
        with pytest.raises(ValueError):
            splat_render_cubics(
                cubics, stroke_widths, canvas_size=32,
                tiling="tiles", tile_size=bad_tile_size,
            )

    # tile_size is only validated on the tiled path; the dense default
    # ignores it entirely.
    out = splat_render_cubics(
        cubics, stroke_widths, canvas_size=32, tiling="none", tile_size=0
    )
    assert out.shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# 8. Default-unchanged guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_tiling_none_bitwise_matches_default(device):
    """tiling="none" is bitwise identical to omitting the kwarg entirely."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene(device, num_strokes=6, seed=81)
    kwargs = dict(canvas_size=64, num_samples=NUM_SAMPLES, opacities=opacities)

    out_default = splat_render_cubics(cubics, stroke_widths, **kwargs)
    out_none = splat_render_cubics(cubics, stroke_widths, tiling="none", **kwargs)

    assert torch.equal(out_none, out_default)
    assert (out_none - out_default).abs().max().item() == 0.0
