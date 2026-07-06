"""Tests for the speed-oriented kwargs of splat_render_cubics.

Covers:
- pixel_box: windowed rendering must exactly match slicing the full render,
  in both the forward pass and the gradients.
- use_checkpoint=False: skipping gradient checkpointing must not change
  outputs or gradients.
- _PIXEL_GRID_CACHE: LRU cache of flattened pixel grids.
"""

import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Small case: odd offsets and a pixel_chunk_size that forces multiple chunks
# (24 * 20 = 480 window pixels / 300 per chunk = 2 chunks).
SMALL_CANVAS = 64
SMALL_BOX = (17, 9, 24, 20)
SMALL_CHUNK = 300

# Larger case mirroring downstream usage: canvas 384, 6 strokes, 16 samples.
LARGE_CANVAS = 384
LARGE_BOX = (144, 144, 96, 96)


def _make_scene(device, num_strokes=3, seed=0):
    """Random cubics in [-1, 1], widths, and opacities on the given device."""
    torch.manual_seed(seed)
    cubics = torch.rand(1, num_strokes, 4, 2) * 2.0 - 1.0
    stroke_widths = torch.rand(1, num_strokes) * 2.0 + 0.5
    opacities = torch.rand(1, num_strokes) * 0.5 + 0.5
    return cubics.to(device), stroke_widths.to(device), opacities.to(device)


def _window_grads(
    cubics,
    stroke_widths,
    opacities,
    canvas_size,
    num_samples,
    pixel_chunk_size,
    pixel_box,
    loss_fn,
    via_slice,
):
    """Gradients of a window loss w.r.t. cubics, stroke_widths, opacities.

    via_slice=False renders only the window (pixel_box path); via_slice=True
    renders the full canvas and slices the window out afterwards.
    """
    from pydiffvg.splat_render import splat_render_cubics

    c = cubics.clone().requires_grad_(True)
    s = stroke_widths.clone().requires_grad_(True)
    o = opacities.clone().requires_grad_(True)

    y0, x0, h, w = pixel_box
    if via_slice:
        full = splat_render_cubics(
            c, s,
            canvas_size=canvas_size,
            num_samples=num_samples,
            pixel_chunk_size=pixel_chunk_size,
            opacities=o,
        )
        out = full[:, y0:y0 + h, x0:x0 + w]
    else:
        out = splat_render_cubics(
            c, s,
            canvas_size=canvas_size,
            num_samples=num_samples,
            pixel_chunk_size=pixel_chunk_size,
            opacities=o,
            pixel_box=pixel_box,
        )

    loss = loss_fn(out)
    return torch.autograd.grad(loss, [c, s, o])


@pytest.mark.parametrize("device", DEVICES)
def test_pixel_box_forward_matches_slice(device):
    """Windowed forward render is bitwise identical to the full-render slice."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene(device)

    full = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
    )
    window = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
        pixel_box=SMALL_BOX,
    )

    y0, x0, h, w = SMALL_BOX
    assert window.shape == (1, h, w)
    sliced = full[:, y0:y0 + h, x0:x0 + w]
    assert torch.equal(window, sliced)
    assert (window - sliced).abs().max().item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_pixel_box_full_canvas_equals_full_render(device):
    """pixel_box=(0, 0, H, W) is bitwise identical to the default full render."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene(device)

    full = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
    )
    boxed = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
        pixel_box=(0, 0, SMALL_CANVAS, SMALL_CANVAS),
    )

    assert boxed.shape == full.shape
    assert torch.equal(boxed, full)
    assert (boxed - full).abs().max().item() == 0.0


def test_pixel_box_forward_matches_slice_large():
    """384-canvas case mirroring downstream usage: 6 strokes, 16 samples."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene("cpu", num_strokes=6)

    full = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=LARGE_CANVAS,
        num_samples=16,
        opacities=opacities,
    )
    window = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=LARGE_CANVAS,
        num_samples=16,
        opacities=opacities,
        pixel_box=LARGE_BOX,
    )

    y0, x0, h, w = LARGE_BOX
    assert window.shape == (1, h, w)
    sliced = full[:, y0:y0 + h, x0:x0 + w]
    assert torch.equal(window, sliced)
    assert (window - sliced).abs().max().item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_pixel_box_gradients_match_slice(device):
    """MSE-window-loss gradients agree between the pixel_box path and the
    full-render-then-slice path, w.r.t. cubics, stroke widths, and opacities.

    Measured max abs diff is ~3e-8 (both CPU and CUDA); assert <= 1e-6.
    """
    cubics, stroke_widths, opacities = _make_scene(device)

    _, _, h, w = SMALL_BOX
    torch.manual_seed(123)
    target = torch.rand(1, h, w).to(device)

    def mse_loss(out):
        return ((out - target) ** 2).mean()

    grads_window = _window_grads(
        cubics, stroke_widths, opacities,
        SMALL_CANVAS, 20, SMALL_CHUNK, SMALL_BOX, mse_loss, via_slice=False,
    )
    grads_slice = _window_grads(
        cubics, stroke_widths, opacities,
        SMALL_CANVAS, 20, SMALL_CHUNK, SMALL_BOX, mse_loss, via_slice=True,
    )

    for g_win, g_full in zip(grads_window, grads_slice):
        assert (g_win - g_full).abs().max().item() <= 1e-6


def test_pixel_box_gradients_match_slice_large():
    """Gradient agreement for the 384-canvas downstream-style case.

    Measured max abs diff is ~8e-9; assert <= 1e-6.
    """
    cubics, stroke_widths, opacities = _make_scene("cpu", num_strokes=6)

    _, _, h, w = LARGE_BOX
    torch.manual_seed(123)
    target = torch.rand(1, h, w)

    def mse_loss(out):
        return ((out - target) ** 2).mean()

    grads_window = _window_grads(
        cubics, stroke_widths, opacities,
        LARGE_CANVAS, 16, 2048, LARGE_BOX, mse_loss, via_slice=False,
    )
    grads_slice = _window_grads(
        cubics, stroke_widths, opacities,
        LARGE_CANVAS, 16, 2048, LARGE_BOX, mse_loss, via_slice=True,
    )

    for g_win, g_full in zip(grads_window, grads_slice):
        assert (g_win - g_full).abs().max().item() <= 1e-6


def test_pixel_box_gradients_stress_loss_relative():
    """Harsher loss (randn-weighted sum) only agrees to relative precision.

    The window path and the slice path chunk pixels differently, so float32
    sums accumulate in a different order. With gradients of magnitude ~50 the
    resulting accumulation-order noise reaches a few 1e-6 in absolute terms
    (measured max abs diff ~4e-6), which is pure fp32 round-off, not a bug.
    Hence torch.allclose with rtol/atol rather than a tight absolute bound.
    """
    cubics, stroke_widths, opacities = _make_scene("cpu")

    _, _, h, w = SMALL_BOX
    torch.manual_seed(123)
    weights = torch.randn(1, h, w)

    def stress_loss(out):
        return (out * weights).sum()

    grads_window = _window_grads(
        cubics, stroke_widths, opacities,
        SMALL_CANVAS, 20, SMALL_CHUNK, SMALL_BOX, stress_loss, via_slice=False,
    )
    grads_slice = _window_grads(
        cubics, stroke_widths, opacities,
        SMALL_CANVAS, 20, SMALL_CHUNK, SMALL_BOX, stress_loss, via_slice=True,
    )

    for g_win, g_full in zip(grads_window, grads_slice):
        assert torch.allclose(g_win, g_full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_use_checkpoint_false_matches_default(device):
    """use_checkpoint=False must reproduce the default (checkpointed) path.

    Checkpointing only changes when the forward is recomputed, not what is
    computed, so the forward is bitwise identical and gradients were measured
    bitwise identical on both CPU and CUDA; assert <= 1e-6 on the gradients
    to stay robust to nondeterministic kernels.
    """
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene(device)

    out_default = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
    )
    out_direct = splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
        use_checkpoint=False,
    )
    assert torch.equal(out_default, out_direct)

    torch.manual_seed(7)
    target = torch.rand(1, SMALL_CANVAS, SMALL_CANVAS).to(device)

    grads = []
    for use_checkpoint in (True, False):
        c = cubics.clone().requires_grad_(True)
        s = stroke_widths.clone().requires_grad_(True)
        o = opacities.clone().requires_grad_(True)
        out = splat_render_cubics(
            c, s,
            canvas_size=SMALL_CANVAS,
            pixel_chunk_size=SMALL_CHUNK,
            opacities=o,
            use_checkpoint=use_checkpoint,
        )
        loss = ((out - target) ** 2).mean()
        grads.append(torch.autograd.grad(loss, [c, s, o]))

    for g_ckpt, g_direct in zip(grads[0], grads[1]):
        assert (g_ckpt - g_direct).abs().max().item() <= 1e-6


def test_pixel_box_validation():
    """Invalid pixel_box values raise ValueError."""
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, _ = _make_scene("cpu", num_strokes=1)
    canvas = 32

    def render(box):
        return splat_render_cubics(
            cubics, stroke_widths, canvas_size=canvas, pixel_box=box
        )

    # Zero or negative height/width.
    with pytest.raises(ValueError):
        render((0, 0, 0, 10))
    with pytest.raises(ValueError):
        render((0, 0, 10, 0))
    with pytest.raises(ValueError):
        render((0, 0, -5, 10))
    with pytest.raises(ValueError):
        render((0, 0, 10, -5))

    # Box exceeding the canvas.
    with pytest.raises(ValueError):
        render((30, 0, 10, 10))  # y0 + h > canvas
    with pytest.raises(ValueError):
        render((0, 30, 10, 10))  # x0 + w > canvas
    with pytest.raises(ValueError):
        render((-1, 0, 10, 10))  # negative y0
    with pytest.raises(ValueError):
        render((0, -1, 10, 10))  # negative x0

    # A valid box at the canvas boundary must not raise.
    out = render((22, 22, 10, 10))
    assert out.shape == (1, 10, 10)


def test_pixel_grid_cache_reuse():
    """Repeated calls reuse the cached pixel grid and produce identical output."""
    from pydiffvg import splat_render
    from pydiffvg.splat_render import _PIXEL_GRID_CACHE, splat_render_cubics

    _PIXEL_GRID_CACHE.clear()

    cubics, stroke_widths, opacities = _make_scene("cpu")

    kwargs = dict(
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
        pixel_box=SMALL_BOX,
    )
    out_first = splat_render_cubics(cubics, stroke_widths, **kwargs)

    boxed_key = (SMALL_CANVAS, SMALL_BOX, cubics.device, cubics.dtype)
    assert boxed_key in _PIXEL_GRID_CACHE
    assert len(_PIXEL_GRID_CACHE) == 1
    cached_grid = _PIXEL_GRID_CACHE[boxed_key]

    out_second = splat_render_cubics(cubics, stroke_widths, **kwargs)
    assert torch.equal(out_first, out_second)
    # Same call signature must not grow the cache, and must reuse the same tensor.
    assert len(_PIXEL_GRID_CACHE) == 1
    assert _PIXEL_GRID_CACHE[boxed_key] is cached_grid

    # A full-canvas render uses a separate (pixel_box=None) cache entry.
    splat_render_cubics(
        cubics, stroke_widths,
        canvas_size=SMALL_CANVAS,
        pixel_chunk_size=SMALL_CHUNK,
        opacities=opacities,
    )
    full_key = (SMALL_CANVAS, None, cubics.device, cubics.dtype)
    assert full_key in _PIXEL_GRID_CACHE
    assert len(_PIXEL_GRID_CACHE) == 2

    # The module-level cache object is the one splat_render exposes.
    assert splat_render._PIXEL_GRID_CACHE is _PIXEL_GRID_CACHE


def test_pixel_grid_cache_lru_eviction(monkeypatch):
    """With the cap lowered to 2, a third distinct key evicts the oldest."""
    from pydiffvg import splat_render
    from pydiffvg.splat_render import splat_render_cubics

    # Eviction reads the module-level constant at call time, so patching the
    # module global takes effect for subsequent calls.
    monkeypatch.setattr(splat_render, "_PIXEL_GRID_CACHE_MAX_ENTRIES", 2)
    splat_render._PIXEL_GRID_CACHE.clear()

    cubics, stroke_widths, _ = _make_scene("cpu", num_strokes=1)

    canvas_sizes = [16, 24, 32]
    for canvas_size in canvas_sizes:
        splat_render_cubics(cubics, stroke_widths, canvas_size=canvas_size)

    assert len(splat_render._PIXEL_GRID_CACHE) == 2

    device, dtype = cubics.device, cubics.dtype
    # Oldest key (canvas 16) was evicted; the two most recent remain.
    assert (16, None, device, dtype) not in splat_render._PIXEL_GRID_CACHE
    assert (24, None, device, dtype) in splat_render._PIXEL_GRID_CACHE
    assert (32, None, device, dtype) in splat_render._PIXEL_GRID_CACHE
