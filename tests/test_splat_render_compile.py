"""Tests for the use_compile kwarg of splat_render_cubics.

use_compile=True runs the splat kernel through torch.compile. Compiled
outputs match eager to fp32 noise (not bitwise), so assertions use <= 1e-5.
Where inductor cannot build (missing compilers/headers), the renderer must
warn once and fall back to eager; tests exercise both branches.
"""

import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

CANVAS = 64
CHUNK = 300  # force multiple chunks (and differing last-chunk shape)


def _make_scene(device, num_strokes=4, seed=0):
    torch.manual_seed(seed)
    cubics = torch.rand(1, num_strokes, 4, 2) * 2.0 - 1.0
    stroke_widths = torch.rand(1, num_strokes) * 2.0 + 0.5
    opacities = torch.rand(1, num_strokes) * 0.5 + 0.5
    return cubics.to(device), stroke_widths.to(device), opacities.to(device)


def _render_grads(device, **kwargs):
    from pydiffvg.splat_render import splat_render_cubics

    cubics, stroke_widths, opacities = _make_scene(device)
    c = cubics.clone().requires_grad_(True)
    s = stroke_widths.clone().requires_grad_(True)
    o = opacities.clone().requires_grad_(True)
    out = splat_render_cubics(
        c, s, canvas_size=CANVAS, pixel_chunk_size=CHUNK, opacities=o, **kwargs
    )
    torch.manual_seed(1)
    target = torch.rand_like(out)
    loss = ((out - target) ** 2).mean()
    grads = torch.autograd.grad(loss, [c, s, o])
    return out.detach(), grads


@pytest.fixture
def clean_compile_state():
    """Reset the module-level compile caches around a test."""
    from pydiffvg import splat_render

    saved_ok = dict(splat_render._COMPILE_OK)
    saved_fn = splat_render._COMPILED_SPLAT_CHUNK
    splat_render._COMPILE_OK.clear()
    splat_render._COMPILED_SPLAT_CHUNK = None
    yield
    splat_render._COMPILE_OK.clear()
    splat_render._COMPILE_OK.update(saved_ok)
    splat_render._COMPILED_SPLAT_CHUNK = saved_fn


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_checkpoint", [True, False])
def test_use_compile_matches_eager(device, use_checkpoint):
    """Compiled kernel matches eager to <= 1e-5 (measured ~1e-7) in forward
    and gradients, under both checkpointing modes."""
    from pydiffvg import splat_render

    if not splat_render._compile_available(torch.device(device)):
        pytest.skip(f"torch.compile unavailable on {device}")

    out_eager, grads_eager = _render_grads(device, use_checkpoint=use_checkpoint)
    out_comp, grads_comp = _render_grads(
        device, use_checkpoint=use_checkpoint, use_compile=True
    )

    assert (out_eager - out_comp).abs().max().item() <= 1e-5
    for g_e, g_c in zip(grads_eager, grads_comp):
        assert (g_e - g_c).abs().max().item() <= 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_use_compile_with_pixel_box(device):
    """use_compile composes with pixel_box."""
    from pydiffvg import splat_render

    if not splat_render._compile_available(torch.device(device)):
        pytest.skip(f"torch.compile unavailable on {device}")

    box = (17, 9, 24, 20)
    out_eager, grads_eager = _render_grads(device, pixel_box=box)
    out_comp, grads_comp = _render_grads(device, pixel_box=box, use_compile=True)

    assert out_comp.shape == (1, 24, 20)
    assert (out_eager - out_comp).abs().max().item() <= 1e-5
    for g_e, g_c in zip(grads_eager, grads_comp):
        assert (g_e - g_c).abs().max().item() <= 1e-5


def test_fallback_warns_once_and_matches_eager(clean_compile_state, monkeypatch):
    """When the compile preflight fails, use_compile=True warns once and
    produces the eager result bitwise (torch.compile is never attempted
    in-process, since a failed in-process compile can poison dynamo state)."""
    from pydiffvg import splat_render

    monkeypatch.setattr(
        splat_render, "_run_compile_preflight",
        lambda device_type: (False, "simulated inductor failure"),
    )

    def broken_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not run in-process after a failed preflight")

    monkeypatch.setattr(torch, "compile", broken_compile)

    out_eager, _ = _render_grads("cpu")
    with pytest.warns(RuntimeWarning, match="preflight"):
        out_fallback, _ = _render_grads("cpu", use_compile=True)
    assert torch.equal(out_eager, out_fallback)

    # Second call: cached failure, no new warning.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning would raise
        out_again, _ = _render_grads("cpu", use_compile=True)
    assert torch.equal(out_eager, out_again)


def test_default_path_never_touches_compile(clean_compile_state, monkeypatch):
    """use_compile=False (the default) must invoke neither torch.compile nor
    the preflight subprocess."""
    from pydiffvg import splat_render

    def broken_compile(*args, **kwargs):
        raise AssertionError("torch.compile must not be called on the default path")

    def broken_preflight(device_type):
        raise AssertionError("the preflight must not run on the default path")

    monkeypatch.setattr(torch, "compile", broken_compile)
    monkeypatch.setattr(splat_render, "_run_compile_preflight", broken_preflight)

    out, grads = _render_grads("cpu")  # would raise if compile were touched
    assert out.shape == (1, CANVAS, CANVAS)
    assert all(g.abs().sum() > 0 for g in grads)
