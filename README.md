# easydiffvg

Pure-PyTorch differentiable vector graphics — a drop-in replacement for
[diffvg](https://github.com/BachiLi/diffvg) with **no C++ compilation and no
CUDA toolkit required**. `uv add` / `pip install` it and it works, on any
device PyTorch supports.

The package installs as `pydiffvg` and mirrors the original API: shapes,
shape groups, SVG I/O, and a differentiable rasterizer whose gradients flow
through both shape geometry and colors.

```python
import torch
import pydiffvg

circle = pydiffvg.Circle(
    radius=torch.tensor(20.0, requires_grad=True),
    center=torch.tensor([32.0, 32.0], requires_grad=True),
)
group = pydiffvg.ShapeGroup(
    shape_ids=torch.tensor([0]),
    fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]),
)

img = pydiffvg.render(64, 64, [circle], [group])   # (64, 64, 4) RGBA
loss = ((img - target) ** 2).mean()
loss.backward()                                     # gradients w.r.t. radius/center
```

## Two renderers

**`RenderFunction` / `render`** — faithful reimplementation of diffvg's
rasterizer (winding numbers, distance fields, boundary sampling for gradients
via Reynolds transport). Use it when you need diffvg-compatible output for
arbitrary shapes, fills, and gradients.

**`splat_render_cubics` / `SplatRenderFunction`** — a fast gaussian-splatting
renderer for cubic Bézier *strokes* (based on
[Bézier splatting, arXiv:2503.16424](https://arxiv.org/abs/2503.16424)).
Batched, fully differentiable, built for optimization inner loops that render
thousands of times:

```python
from pydiffvg import splat_render_cubics

# cubics: (B, num_strokes, 4, 2) control points in [-1, 1]
# widths: (B, num_strokes) gaussian sigma in pixels
img = splat_render_cubics(cubics, widths, canvas_size=384,
                          num_samples=16, opacities=opacities)  # (B, H, W)
```

White background (1.0), black ink (0.0); the compositor is an
order-independent `1 − ∏(1−αᵢ)`, so disjoint stroke sets composite exactly by
multiplication.

### Performance knobs (all opt-in; defaults reproduce the baseline bit-for-bit)

| kwarg | what it does | when to use |
|---|---|---|
| `pixel_box=(y0, x0, h, w)` | Rasterize only that window, returning `(B, h, w)` — exactly the full render's slice | Local-window fitting; 63 → 2.9 ms/iter on a 6-stroke 96×96-window fit (21.9×) |
| `use_checkpoint=False` | Skip gradient checkpointing (recompute overhead) | Small gaussian counts; keep `True` at large counts to bound memory |
| `use_compile=True` | Run the splat kernel through `torch.compile` (falls back to eager with a warning if inductor is unavailable) | ~1.7× at 100 gaussians, ~8.6× at 10k; costs seconds of compile time on first call per shape |
| `tiling="tiles"` (or `"auto"`), `tile_size` | Tile-culled evaluation: each gaussian only touches the tiles its ~4.5σ support overlaps | Won every measured config: 8–22× on full frames (G = 96…40,960), 1.4× even on a 96×96 window, lower peak memory at scale; `"auto"` currently always tiles |
| `tiling="triton"`, `tile_size` | Same tile culling as `"tiles"`, but the per-tile evaluation runs as Triton kernels (CUDA + fp32 only; `tile_size=16` measured fastest) | ~16× over `tiling="tiles"` at 10k gaussians (47.0 → 3.0 ms/iter, canvas 384), up to ~34× at 768² (136.5 → 4.0 ms/iter; gains shrink toward ~1.7× at ~100 gaussians); peak memory 33 MB vs 728 MB at G=10,240. First call per shape pays Triton JIT compile latency (seconds) |

Numbers above: RTX 5090, fp32, forward+backward, canvas 384. Reproduce with
the scripts in `benchmarks/`. Exactness: `pixel_box` and `use_checkpoint` are
bitwise-identical to the baseline; `use_compile` and `tiling` match to fp32
noise (≲1e-6; gated at 1e-5 in tests, forward and gradients).

## Install & development

```bash
uv sync              # install deps + package (editable)
uv run pytest        # test suite
```

Python ≥ 3.12, PyTorch ≥ 2.10. Dependency management uses
[uv](https://docs.astral.sh/uv/) — prefer `uv add` over pip.

## Repository layout

| path | contents |
|---|---|
| `pydiffvg/shapes.py`, `groups.py`, `color.py` | Shape primitives, ShapeGroup, colors/gradients |
| `pydiffvg/render.py`, `rasterize.py`, `gradients.py` | diffvg-compatible renderer + boundary-sampling backward |
| `pydiffvg/splat_render.py` | Gaussian-splatting stroke renderer (`splat_render_cubics`) |
| `pydiffvg/svg/` | SVG parsing and saving |
| `pydiffvg/utils/` | Bézier math, winding numbers, distance fields |
| `benchmarks/` | Renderer benchmarks and torch.compile / precision experiments |
| `tests/` | pytest suite, including bitwise/1e-5 exactness gates for every performance path |
| `.original_diffvg/` | The original diffvg source, kept as the reference implementation |
