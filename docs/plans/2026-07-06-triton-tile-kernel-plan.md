# Triton Tile Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Triton-kernel rendering path (`tiling="triton"`) to `splat_render_cubics` that reproduces the tiled path's output within the 1e-5 exactness gates at ≥5× its speed at G=10,240 (validated spike: 16.9×).

**Architecture:** Reuse the existing PyTorch pair-building (`_build_tile_pairs`), add a sort-into-per-tile-segments step, then one Triton program per (batch, tile) segment holding the per-pixel log-transmittance accumulator in registers (single store per pixel, no atomics in forward). Backward is a second kernel that recomputes alpha and atomically accumulates per-gaussian parameter gradients. A `torch.autograd.Function` wraps ONLY the accumulator computation; `1 − exp(accum)`, cropping, and the final clamp stay in PyTorch autograd.

**Tech Stack:** PyTorch ≥ 2.10 (cu128), Triton 3.6.0 (ships with the torch CUDA wheel — no new dependency), pytest, uv.

## Context for a team with zero prior exposure (read before Task 1)

- Repo: `~/code/easydiffvg`, pure-PyTorch diffvg replacement; the file that matters is `pydiffvg/splat_render.py`. Read its module docstring, `_splat_chunk`, `_splat_tiled`, `_build_tile_pairs`, `_splat_tile_chunk`, and `splat_render_cubics` end to end before writing anything. Use `uv run` for everything; never pip.
- The renderer composites strokes as `1 − ∏(1 − αᵢ)` — an **order-independent product**. That is why per-pixel accumulation of `log1p(−α)` needs no depth sort and why per-tile gaussian order is irrelevant.
- **The gray-wash invariant (do not regress):** every alpha is hard-cut with `alpha * (mahal_sq < 20.0)` (strict `<`). History: clamping without zeroing left an `exp(−10)` alpha floor that composited into a visible gray background at ~40k gaussians. The Triton kernels below replicate this exactly (`tl.where(m < 20.0, ...)`); any change to it must fail loudly in review.
- The tiled PyTorch path (`tiling="tiles"`) is the correctness reference for the Triton path; the dense path (`tiling="none"`) is the ground truth both are gated against (≤1e-5 forward and gradients — the same gates in `tests/test_splat_render_tiling.py`).
- A working spike of both kernels (forward + hand-derived backward) was validated on 2026-07-06 on the RTX 5090: forward ≤3.6e-7 and gradients ≤1.5e-8 vs dense across tile 16/32 and odd canvases; 16.9× over PyTorch-tiled at G=10,240 (2.8 ms/iter fwd+bwd, 33 MB peak vs 727 MB), 25.6× at G=40,960. The code in Tasks 2–3 is that spike, verbatim. Treat deviations from it as regressions to explain, not choices.

## Environment prerequisites & landmines

1. **PR #6 must be merged first** (it lands the tiled path on main — PR #5 was accidentally merged into a side branch). Verify: `git show origin/main:pydiffvg/splat_render.py | grep -c _build_tile_pairs` must print ≥ 1. If 0, stop and escalate.
2. **Python.h landmine:** Triton's launcher build (and torch.compile) fails on this machine without headers. Prefix every GPU command with
   `CPATH=/home/naka/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12`
   (headers came from `uv python install 3.12`). Symptom without it: `CalledProcessError` on `/usr/bin/gcc ... __triton_launcher.c`.
3. **This Triton version has no `tl.math.log1p`.** Use `import triton.language.extra.libdevice as libdevice` → `libdevice.log1p(...)`. Plain `tl.log(1-α)` is NOT acceptable (precision loss for α ≲ 6e-8 breaks parity with the PyTorch path, which uses `torch.log1p`).
4. Module-level Triton constants must be `tl.constexpr(...)` instances (e.g. `_MAHAL = tl.constexpr(20.0)`), not plain floats.
5. A long training job may own the GPU. `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l` must print 0 before benchmarks; correctness tests at small sizes are fine anytime.
6. Do not touch `_splat_chunk` or any default-path code. The byte-identity harness (Task 6) will catch you if you do.

## Global Constraints

- All changes additive: new `tiling="triton"` value + new module `pydiffvg/splat_triton.py`; every existing kwarg default and code path byte-identical (forward AND gradients, verified vs origin/main).
- The alpha cutoff `alpha * (mahal_sq < 20.0)` semantics preserved exactly (strict `<`; clamp at 20 inside `exp`).
- Triton path: CUDA + float32 only in v1 — raise `RuntimeError` with an actionable message otherwise (no silent fallback; callers opt in explicitly).
- Exactness gates: ≤1e-5 max abs diff vs dense, forward and `autograd.grad` w.r.t. cubics/stroke_widths/opacities, on every case in the existing tiling test suite.
- No new dependencies. Python ≥ 3.12, `uv` only.
- Acceptance perf bar: ≥5× over `tiling="tiles"` at G=10,240, canvas 384, on the RTX 5090 (spike: 16.9×; the margin allows for integration overhead).
- Work on a branch off main; PR at the end; do not merge.

---

### Task 1: Segment builder (`pydiffvg/splat_triton.py`)

**Files:**
- Create: `pydiffvg/splat_triton.py`
- Test: `tests/test_splat_triton.py`

**Interfaces:**
- Consumes: `_build_tile_pairs(means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg, region, tile_size, n_tx, n_ty) -> (pair_gauss, pair_tile_x, pair_tile_y)` from `pydiffvg.splat_render` (all `torch.long`, `pair_gauss` indexes the flattened `(B*G,)` parameter arrays).
- Produces: `build_tile_segments(pair_gauss, pair_tile_x, pair_tile_y, G, n_tx, n_ty) -> TileSegments` where `TileSegments` is a NamedTuple of `pg (long, N)`, `seg_start (int32, S)`, `seg_count (int32, S)`, `seg_b (int32, S)`, `seg_ty (int32, S)`, `seg_tx (int32, S)` — pairs sorted so each segment `[seg_start[i], seg_start[i]+seg_count[i])` of `pg` holds exactly the gaussians of one (batch, tile).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splat_triton.py
"""Tests for the Triton splat path. GPU-only tests skip cleanly without CUDA."""
import pytest
import torch


def test_build_tile_segments_partitions_pairs():
    from pydiffvg.splat_triton import build_tile_segments

    # Hand-built pairs: gaussians 0..4 (G=3 per batch, so 3,4 are batch 1),
    # tiles in a 2x2 grid (n_tx=n_ty=2).
    pair_gauss = torch.tensor([0, 0, 1, 2, 3, 4, 4])
    pair_tx = torch.tensor([0, 1, 0, 1, 0, 0, 1])
    pair_ty = torch.tensor([0, 0, 1, 1, 0, 1, 1])
    segs = build_tile_segments(pair_gauss, pair_tx, pair_ty, G=3, n_tx=2, n_ty=2)

    # Every pair appears exactly once across all segments.
    total = int(segs.seg_count.sum().item())
    assert total == 7
    # Segments are consistent: reconstruct (b, ty, tx, gauss) triples and
    # compare as sets against the input.
    got = set()
    for i in range(segs.seg_start.shape[0]):
        s, c = int(segs.seg_start[i]), int(segs.seg_count[i])
        for g in segs.pg[s:s + c].tolist():
            got.add((int(segs.seg_b[i]), int(segs.seg_ty[i]), int(segs.seg_tx[i]), g))
    expect = set()
    for g, tx, ty in zip(pair_gauss.tolist(), pair_tx.tolist(), pair_ty.tolist()):
        expect.add((g // 3, ty, tx, g))
    assert got == expect
    # Dtypes the kernel relies on.
    assert segs.pg.dtype == torch.long
    for t in (segs.seg_start, segs.seg_count, segs.seg_b, segs.seg_ty, segs.seg_tx):
        assert t.dtype == torch.int32


def test_build_tile_segments_empty():
    from pydiffvg.splat_triton import build_tile_segments

    empty = torch.empty(0, dtype=torch.long)
    segs = build_tile_segments(empty, empty, empty, G=4, n_tx=3, n_ty=3)
    assert segs.seg_start.shape[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splat_triton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pydiffvg.splat_triton'`

- [ ] **Step 3: Write the implementation**

```python
# pydiffvg/splat_triton.py
"""Triton tile kernels for the splat renderer (tiling="triton").

One Triton program per (batch, tile) segment; the per-pixel
log-transmittance accumulator lives in registers, so the forward does a
single global store per pixel and needs no atomics. The backward kernel
recomputes alpha and atomically accumulates per-gaussian parameter grads.
Only the accumulator is computed here; 1 - exp(accum), cropping, and the
output clamp remain ordinary PyTorch autograd (see splat_render.py).

Requires CUDA + float32. Triton ships with the torch CUDA wheels.
"""

from typing import NamedTuple

import torch


class TileSegments(NamedTuple):
    pg: torch.Tensor         # (N,) long — sorted gaussian indices into (B*G,)
    seg_start: torch.Tensor  # (S,) int32 — segment offsets into pg
    seg_count: torch.Tensor  # (S,) int32
    seg_b: torch.Tensor      # (S,) int32 — batch index per segment
    seg_ty: torch.Tensor     # (S,) int32 — tile row per segment
    seg_tx: torch.Tensor     # (S,) int32 — tile col per segment


def build_tile_segments(pair_gauss, pair_tile_x, pair_tile_y, G, n_tx, n_ty):
    """Sort (gaussian, tile) pairs into contiguous per-(batch, tile) segments."""
    device = pair_gauss.device
    if pair_gauss.numel() == 0:
        i32 = lambda: torch.empty(0, dtype=torch.int32, device=device)  # noqa: E731
        return TileSegments(pair_gauss, i32(), i32(), i32(), i32(), i32())
    pair_b = pair_gauss // G
    key = (pair_b * n_ty + pair_tile_y) * n_tx + pair_tile_x
    order = torch.argsort(key)
    pg = pair_gauss[order].contiguous()
    keys_sorted = key[order]
    uniq, counts = torch.unique_consecutive(keys_sorted, return_counts=True)
    seg_count = counts.to(torch.int32)
    seg_start = torch.zeros_like(seg_count)
    seg_start[1:] = counts.cumsum(0)[:-1].to(torch.int32)
    seg_b = (uniq // (n_ty * n_tx)).to(torch.int32)
    rem = uniq % (n_ty * n_tx)
    return TileSegments(
        pg, seg_start, seg_count, seg_b,
        (rem // n_tx).to(torch.int32), (rem % n_tx).to(torch.int32),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_splat_triton.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add pydiffvg/splat_triton.py tests/test_splat_triton.py
git commit -m "feat(splat): tile-segment builder for the Triton path"
```

---

### Task 2: Forward kernel

**Files:**
- Modify: `pydiffvg/splat_triton.py` (append)
- Test: `tests/test_splat_triton.py` (append)

**Interfaces:**
- Consumes: `TileSegments` from Task 1.
- Produces: `_fwd_kernel` (Triton JIT) and `triton_available() -> bool` (True iff CUDA is available and `import triton` succeeds). Task 3 wraps `_fwd_kernel` in the autograd Function; Task 4 uses `triton_available()`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_splat_triton.py`)

```python
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _scene(strokes, seed=0, device="cuda"):
    torch.manual_seed(seed)
    c = (torch.rand(1, strokes, 4, 2, device=device) * 2 - 1).requires_grad_(True)
    w = (torch.rand(1, strokes, device=device) * 3 + 0.5).requires_grad_(True)
    o = (torch.rand(1, strokes, device=device) * 0.5 + 0.5).requires_grad_(True)
    return c, w, o


@cuda_only
@pytest.mark.parametrize("tile", [16, 32])
def test_triton_forward_matches_dense(tile):
    from pydiffvg.splat_render import splat_render_cubics

    c, w, o = _scene(6)
    dense = splat_render_cubics(c, w, canvas_size=384, num_samples=16, opacities=o)
    tri = splat_render_cubics(c, w, canvas_size=384, num_samples=16, opacities=o,
                              tiling="triton", tile_size=tile)
    assert (dense - tri).abs().max().item() <= 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CPATH=/home/naka/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 uv run pytest tests/test_splat_triton.py -v -k triton_forward`
Expected: FAIL with `ValueError: tiling must be "none", "tiles" or "auto"` (the kwarg value does not exist until Task 4 — that is fine; this test goes green at the end of Task 4. To validate the kernel *now*, Step 3 also adds a direct-call test that passes within this task.)

- [ ] **Step 3: Write the kernel + availability probe + direct test** (append to `pydiffvg/splat_triton.py`)

This code is spike-validated verbatim (2026-07-06, RTX 5090) — copy exactly:

```python
try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as libdevice

    _TRITON_IMPORTED = True
except Exception:  # pragma: no cover - platforms without triton
    _TRITON_IMPORTED = False


def triton_available() -> bool:
    return _TRITON_IMPORTED and torch.cuda.is_available()


if _TRITON_IMPORTED:
    # Must match the literals in splat_render._splat_chunk / _MAHAL_SQ_CUTOFF.
    _MAHAL = tl.constexpr(20.0)

    @triton.jit
    def _fwd_kernel(
        pg_ptr, seg_start_ptr, seg_count_ptr, seg_b_ptr, seg_ty_ptr, seg_tx_ptr,
        mx_ptr, my_ptr, cos_ptr, sin_ptr, isa_ptr, isc_ptr, op_ptr,
        accum_ptr,
        x0, y0, Hp, Wp,
        TILE: tl.constexpr, BLOCK_G: tl.constexpr, EPS_C: tl.constexpr,
    ):
        seg = tl.program_id(0)
        start = tl.load(seg_start_ptr + seg)
        count = tl.load(seg_count_ptr + seg)
        b = tl.load(seg_b_ptr + seg)
        ty = tl.load(seg_ty_ptr + seg)
        tx = tl.load(seg_tx_ptr + seg)

        offs = tl.arange(0, TILE * TILE)
        py_l = offs // TILE
        px_l = offs % TILE
        px = x0 + (tx * TILE + px_l).to(tl.float32) + 0.5
        py = y0 + (ty * TILE + py_l).to(tl.float32) + 0.5

        acc = tl.zeros([TILE * TILE], dtype=tl.float32)
        for i in range(0, count, BLOCK_G):
            g_off = i + tl.arange(0, BLOCK_G)
            g_mask = g_off < count
            g_idx = tl.load(pg_ptr + start + g_off, mask=g_mask, other=0)
            mx = tl.load(mx_ptr + g_idx, mask=g_mask, other=0.0)
            my = tl.load(my_ptr + g_idx, mask=g_mask, other=0.0)
            c = tl.load(cos_ptr + g_idx, mask=g_mask, other=0.0)
            s = tl.load(sin_ptr + g_idx, mask=g_mask, other=0.0)
            isa = tl.load(isa_ptr + g_idx, mask=g_mask, other=0.0)
            isc = tl.load(isc_ptr + g_idx, mask=g_mask, other=0.0)
            op = tl.load(op_ptr + g_idx, mask=g_mask, other=0.0)

            dx = px[None, :] - mx[:, None]
            dy = py[None, :] - my[:, None]
            da = c[:, None] * dx + s[:, None] * dy
            dc = -s[:, None] * dx + c[:, None] * dy
            m = da * da * isa[:, None] + dc * dc * isc[:, None]
            alpha = tl.exp(-0.5 * tl.minimum(m, _MAHAL))
            # Hard cutoff (strict <): see the gray-wash note in splat_render.py.
            alpha = tl.where(m < _MAHAL, alpha, 0.0) * op[:, None]
            alpha = tl.where(g_mask[:, None], alpha, 0.0)
            alpha = tl.minimum(alpha, 1.0 - EPS_C)
            acc += tl.sum(libdevice.log1p(-alpha), axis=0)

        flat = (b * Hp + ty * TILE + py_l) * Wp + tx * TILE + px_l
        tl.store(accum_ptr + flat, acc)
```

Also append a direct-call parity test to `tests/test_splat_triton.py` so the kernel is validated inside this task (against the PyTorch tiled path's public output, before the kwarg exists):

```python
@cuda_only
def test_fwd_kernel_accum_matches_tiles_path():
    from pydiffvg import splat_render as sr
    from pydiffvg import splat_triton as st

    if not st.triton_available():
        pytest.skip("triton unavailable")

    torch.manual_seed(3)
    B, G = 2, 64
    means = torch.rand(B * G, 2, device="cuda") * 384
    ang = torch.rand(B * G, device="cuda") * 6.28
    cos_, sin_ = torch.cos(ang), torch.sin(ang)
    isa = 1.0 / (torch.rand(B * G, device="cuda") * 9 + 0.01)
    isc = 1.0 / (torch.rand(B * G, device="cuda") * 4 + 0.01)
    op = torch.rand(B * G, device="cuda")

    T, canvas = 16, 384
    n_t = canvas // T
    pg, ptx, pty = sr._build_tile_pairs(means, cos_, sin_, isa, isc,
                                        (0, 0, canvas, canvas), T, n_t, n_t)
    segs = st.build_tile_segments(pg, ptx, pty, G, n_t, n_t)
    accum = torch.zeros(B * canvas * canvas, device="cuda")
    st._fwd_kernel[(segs.seg_start.shape[0],)](
        segs.pg, segs.seg_start, segs.seg_count, segs.seg_b, segs.seg_ty, segs.seg_tx,
        means[:, 0].contiguous(), means[:, 1].contiguous(), cos_, sin_, isa, isc, op,
        accum, 0.0, 0.0, canvas, canvas,
        TILE=T, BLOCK_G=16, EPS_C=torch.finfo(torch.float32).eps,
    )
    tri = 1.0 - (1.0 - torch.exp(accum)).clamp(0.0, 1.0).reshape(B, canvas, canvas)

    ref = sr._splat_tiled(means.reshape(B, G, 2), cos_.reshape(B, G),
                          sin_.reshape(B, G), isa.reshape(B, G), isc.reshape(B, G),
                          op.reshape(B, G), (0, 0, canvas, canvas), T, True)
    ref = 1.0 - ref.clamp(0.0, 1.0)
    assert (tri - ref).abs().max().item() <= 1e-5
```

- [ ] **Step 4: Run the direct test to verify it passes**

Run: `CPATH=/home/naka/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 uv run pytest tests/test_splat_triton.py -v -k accum_matches`
Expected: PASS (first run pays ~5–20 s of Triton JIT). If it fails on gcc/`Python.h`, re-read "Environment prerequisites" above.

- [ ] **Step 5: Commit**

```bash
git add pydiffvg/splat_triton.py tests/test_splat_triton.py
git commit -m "feat(splat): Triton forward tile kernel with register accumulator"
```

---

### Task 3: Backward kernel + autograd.Function

**Files:**
- Modify: `pydiffvg/splat_triton.py` (append)
- Test: `tests/test_splat_triton.py` (append)

**Interfaces:**
- Consumes: `_fwd_kernel`, `TileSegments`.
- Produces: `triton_splat_accum(mx, my, cos, sin, isa, isc, op, segs, x0, y0, Hp, Wp, B, tile, block_g=16) -> accum (B*Hp*Wp,)` — differentiable w.r.t. the seven parameter tensors. Task 4 calls exactly this.

- [ ] **Step 1: Write the failing test** (append)

```python
@cuda_only
def test_triton_accum_gradients_match_pytorch_tiled():
    from pydiffvg import splat_render as sr
    from pydiffvg import splat_triton as st

    if not st.triton_available():
        pytest.skip("triton unavailable")

    torch.manual_seed(4)
    B, G, T, canvas = 1, 128, 16, 128
    n_t = canvas // T

    def make_params():
        means = (torch.rand(B * G, 2, device="cuda") * canvas).requires_grad_(True)
        ang = torch.rand(B * G, device="cuda") * 6.28
        cos_ = torch.cos(ang).detach().requires_grad_(True)
        sin_ = torch.sin(ang).detach().requires_grad_(True)
        isa = (1.0 / (torch.rand(B * G, device="cuda") * 9 + 0.01)).requires_grad_(True)
        isc = (1.0 / (torch.rand(B * G, device="cuda") * 4 + 0.01)).requires_grad_(True)
        op = torch.rand(B * G, device="cuda").requires_grad_(True)
        return means, cos_, sin_, isa, isc, op

    means, cos_, sin_, isa, isc, op = make_params()
    with torch.no_grad():
        pg, ptx, pty = sr._build_tile_pairs(means, cos_, sin_, isa, isc,
                                            (0, 0, canvas, canvas), T, n_t, n_t)
    segs = st.build_tile_segments(pg, ptx, pty, G, n_t, n_t)

    accum = st.triton_splat_accum(
        means[:, 0].contiguous(), means[:, 1].contiguous(), cos_, sin_, isa, isc, op,
        segs, 0.0, 0.0, canvas, canvas, B, T,
    )
    out_tri = (1.0 - torch.exp(accum)).reshape(B, canvas, canvas)

    out_ref = sr._splat_tiled(means.reshape(B, G, 2), cos_.reshape(B, G),
                              sin_.reshape(B, G), isa.reshape(B, G), isc.reshape(B, G),
                              op.reshape(B, G), (0, 0, canvas, canvas), T, True)

    torch.manual_seed(5)
    tgt = torch.rand_like(out_ref)
    inputs = [means, cos_, sin_, isa, isc, op]
    g_ref = torch.autograd.grad(((out_ref - tgt) ** 2).mean(), inputs, retain_graph=True)
    g_tri = torch.autograd.grad(((out_tri - tgt) ** 2).mean(), inputs)
    for a, b in zip(g_ref, g_tri):
        assert (a - b).abs().max().item() <= 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CPATH=... uv run pytest tests/test_splat_triton.py -v -k gradients_match`
Expected: FAIL with `AttributeError: ... has no attribute 'triton_splat_accum'`

- [ ] **Step 3: Write the backward kernel + Function** (append; spike-validated verbatim)

```python
if _TRITON_IMPORTED:

    @triton.jit
    def _bwd_kernel(
        pg_ptr, seg_start_ptr, seg_count_ptr, seg_b_ptr, seg_ty_ptr, seg_tx_ptr,
        mx_ptr, my_ptr, cos_ptr, sin_ptr, isa_ptr, isc_ptr, op_ptr,
        gacc_ptr,
        gmx_ptr, gmy_ptr, gcos_ptr, gsin_ptr, gisa_ptr, gisc_ptr, gop_ptr,
        x0, y0, Hp, Wp,
        TILE: tl.constexpr, BLOCK_G: tl.constexpr, EPS_C: tl.constexpr,
    ):
        seg = tl.program_id(0)
        start = tl.load(seg_start_ptr + seg)
        count = tl.load(seg_count_ptr + seg)
        b = tl.load(seg_b_ptr + seg)
        ty = tl.load(seg_ty_ptr + seg)
        tx = tl.load(seg_tx_ptr + seg)

        offs = tl.arange(0, TILE * TILE)
        py_l = offs // TILE
        px_l = offs % TILE
        px = x0 + (tx * TILE + px_l).to(tl.float32) + 0.5
        py = y0 + (ty * TILE + py_l).to(tl.float32) + 0.5
        flat = (b * Hp + ty * TILE + py_l) * Wp + tx * TILE + px_l
        g_up = tl.load(gacc_ptr + flat)  # dL/d accum per pixel

        for i in range(0, count, BLOCK_G):
            g_off = i + tl.arange(0, BLOCK_G)
            g_mask = g_off < count
            g_idx = tl.load(pg_ptr + start + g_off, mask=g_mask, other=0)
            mx = tl.load(mx_ptr + g_idx, mask=g_mask, other=0.0)
            my = tl.load(my_ptr + g_idx, mask=g_mask, other=0.0)
            c = tl.load(cos_ptr + g_idx, mask=g_mask, other=0.0)
            s = tl.load(sin_ptr + g_idx, mask=g_mask, other=0.0)
            isa = tl.load(isa_ptr + g_idx, mask=g_mask, other=0.0)
            isc = tl.load(isc_ptr + g_idx, mask=g_mask, other=0.0)
            op = tl.load(op_ptr + g_idx, mask=g_mask, other=0.0)

            dx = px[None, :] - mx[:, None]
            dy = py[None, :] - my[:, None]
            da = c[:, None] * dx + s[:, None] * dy
            dc = -s[:, None] * dx + c[:, None] * dy
            m = da * da * isa[:, None] + dc * dc * isc[:, None]
            in_cut = m < _MAHAL
            alpha_pre = tl.where(in_cut, tl.exp(-0.5 * tl.minimum(m, _MAHAL)), 0.0)
            alpha = alpha_pre * op[:, None]
            alpha = tl.where(g_mask[:, None], alpha, 0.0)
            alpha_cl = tl.minimum(alpha, 1.0 - EPS_C)

            # d/d alpha of log1p(-alpha_cl): -1/(1-alpha_cl), gated like
            # torch.clamp's backward (grad passes where alpha <= max).
            d_alpha = -g_up[None, :] / (1.0 - alpha_cl)
            d_alpha = tl.where(alpha <= 1.0 - EPS_C, d_alpha, 0.0)
            d_alpha = tl.where(g_mask[:, None], d_alpha, 0.0)

            d_op_px = d_alpha * alpha_pre           # alpha = alpha_pre * op
            d_pre = d_alpha * op[:, None]
            d_m = tl.where(in_cut, d_pre * (-0.5) * alpha_pre, 0.0)
            d_da = d_m * 2.0 * da * isa[:, None]    # m = da^2 isa + dc^2 isc
            d_dc = d_m * 2.0 * dc * isc[:, None]
            d_isa_px = d_m * da * da
            d_isc_px = d_m * dc * dc
            d_mx_px = -(d_da * c[:, None]) + d_dc * s[:, None]  # dx = px - mx
            d_my_px = -(d_da * s[:, None]) - d_dc * c[:, None]
            d_cos_px = d_da * dx + d_dc * dy
            d_sin_px = d_da * dy - d_dc * dx

            tl.atomic_add(gmx_ptr + g_idx, tl.sum(d_mx_px, axis=1), mask=g_mask)
            tl.atomic_add(gmy_ptr + g_idx, tl.sum(d_my_px, axis=1), mask=g_mask)
            tl.atomic_add(gcos_ptr + g_idx, tl.sum(d_cos_px, axis=1), mask=g_mask)
            tl.atomic_add(gsin_ptr + g_idx, tl.sum(d_sin_px, axis=1), mask=g_mask)
            tl.atomic_add(gisa_ptr + g_idx, tl.sum(d_isa_px, axis=1), mask=g_mask)
            tl.atomic_add(gisc_ptr + g_idx, tl.sum(d_isc_px, axis=1), mask=g_mask)
            tl.atomic_add(gop_ptr + g_idx, tl.sum(d_op_px, axis=1), mask=g_mask)


class _TritonSplatAccum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mx, my, cos, sin, isa, isc, op,
                pg, seg_start, seg_count, seg_b, seg_ty, seg_tx,
                x0, y0, Hp, Wp, B, tile, block_g):
        accum = torch.zeros(B * Hp * Wp, device=mx.device, dtype=torch.float32)
        n_seg = seg_start.shape[0]
        eps = torch.finfo(torch.float32).eps
        if n_seg > 0:
            _fwd_kernel[(n_seg,)](
                pg, seg_start, seg_count, seg_b, seg_ty, seg_tx,
                mx, my, cos, sin, isa, isc, op, accum,
                float(x0), float(y0), Hp, Wp,
                TILE=tile, BLOCK_G=block_g, EPS_C=eps,
            )
        ctx.save_for_backward(mx, my, cos, sin, isa, isc, op,
                              pg, seg_start, seg_count, seg_b, seg_ty, seg_tx)
        ctx.meta = (x0, y0, Hp, Wp, B, tile, block_g)
        return accum

    @staticmethod
    def backward(ctx, grad_accum):
        (mx, my, cos, sin, isa, isc, op,
         pg, seg_start, seg_count, seg_b, seg_ty, seg_tx) = ctx.saved_tensors
        x0, y0, Hp, Wp, B, tile, block_g = ctx.meta
        grads = [torch.zeros_like(t) for t in (mx, my, cos, sin, isa, isc, op)]
        n_seg = seg_start.shape[0]
        eps = torch.finfo(torch.float32).eps
        if n_seg > 0:
            _bwd_kernel[(n_seg,)](
                pg, seg_start, seg_count, seg_b, seg_ty, seg_tx,
                mx, my, cos, sin, isa, isc, op,
                grad_accum.contiguous(), *grads,
                float(x0), float(y0), Hp, Wp,
                TILE=tile, BLOCK_G=block_g, EPS_C=eps,
            )
        # 7 parameter grads + 13 Nones (6 index tensors + 7 scalars).
        return (*grads, None, None, None, None, None, None,
                None, None, None, None, None, None, None)


def triton_splat_accum(mx, my, cos, sin, isa, isc, op, segs,
                       x0, y0, Hp, Wp, B, tile, block_g=16):
    return _TritonSplatAccum.apply(
        mx, my, cos, sin, isa, isc, op,
        segs.pg, segs.seg_start, segs.seg_count, segs.seg_b, segs.seg_ty, segs.seg_tx,
        x0, y0, Hp, Wp, B, tile, block_g,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CPATH=... uv run pytest tests/test_splat_triton.py -v -k gradients_match`
Expected: PASS (grad diffs measured ~1e-8 in the spike; the 1e-5 gate has huge margin)

- [ ] **Step 5: Commit**

```bash
git add pydiffvg/splat_triton.py tests/test_splat_triton.py
git commit -m "feat(splat): Triton backward kernel + autograd wrapper"
```

---

### Task 4: Wire `tiling="triton"` into `splat_render_cubics`

**Files:**
- Modify: `pydiffvg/splat_render.py` — the `tiling` validation (search for `'tiling must be'`) and the `if tiling == "tiles":` branch (search for `_splat_tiled(`); the docstring `tiling:` entry.
- Test: `tests/test_splat_triton.py` (append)

**Interfaces:**
- Consumes: `build_tile_segments`, `triton_splat_accum`, `triton_available` from `pydiffvg.splat_triton`; `_build_tile_pairs` and the flattened params already present in `splat_render_cubics`.
- Produces: `splat_render_cubics(..., tiling="triton", tile_size=16)` — same contract as `tiling="tiles"` (shape, pixel_box composition, zero-pair grads). `use_checkpoint`/`use_compile` are ignored on this path (documented).

- [ ] **Step 1: Write the failing tests** (append)

```python
@cuda_only
@pytest.mark.parametrize("tile", [16, 32])
def test_triton_path_end_to_end(tile):
    from pydiffvg.splat_render import splat_render_cubics

    c, w, o = _scene(20, seed=7)
    kw = dict(canvas_size=384, num_samples=16, opacities=o)
    dense = splat_render_cubics(c, w, **kw)
    tri = splat_render_cubics(c, w, **kw, tiling="triton", tile_size=tile)
    assert (dense - tri).abs().max().item() <= 1e-5

    torch.manual_seed(8)
    tgt = torch.rand_like(dense)
    gd = torch.autograd.grad(((dense - tgt) ** 2).mean(), [c, w, o], retain_graph=True)
    gt = torch.autograd.grad(((tri - tgt) ** 2).mean(), [c, w, o])
    for a, b in zip(gd, gt):
        assert (a - b).abs().max().item() <= 1e-5


@cuda_only
def test_triton_path_pixel_box_and_zero_pairs():
    from pydiffvg.splat_render import splat_render_cubics

    c, w, o = _scene(6, seed=9)
    box = (17, 9, 24, 20)
    dense = splat_render_cubics(c, w, canvas_size=64, num_samples=16, opacities=o,
                                pixel_box=box)
    tri = splat_render_cubics(c, w, canvas_size=64, num_samples=16, opacities=o,
                              pixel_box=box, tiling="triton", tile_size=16)
    assert tri.shape == (1, 24, 20)
    assert (dense - tri).abs().max().item() <= 1e-5

    # Fully off-canvas: white output, exact-zero grads (graph stub).
    torch.manual_seed(10)
    c_off = (torch.rand(1, 4, 4, 2, device="cuda") + 5.0).requires_grad_(True)
    w_off = (torch.rand(1, 4, device="cuda") + 0.5).requires_grad_(True)
    out = splat_render_cubics(c_off, w_off, canvas_size=64, num_samples=16,
                              tiling="triton")
    assert (out == 1.0).all()
    g = torch.autograd.grad(out.sum(), [c_off, w_off])
    assert all((x == 0).all() for x in g)


def test_triton_path_rejects_cpu_and_bad_dtype():
    from pydiffvg.splat_render import splat_render_cubics

    torch.manual_seed(0)
    c = torch.rand(1, 2, 4, 2) * 2 - 1
    w = torch.rand(1, 2) + 0.5
    with pytest.raises(RuntimeError, match="triton"):
        splat_render_cubics(c, w, canvas_size=32, tiling="triton")
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="float32"):
            splat_render_cubics(c.cuda().double(), w.cuda().double(),
                                canvas_size=32, tiling="triton")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CPATH=... uv run pytest tests/test_splat_triton.py -v -k "triton_path"`
Expected: FAIL with `ValueError: tiling must be "none", "tiles" or "auto", got 'triton'`

- [ ] **Step 3: Implement the wiring** (in `pydiffvg/splat_render.py`)

Change the validation line to:

```python
    if tiling not in ("none", "tiles", "auto", "triton"):
        raise ValueError(
            f'tiling must be "none", "tiles", "auto" or "triton", got {tiling!r}'
        )
```

(`tiling="auto"` continues to resolve to `"tiles"` — flipping auto to triton is a
separate decision after downstream soak; do NOT change it in this plan.)

Immediately after the existing `if tiling == "tiles":` block's `return`, add:

```python
    if tiling == "triton":
        from pydiffvg import splat_triton

        if cubics.device.type != "cuda" or not splat_triton.triton_available():
            raise RuntimeError(
                'tiling="triton" requires CUDA with triton importable; '
                'use tiling="tiles" elsewhere'
            )
        if dtype != torch.float32:
            raise RuntimeError(f'tiling="triton" supports float32 only, got {dtype}')
        region = pixel_box if pixel_box is not None else (0, 0, H, W)
        ry0, rx0, rh, rw = region
        T = tile_size
        n_tx, n_ty = (rw + T - 1) // T, (rh + T - 1) // T
        Hp, Wp = n_ty * T, n_tx * T
        means_bg = means_flat.reshape(B * num_strokes * K, 2)
        cos_bg = cos_flat.reshape(-1)
        sin_bg = sin_flat.reshape(-1)
        inv_sa2_bg = inv_sa2_flat.reshape(-1)
        inv_sc2_bg = inv_sc2_flat.reshape(-1)
        opacity_bg = opacity_flat.reshape(-1)
        pair_gauss, ptx, pty = _build_tile_pairs(
            means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg, region, T, n_tx, n_ty
        )
        segs = splat_triton.build_tile_segments(
            pair_gauss, ptx, pty, num_strokes * K, n_tx, n_ty
        )
        accum = splat_triton.triton_splat_accum(
            means_bg[:, 0].contiguous(), means_bg[:, 1].contiguous(),
            cos_bg.contiguous(), sin_bg.contiguous(),
            inv_sa2_bg.contiguous(), inv_sc2_bg.contiguous(),
            opacity_bg.contiguous(),
            segs, rx0, ry0, Hp, Wp, B, T,
        )
        if pair_gauss.numel() == 0 and torch.is_grad_enabled():
            connect = (
                means_bg.sum() + cos_bg.sum() + sin_bg.sum()
                + inv_sa2_bg.sum() + inv_sc2_bg.sum() + opacity_bg.sum()
            )
            accum = accum + connect * 0.0
        output = (1.0 - torch.exp(accum)).reshape(B, Hp, Wp)[:, :rh, :rw]
        return 1.0 - output.clamp(0.0, 1.0)
```

Docstring: extend the `tiling:` entry with
`"triton" runs the tile-culled evaluation as Triton kernels (CUDA + float32 only; ~17x over "tiles" at 10k gaussians). use_checkpoint and use_compile are ignored on this path.`

- [ ] **Step 4: Run the new tests, the whole Triton file, and the FULL suite**

Run: `CPATH=... uv run pytest tests/test_splat_triton.py -v` — Expected: all PASS
Run: `CPATH=... uv run pytest -q` — Expected: **everything passes** (the pre-existing 176 + new; 2 pre-existing CPU skips are normal)

- [ ] **Step 5: Run the existing tiled exactness suite against the Triton path**

The adversarial cases in `tests/test_splat_render_tiling.py` (border-straddling, near-zero sigma, opacity 0, mixed off-canvas, B=3) gate the tiles path; run them manually swapped to triton with a one-off:

```bash
CPATH=... uv run python - <<'EOF'
# Adversarial sweep: run every scenario from the tiling suite with tiling="triton".
import torch
from pydiffvg.splat_render import splat_render_cubics

torch.manual_seed(0)
dev = "cuda"
cases = []
# border-straddling + random strokes, canvas 64 T=16
c = torch.zeros(1, 22, 4, 2, device=dev)
c[0, :20] = torch.rand(20, 4, 2, device=dev) * 2 - 1
c[0, 20, :, 1] = 16.0 / 32.0 - 1.0   # horizontal stroke at pixel y=16.0
c[0, 20, :, 0] = torch.linspace(-0.9, 0.9, 4, device=dev)
c[0, 21, :, 1] = 15.5 / 32.0 - 1.0
c[0, 21, :, 0] = torch.linspace(-0.9, 0.9, 4, device=dev)
cases.append(("border", c, torch.rand(1, 22, device=dev) * 2 + 0.5,
              torch.rand(1, 22, device=dev), 64))
# near-zero widths
cases.append(("tiny-sigma", torch.rand(1, 8, 4, 2, device=dev) * 2 - 1,
              torch.full((1, 8), 1e-6, device=dev), torch.rand(1, 8, device=dev), 64))
# half zero opacity
o = torch.rand(1, 20, device=dev); o[:, :10] = 0.0
cases.append(("zero-op", torch.rand(1, 20, 4, 2, device=dev) * 2 - 1,
              torch.rand(1, 20, device=dev) * 2 + 0.5, o, 128))
worst_f = worst_g = 0.0
for name, c, w, o, canvas in cases:
    c = c.clone().requires_grad_(True); w = w.clone().requires_grad_(True)
    o = o.clone().requires_grad_(True)
    d = splat_render_cubics(c, w, canvas_size=canvas, num_samples=16, opacities=o)
    t = splat_render_cubics(c, w, canvas_size=canvas, num_samples=16, opacities=o,
                            tiling="triton", tile_size=16)
    f = (d - t).abs().max().item()
    tgt = torch.rand_like(d)
    gd = torch.autograd.grad(((d - tgt) ** 2).mean(), [c, w, o], retain_graph=True)
    gt = torch.autograd.grad(((t - tgt) ** 2).mean(), [c, w, o])
    g = max((a - b).abs().max().item() for a, b in zip(gd, gt))
    worst_f, worst_g = max(worst_f, f), max(worst_g, g)
    print(f"{name}: fwd {f:.3g} grad {g:.3g}")
assert worst_f <= 1e-5 and worst_g <= 1e-5, "ADVERSARIAL GATE FAILED"
print("adversarial gate OK")
EOF
```

Expected: `adversarial gate OK`. Record the printed diffs for the PR.

- [ ] **Step 6: Commit**

```bash
git add pydiffvg/splat_render.py tests/test_splat_triton.py
git commit -m "feat(splat): wire tiling=\"triton\" into splat_render_cubics"
```

---

### Task 5: Benchmark arm + acceptance run

**Files:**
- Modify: `benchmarks/bench_splat_tiling.py` — add a `tiled-triton` arm alongside the existing `tiled ts=16/32` arms (same `timed_run` harness; call `splat_render_cubics(..., tiling="triton", tile_size=16)`). Follow the file's existing arm pattern exactly (exactness gate before timing, OOM guard, peak-memory row).
- Modify: `README.md` — performance-knobs table: add a `tiling="triton"` row with the measured numbers.

**Interfaces:**
- Consumes: the Task 4 kwarg.
- Produces: benchmark output for the PR; no code interfaces.

- [ ] **Step 1: Add the arm** — copy the existing tiled-arm block in `benchmarks/bench_splat_tiling.py`, switch the kwargs to `tiling="triton", tile_size=16`, label it `triton`. Guard: skip the arm with a printed note when `not pydiffvg.splat_triton.triton_available()`.
- [ ] **Step 2: Validate quick mode**: `CPATH=... uv run python benchmarks/bench_splat_tiling.py --quick --device cuda --skip-compile-arm` — Expected: table includes a `triton` column, exactness checks print OK.
- [ ] **Step 3: Confirm the GPU is idle** (`nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l` → 0), then full run: `CPATH=... uv run python benchmarks/bench_splat_tiling.py --device cuda`.
   Acceptance: triton ≥5× over tiled ts=16 at G=10,240 canvas 384 (spike reference: 47.4 → 2.8 ms/iter, 16.9×; peak memory 33 vs 727 MB). If below 5×, STOP — something regressed vs the spike; compare your kernel against the code blocks in Tasks 2–3 before proceeding.
- [ ] **Step 4: Update README** row with the measured numbers and commit:

```bash
git add benchmarks/bench_splat_tiling.py README.md
git commit -m "bench(splat): triton arm in the tiling sweep + README numbers"
```

---

### Task 6: Byte-identity guard, review, PR

**Files:**
- No new files; runs verification and opens the PR.

- [ ] **Step 1: Default-path byte-identity vs origin/main.** Save this as `/tmp/check_default_identity.py` and run `uv run python /tmp/check_default_identity.py`:

```python
import importlib.util, subprocess, sys, tempfile, os
import torch

src = subprocess.run(["git", "show", "origin/main:pydiffvg/splat_render.py"],
                     capture_output=True, text=True, check=True).stdout
fd, path = tempfile.mkstemp(suffix="_main.py")
with os.fdopen(fd, "w") as f:
    f.write(src)
spec = importlib.util.spec_from_file_location("splat_main", path)
old = importlib.util.module_from_spec(spec)
sys.modules["splat_main"] = old
spec.loader.exec_module(old)
from pydiffvg import splat_render as new

ok = True
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    for seed, S, canvas in [(0, 1, 224), (1, 6, 64), (2, 40, 100)]:
        g = torch.Generator().manual_seed(seed)
        c = (torch.rand(1, S, 4, 2, generator=g) * 2 - 1).to(dev).requires_grad_(True)
        w = (torch.rand(1, S, generator=g) * 3 + 0.5).to(dev).requires_grad_(True)
        o = torch.rand(1, S, generator=g).to(dev).requires_grad_(True)
        a = old.splat_render_cubics(c, w, canvas_size=canvas, num_samples=16, opacities=o)
        b = new.splat_render_cubics(c, w, canvas_size=canvas, num_samples=16, opacities=o)
        ga = torch.autograd.grad(a.sum(), [c, w, o], retain_graph=True)
        gb = torch.autograd.grad(b.sum(), [c, w, o])
        same = torch.equal(a, b) and all(torch.equal(x, y) for x, y in zip(ga, gb))
        ok &= same
        print(f"[{dev}] S={S} canvas={canvas}: bitwise={same}")
print("BYTE-IDENTICAL" if ok else "MISMATCH — DO NOT SHIP")
sys.exit(0 if ok else 1)
```

Expected output ends with `BYTE-IDENTICAL`.

- [ ] **Step 2: Full suite, both env flavors**: `uv run pytest -q` and `CPATH=... uv run pytest -q` — all pass.
- [ ] **Step 3: Dispatch a code-review subagent** over `git diff origin/main...HEAD` with the Global Constraints section as its review charter; fix findings; re-run Step 2.
- [ ] **Step 4: Push and open the PR** (do not merge). PR body must include: the benchmark table (with GPU idle/contended state), worst-case adversarial diffs from Task 4 Step 5, the byte-identity output, and a determinism note (Triton forward is deterministic per tile — an improvement over the scatter-add path; backward atomics remain ~1-ulp nondeterministic).

```bash
git push -u origin HEAD
gh pr create --base main --title "perf(splat): Triton tile kernels (tiling=\"triton\")" --body "..."
```

---

## Self-review notes (already applied)

- Spec coverage: segments (T1), forward (T2), backward (T3), integration + adversarial gates + pixel_box/zero-pair/dtype guards (T4), benchmark + acceptance (T5), byte-identity + review + PR (T6). `tiling="auto"` intentionally unchanged (stated in T4).
- The kernel/Function code in T2–T3 is spike-validated, not speculative; the direct-call test in T2 exists so the kernel is proven before the kwarg plumbing lands.
- Known open question deliberately deferred: making `tiling="auto"` select triton, and tuning `BLOCK_G`/`num_warps` (T16/BLOCK_G=16 already clears the bar; tuning is optional follow-up, time-boxed to one session if attempted).
