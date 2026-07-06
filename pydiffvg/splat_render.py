"""Pure PyTorch Bezier splatting renderer.

Drop-in replacement for RenderFunction that uses Gaussian splatting
instead of diffvg's CUDA kernels. Works on all GPU architectures.

Based on "Bezier Splatting for Fast and Differentiable Vector Graphics Rendering"
(arXiv:2503.16424).
"""

import collections
import warnings

import torch
from torch.utils.checkpoint import checkpoint

# LRU cache of flattened pixel grids keyed by (canvas_size, pixel_box, device,
# dtype). Capped because callers that slide a pixel_box window across the
# canvas would otherwise accumulate one grid per window position.
_PIXEL_GRID_CACHE: collections.OrderedDict = collections.OrderedDict()
_PIXEL_GRID_CACHE_MAX_ENTRIES = 64


def _get_pixel_grid(
    canvas_size: int,
    pixel_box: tuple[int, int, int, int] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return flattened pixel-center coordinates, (num_pixels, 2) as (x, y).

    pixel_box=(y0, x0, h, w) restricts the grid to that window; None covers
    the full canvas. Grids are cached per (canvas_size, pixel_box, device,
    dtype) with LRU eviction.
    """
    key = (canvas_size, pixel_box, device, dtype)
    pixels = _PIXEL_GRID_CACHE.get(key)
    if pixels is None:
        y0, x0, h, w = pixel_box if pixel_box is not None else (0, 0, canvas_size, canvas_size)
        py = torch.arange(y0, y0 + h, device=device, dtype=dtype) + 0.5
        px = torch.arange(x0, x0 + w, device=device, dtype=dtype) + 0.5
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
        pixels = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        _PIXEL_GRID_CACHE[key] = pixels
        if len(_PIXEL_GRID_CACHE) > _PIXEL_GRID_CACHE_MAX_ENTRIES:
            _PIXEL_GRID_CACHE.popitem(last=False)
    else:
        _PIXEL_GRID_CACHE.move_to_end(key)
    return pixels


def _evaluate_bezier(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curves at parameter values t.

    Uses Bernstein basis:
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Args:
        control_points: (B, num_strokes, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, num_strokes, K, 2) positions on curves.
    """
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t

    b0 = omt * omt * omt
    b1 = 3.0 * omt * omt * t
    b2 = 3.0 * omt * t * t
    b3 = t * t * t

    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _evaluate_bezier_tangent(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate tangent (derivative) of cubic Bezier curves at parameter values t.

    B'(t) = 3[(1-t)^2 (P1-P0) + 2(1-t)t (P2-P1) + t^2 (P3-P2)]

    Args:
        control_points: (B, num_strokes, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, num_strokes, K, 2) tangent vectors.
    """
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t

    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    d0 = omt * omt
    d1 = 2.0 * omt * t
    d2 = t * t

    return 3.0 * (d0 * (p1 - p0) + d1 * (p2 - p1) + d2 * (p3 - p2))


# torch.compile support for the splat kernel. Whether compilation works is
# established once per device type by compiling the REAL kernel
# (forward + backward) in a subprocess. Two reasons it must be the real
# kernel and must be out-of-process:
# - a trivial probe can pass off inductor's on-disk cache while the actual
#   kernel build still fails (observed with missing Python.h);
# - a failed in-process compile can leave dynamo/inductor state broken so
#   that subsequent grad-mode EAGER calls raise InductorError, and
#   torch._dynamo.reset() does not recover (observed empirically). A broken
#   probe subprocess costs seconds; a poisoned training process costs the run.
_COMPILE_OK: dict = {}  # device type -> bool
_COMPILED_SPLAT_CHUNK = None

_COMPILE_PREFLIGHT_CODE = """
import torch
from pydiffvg.splat_render import _splat_chunk

device = "{device_type}"
g, p = 3, 5
means = torch.rand(1, g, 2, device=device, requires_grad=True)
angles = torch.rand(1, g, device=device)
out = torch.compile(_splat_chunk)(
    torch.rand(p, 2, device=device),
    means,
    torch.cos(angles),
    torch.sin(angles),
    torch.rand(1, g, device=device) + 0.5,
    torch.rand(1, g, device=device) + 0.5,
    torch.rand(1, g, device=device),
)
out.sum().backward()
assert means.grad is not None
"""


def _run_compile_preflight(device_type: str) -> tuple[bool, str]:
    """Compile the splat kernel in a subprocess; return (ok, failure detail)."""
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _COMPILE_PREFLIGHT_CODE.format(device_type=device_type)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - any launch failure means "unavailable"
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, ""
    tail = proc.stderr.strip().splitlines()
    return False, tail[-1] if tail else f"exit code {proc.returncode}"


def _compile_available(device: torch.device) -> bool:
    ok = _COMPILE_OK.get(device.type)
    if ok is None:
        ok, detail = _run_compile_preflight(device.type)
        if not ok:
            warnings.warn(
                f"torch.compile of the splat kernel failed in a preflight "
                f"subprocess on {device.type} ({detail}); use_compile=True "
                "falls back to eager.",
                RuntimeWarning,
                stacklevel=3,
            )
        _COMPILE_OK[device.type] = ok
    return ok


def _get_splat_chunk_fn(use_compile: bool, device: torch.device):
    global _COMPILED_SPLAT_CHUNK
    if not (use_compile and _compile_available(device)):
        return _splat_chunk
    if _COMPILED_SPLAT_CHUNK is None:
        if torch.cuda.is_available():
            # Inductor may lazily initialize CUDA during the first compiled
            # call (even when targeting CPU); if that first call happens
            # inside torch.utils.checkpoint's forward, checkpoint raises
            # "device state was initialized in the forward pass". Initialize
            # device state up front instead.
            torch.cuda.init()
        _COMPILED_SPLAT_CHUNK = torch.compile(_splat_chunk)
    return _COMPILED_SPLAT_CHUNK


def _splat_chunk(chunk_pixels, means_f, cos_f, sin_f, inv_sa2_f, inv_sc2_f, opacity_f):
    """Compute combined alpha for a chunk of pixels using over compositing."""
    dx = chunk_pixels[None, None, :, 0] - means_f[:, :, None, 0]
    dy = chunk_pixels[None, None, :, 1] - means_f[:, :, None, 1]
    d_along = cos_f[:, :, None] * dx + sin_f[:, :, None] * dy
    d_across = -sin_f[:, :, None] * dx + cos_f[:, :, None] * dy
    mahal_sq = d_along.square() * inv_sa2_f[:, :, None] + d_across.square() * inv_sc2_f[:, :, None]
    alpha = torch.exp(-0.5 * mahal_sq.clamp(max=20.0))
    # Hard cutoff beyond the clamp radius (~4.5 sigma): clamping alone
    # leaves an exp(-10) alpha floor on EVERY pixel for EVERY gaussian,
    # which composites into a visible gray background wash at scale
    # (e.g. 40k gaussians -> (1 - exp(-10))^40k ~ 0.2 background).
    alpha = alpha * (mahal_sq < 20.0)
    alpha = alpha * opacity_f[:, :, None]  # Apply per-stroke opacity
    # Alpha compositing: combined = 1 - prod(1 - alpha_i)
    # This gives natural overlap behavior instead of saturation
    transmittance = (1.0 - alpha.clamp(0.0, 1.0)).prod(dim=1)
    return 1.0 - transmittance  # (B, num_pixels)


# --------------------------------------------------------------------------
# Tile-culled splatting (tiling="tiles")
#
# The dense path evaluates every gaussian against every pixel even though the
# hard cutoff zeroes everything beyond mahal_sq >= 20 (~4.5 sigma). The tiled
# path splits the render region into fixed-size tiles, computes a conservative
# axis-aligned bounding box per gaussian from the same cutoff, and evaluates
# each gaussian only against the pixels of the tiles its box overlaps. Because
# the compositor 1 - prod(1 - alpha) is an order-independent product, the
# per-pixel combination can be accumulated as sum(log1p(-alpha)) with
# scatter-add (no sorting, no segmenting), then 1 - exp(.) at the end —
# mathematically identical, differentiable through gather/scatter.
# --------------------------------------------------------------------------

# Must match the cutoff literals inside _splat_chunk and _splat_tile_chunk.
_MAHAL_SQ_CUTOFF = 20.0

# tiling="auto" uses tiles at or above this per-image gaussian count
# (num_strokes * num_samples). benchmarks/bench_splat_tiling.py on an RTX 5090
# found NO crossover: tiled won every measured config (G=16..40960, full-frame
# canvas 384/768 at 8.1-21.9x, and a 96x96 pixel_box window at 1.4x), so auto
# always tiles. Kept as a constant so deployments can re-tune if a slower
# device shows a real crossover.
_TILING_AUTO_THRESHOLD_G = 0


def _splat_tile_chunk(
    pair_gauss, pair_tile_x, pair_tile_y, pair_b,
    means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg, opacity_bg,
    off_x_f, off_y_f, off_x_i, off_y_i,
    x0, y0, tile_size, Hp, Wp, batch,
):
    """Log-transmittance contribution of a chunk of (gaussian, tile) pairs.

    Evaluates each pair's gaussian against its tile's tile_size^2 pixels
    (same math as _splat_chunk, including the hard cutoff) and scatter-adds
    log1p(-alpha) into a flat (batch * Hp * Wp) buffer.
    """
    mx = means_bg[pair_gauss, 0].unsqueeze(1)
    my = means_bg[pair_gauss, 1].unsqueeze(1)
    cos_p = cos_bg[pair_gauss].unsqueeze(1)
    sin_p = sin_bg[pair_gauss].unsqueeze(1)
    inv_sa2_p = inv_sa2_bg[pair_gauss].unsqueeze(1)
    inv_sc2_p = inv_sc2_bg[pair_gauss].unsqueeze(1)
    opacity_p = opacity_bg[pair_gauss].unsqueeze(1)

    # Pixel centers of each pair's tile, in canvas coordinates.
    px = x0 + (pair_tile_x * tile_size).to(off_x_f.dtype).unsqueeze(1) + off_x_f + 0.5
    py = y0 + (pair_tile_y * tile_size).to(off_y_f.dtype).unsqueeze(1) + off_y_f + 0.5

    dx = px - mx
    dy = py - my
    d_along = cos_p * dx + sin_p * dy
    d_across = -sin_p * dx + cos_p * dy
    mahal_sq = d_along.square() * inv_sa2_p + d_across.square() * inv_sc2_p
    alpha = torch.exp(-0.5 * mahal_sq.clamp(max=_MAHAL_SQ_CUTOFF))
    alpha = alpha * (mahal_sq < _MAHAL_SQ_CUTOFF)  # same hard cutoff as _splat_chunk
    alpha = alpha * opacity_p

    # log1p(-alpha) is -inf at alpha == 1 (and its gradient diverges), so back
    # off by one fp epsilon; forward error is ~1e-7, within the tiled-vs-dense
    # tolerance.
    eps = torch.finfo(alpha.dtype).eps
    log_t = torch.log1p(-alpha.clamp(0.0, 1.0 - eps))  # (N, T^2), <= 0

    row = pair_tile_y.unsqueeze(1) * tile_size + off_y_i
    col = pair_tile_x.unsqueeze(1) * tile_size + off_x_i
    flat = (pair_b.unsqueeze(1) * Hp + row) * Wp + col

    buf = torch.zeros(
        batch * Hp * Wp, device=log_t.device, dtype=log_t.dtype
    )
    return buf.scatter_add(0, flat.reshape(-1), log_t.reshape(-1))


def _build_tile_pairs(means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg,
                      region, tile_size, n_tx, n_ty):
    """Conservative (gaussian, tile) pairs for the tiled path.

    Boxes each gaussian's cutoff support (|d_along| < sqrt(20/inv_sa2), same
    for across, rotated to axis-aligned extents) and rasterizes the box to
    tile ranges. Over-inclusion only costs speed; exactness comes from the
    kernel's own cutoff. Index-building only — runs under no_grad.
    """
    y0, x0, h, w = region
    with torch.no_grad():
        half_a = (_MAHAL_SQ_CUTOFF / inv_sa2_bg).sqrt()
        half_c = (_MAHAL_SQ_CUTOFF / inv_sc2_bg).sqrt()
        hx = cos_bg.abs() * half_a + sin_bg.abs() * half_c
        hy = sin_bg.abs() * half_a + cos_bg.abs() * half_c

        # Bounds in region-local pixel-index space (pixel j has center j+0.5).
        mx = means_bg[:, 0] - x0
        my = means_bg[:, 1] - y0
        x_lo, x_hi = mx - hx - 0.5, mx + hx - 0.5
        y_lo, y_hi = my - hy - 0.5, my + hy - 0.5

        keep = (x_hi >= 0) & (x_lo <= w - 1) & (y_hi >= 0) & (y_lo <= h - 1)
        keep &= torch.isfinite(x_lo) & torch.isfinite(y_lo)
        kept_idx = keep.nonzero(as_tuple=True)[0]
        if kept_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=means_bg.device)
            return empty, empty, empty

        ts = float(tile_size)
        tx0 = torch.floor(x_lo[kept_idx] / ts).clamp(0, n_tx - 1).long()
        tx1 = torch.floor(x_hi[kept_idx] / ts).clamp(0, n_tx - 1).long()
        ty0 = torch.floor(y_lo[kept_idx] / ts).clamp(0, n_ty - 1).long()
        ty1 = torch.floor(y_hi[kept_idx] / ts).clamp(0, n_ty - 1).long()

        ntx = tx1 - tx0 + 1
        nty = ty1 - ty0 + 1
        nt = ntx * nty
        total = int(nt.sum().item())

        ptr = torch.repeat_interleave(
            torch.arange(kept_idx.numel(), device=nt.device), nt
        )
        starts = torch.zeros_like(nt)
        starts[1:] = nt.cumsum(0)[:-1]
        rank = torch.arange(total, device=nt.device) - starts[ptr]
        pair_tile_x = tx0[ptr] + rank % ntx[ptr]
        pair_tile_y = ty0[ptr] + rank // ntx[ptr]
        pair_gauss = kept_idx[ptr]
        return pair_gauss, pair_tile_x, pair_tile_y


def _splat_tiled(means_flat, cos_flat, sin_flat, inv_sa2_flat, inv_sc2_flat,
                 opacity_flat, region, tile_size, use_checkpoint):
    """Tile-culled equivalent of the dense chunk loop.

    Returns the combined alpha image (B, h, w) for the given region
    (y0, x0, h, w) in canvas pixel coordinates.
    """
    B, G = opacity_flat.shape
    device = means_flat.device
    dtype = means_flat.dtype
    y0, x0, h, w = region
    T = tile_size
    n_tx = (w + T - 1) // T
    n_ty = (h + T - 1) // T
    Hp, Wp = n_ty * T, n_tx * T  # padded to whole tiles; cropped at the end

    means_bg = means_flat.reshape(B * G, 2)
    cos_bg = cos_flat.reshape(B * G)
    sin_bg = sin_flat.reshape(B * G)
    inv_sa2_bg = inv_sa2_flat.reshape(B * G)
    inv_sc2_bg = inv_sc2_flat.reshape(B * G)
    opacity_bg = opacity_flat.reshape(B * G)

    pair_gauss, pair_tile_x, pair_tile_y = _build_tile_pairs(
        means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg,
        region, T, n_tx, n_ty,
    )
    pair_b = pair_gauss // G

    # Within-tile pixel offsets, float for coordinates and long for indices.
    off = torch.arange(T, device=device)
    off_y_i, off_x_i = torch.meshgrid(off, off, indexing="ij")
    off_x_i = off_x_i.reshape(-1)
    off_y_i = off_y_i.reshape(-1)
    off_x_f = off_x_i.to(dtype)
    off_y_f = off_y_i.to(dtype)

    # Chunk over pairs to bound peak memory (each pair holds T^2 elements
    # across ~8 intermediates in fp32).
    bytes_per_element = 4 if dtype == torch.float32 else 2
    pair_chunk = max(256, (256 * 1024 * 1024) // (T * T * bytes_per_element * 8))

    accum = torch.zeros(B * Hp * Wp, device=device, dtype=dtype)
    n_pairs = pair_gauss.shape[0]
    if n_pairs == 0 and torch.is_grad_enabled():
        # Keep the output connected to the inputs so backward() yields zero
        # gradients (like the dense path) instead of "unused parameter".
        connect = (
            means_bg.sum() + cos_bg.sum() + sin_bg.sum()
            + inv_sa2_bg.sum() + inv_sc2_bg.sum() + opacity_bg.sum()
        )
        accum = accum + connect * 0.0
    for start in range(0, n_pairs, pair_chunk):
        end = min(start + pair_chunk, n_pairs)
        args = (
            pair_gauss[start:end], pair_tile_x[start:end],
            pair_tile_y[start:end], pair_b[start:end],
            means_bg, cos_bg, sin_bg, inv_sa2_bg, inv_sc2_bg, opacity_bg,
            off_x_f, off_y_f, off_x_i, off_y_i,
            float(x0), float(y0), T, Hp, Wp, B,
        )
        if use_checkpoint:
            contrib = checkpoint(_splat_tile_chunk, *args, use_reentrant=False)
        else:
            contrib = _splat_tile_chunk(*args)
        accum = accum + contrib

    combined = 1.0 - torch.exp(accum)  # accum <= 0, so combined in [0, 1)
    combined = combined.reshape(B, Hp, Wp)[:, :h, :w]
    return combined


def splat_render_cubics(
    cubics: torch.Tensor,
    stroke_widths: torch.Tensor,
    canvas_size: int = 224,
    num_samples: int = 20,
    pixel_chunk_size: int = 2048,
    opacities: torch.Tensor | None = None,
    pixel_box: tuple[int, int, int, int] | None = None,
    use_checkpoint: bool = True,
    use_compile: bool = False,
    tiling: str = "none",
    tile_size: int = 16,
) -> torch.Tensor:
    """Render cubic Bezier strokes via Gaussian splatting.

    Pure PyTorch, fully differentiable, batched.

    Args:
        cubics: (B, num_strokes, 4, 2) cubic Bezier control points in [-1, 1].
        stroke_widths: (B, num_strokes) stroke width sigma in pixels.
        canvas_size: Output image resolution (square).
        num_samples: Number of sample points per Bezier curve.
        pixel_chunk_size: Pixels to process at once (memory control).
        opacities: (B, num_strokes) per-stroke opacity in [0, 1]. Defaults to 1.
        pixel_box: (y0, x0, h, w) in pixel coordinates. When set, only that
            window is rasterized and the output is (B, h, w), matching the
            [y0:y0+h, x0:x0+w] slice of the full render. None (default)
            renders the full canvas.
        use_checkpoint: Wrap per-chunk splatting in gradient checkpointing
            (default True, the original behavior). Checkpointing recomputes
            the forward during backward to save memory; set False to trade
            memory for speed at small gaussian counts.
        use_compile: Run the splat kernel through torch.compile (default
            False, the original eager behavior). Measured ~1.7x at ~100
            gaussians and ~8.6x at ~10k gaussians on an RTX 5090; outputs
            match eager to fp32 noise (~1e-7), not bitwise. First call per
            new shape pays compile latency (seconds). Availability is
            verified once per device type by compiling the kernel in a
            subprocess (seconds, one-time); if that preflight fails, warns
            once and falls back to eager. Applies to the dense path only;
            the tiled path ignores it.
        tiling: "none" (default) is the dense all-pairs path, unchanged.
            "tiles" culls gaussians to fixed-size tiles via conservative
            bounding boxes before evaluation — same hard-cutoff semantics,
            outputs match dense to fp32 noise (not bitwise). "auto" picks
            "tiles" at or above _TILING_AUTO_THRESHOLD_G gaussians
            (num_strokes * num_samples), else "none".
        tile_size: Tile edge length in pixels for the tiled path. Default 16
            measured fastest at gaussian counts >= 512 on an RTX 5090 (32 was
            marginally better below that, within ~5%).

    Note on determinism: the tiled path accumulates with atomic scatter-adds
    on CUDA (forward log-transmittance and backward parameter gradients), so
    tiled outputs and gradients can vary at ~1-ulp scale between identical
    runs on CUDA (measured max 1.2e-7 forward). On CPU the tiled path is
    deterministic; the dense path is deterministic on both.

    Returns:
        (B, H, W) grayscale image ((B, h, w) when pixel_box is set).
        White background (1.0), black strokes (0.0).
    """
    B, num_strokes, _, _ = cubics.shape
    device = cubics.device
    dtype = cubics.dtype
    H = W = canvas_size
    K = num_samples

    if tiling not in ("none", "tiles", "auto"):
        raise ValueError(f'tiling must be "none", "tiles" or "auto", got {tiling!r}')
    if tiling == "auto":
        tiling = "tiles" if num_strokes * K >= _TILING_AUTO_THRESHOLD_G else "none"
    if tiling == "tiles" and (not isinstance(tile_size, int) or tile_size < 1):
        raise ValueError(f"tile_size must be a positive int, got {tile_size!r}")

    if pixel_box is not None:
        y0, x0, box_h, box_w = (int(v) for v in pixel_box)
        if box_h <= 0 or box_w <= 0:
            raise ValueError(f"pixel_box height/width must be positive, got {pixel_box}")
        if y0 < 0 or x0 < 0 or y0 + box_h > H or x0 + box_w > W:
            raise ValueError(f"pixel_box {pixel_box} exceeds canvas of size {canvas_size}")
        pixel_box = (y0, x0, box_h, box_w)
        out_h, out_w = box_h, box_w
    else:
        out_h, out_w = H, W

    # Sample points and tangents along curves
    t_vals = torch.linspace(0, 1, K, device=device, dtype=dtype)
    positions = _evaluate_bezier(cubics, t_vals)  # (B, num_strokes, K, 2)
    tangents = _evaluate_bezier_tangent(cubics, t_vals)

    # Convert from [-1, 1] to pixel coordinates
    means = (positions + 1.0) / 2.0 * canvas_size  # (B, num_strokes, K, 2)

    # Compute rotation angles from tangents
    angles = torch.atan2(tangents[..., 1], tangents[..., 0])  # (B, num_strokes, K)

    # Compute sigma_along (half distance between consecutive samples)
    diffs = means[:, :, 1:, :] - means[:, :, :-1, :]
    dists = torch.norm(diffs, dim=-1)  # (B, num_strokes, K-1)

    sigma_along = torch.zeros(B, num_strokes, K, device=device, dtype=dtype)
    sigma_along[:, :, 1:] += dists
    sigma_along[:, :, :-1] += dists
    sigma_along[:, :, 1:-1] /= 2.0
    sigma_along = sigma_along * 0.5
    sigma_along = sigma_along.clamp(min=0.1)

    # sigma_across from stroke width
    sigma_across = stroke_widths.unsqueeze(-1).expand(-1, -1, K)  # (B, num_strokes, K)

    # Per-stroke opacity (expanded to per-Gaussian)
    if opacities is not None:
        opacity_per_gauss = opacities.unsqueeze(-1).expand(-1, -1, K)  # (B, num_strokes, K)
    else:
        opacity_per_gauss = torch.ones(B, num_strokes, K, device=device, dtype=dtype)

    # Precompute rotation and inverse variance terms
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    inv_sa2 = 1.0 / (sigma_along * sigma_along + 1e-8)
    inv_sc2 = 1.0 / (sigma_across * sigma_across + 1e-8)

    # Flatten for efficient computation
    G = num_strokes * K
    means_flat = means.reshape(B, G, 2)
    cos_flat = cos_a.reshape(B, G)
    sin_flat = sin_a.reshape(B, G)
    inv_sa2_flat = inv_sa2.reshape(B, G)
    inv_sc2_flat = inv_sc2.reshape(B, G)
    opacity_flat = opacity_per_gauss.reshape(B, G)

    if tiling == "tiles":
        region = pixel_box if pixel_box is not None else (0, 0, H, W)
        output = _splat_tiled(
            means_flat, cos_flat, sin_flat, inv_sa2_flat, inv_sc2_flat,
            opacity_flat, region, tile_size, use_checkpoint,
        )
        return 1.0 - output.clamp(0.0, 1.0)

    # Create pixel grid (cached; restricted to pixel_box when set)
    pixels = _get_pixel_grid(canvas_size, pixel_box, device, dtype)
    total_pixels = pixels.shape[0]

    # Adaptive chunk size for memory efficiency
    bytes_per_element = 4 if dtype == torch.float32 else 2
    max_chunk_memory = 512 * 1024 * 1024
    adaptive_chunk = max(64, max_chunk_memory // (B * G * bytes_per_element * 6))
    chunk_size = min(pixel_chunk_size, adaptive_chunk, total_pixels)

    # Process in chunks, with gradient checkpointing unless disabled
    splat_fn = _get_splat_chunk_fn(use_compile, device)
    chunks = []
    for start in range(0, total_pixels, chunk_size):
        end = min(start + chunk_size, total_pixels)
        chunk_pixels = pixels[start:end]
        if use_checkpoint:
            chunk_out = checkpoint(
                splat_fn, chunk_pixels,
                means_flat, cos_flat, sin_flat, inv_sa2_flat, inv_sc2_flat, opacity_flat,
                use_reentrant=False,
            )
        else:
            chunk_out = splat_fn(
                chunk_pixels,
                means_flat, cos_flat, sin_flat, inv_sa2_flat, inv_sc2_flat, opacity_flat,
            )
        chunks.append(chunk_out)

    output = torch.cat(chunks, dim=1)
    output = output.reshape(B, out_h, out_w)
    result = 1.0 - output.clamp(0.0, 1.0)

    return result


def split_path_to_cubics(
    points: torch.Tensor,
    num_control_points: torch.Tensor,
) -> torch.Tensor:
    """Split a multi-segment pydiffvg Path into individual cubic Beziers.

    Converts lines (0 ctrl pts) and quadratics (1 ctrl pt) to cubics (3 ctrl pts).

    Args:
        points: (N, 2) all points in the path
        num_control_points: (num_segments,) control points per segment

    Returns:
        (num_segments, 4, 2) cubic Bezier control points
    """
    cubics = []
    point_idx = 0

    for seg_idx, n_ctrl in enumerate(num_control_points):
        n_ctrl = int(n_ctrl.item())

        p0 = points[point_idx]

        if n_ctrl == 0:
            # Line segment: P0 -> P1
            p3 = points[point_idx + 1]
            # Convert to cubic: control points at 1/3 and 2/3
            p1 = p0 + (p3 - p0) / 3.0
            p2 = p0 + 2.0 * (p3 - p0) / 3.0
            point_idx += 1

        elif n_ctrl == 1:
            # Quadratic: P0, C, P2 -> Cubic: P0, P0+2/3*(C-P0), P2+2/3*(C-P2), P2
            c = points[point_idx + 1]
            p3 = points[point_idx + 2]
            p1 = p0 + 2.0 / 3.0 * (c - p0)
            p2 = p3 + 2.0 / 3.0 * (c - p3)
            point_idx += 2

        elif n_ctrl == 2:
            # Already cubic: P0, C1, C2, P3
            p1 = points[point_idx + 1]
            p2 = points[point_idx + 2]
            p3 = points[point_idx + 3]
            point_idx += 3

        else:
            raise ValueError(f"Unsupported num_control_points: {n_ctrl}")

        cubic = torch.stack([p0, p1, p2, p3], dim=0)
        cubics.append(cubic)

    return torch.stack(cubics, dim=0)


class SplatRenderFunction:
    """Drop-in replacement for pydiffvg.RenderFunction using Bezier splatting.

    Uses pure PyTorch operations, so gradients work on all GPU architectures.
    """

    @staticmethod
    def serialize_scene(
        canvas_width: int,
        canvas_height: int,
        shapes: list,
        shape_groups: list,
        **kwargs,  # Accept but ignore filter, output_type, etc.
    ) -> tuple:
        """Extract rendering data from pydiffvg shapes.

        Returns a tuple that can be passed to apply().
        """
        all_cubics = []
        all_stroke_widths = []
        all_opacities = []

        # Build shape_id -> group mapping for stroke colors
        shape_id_to_group = {}
        for group in shape_groups:
            for sid in group.shape_ids:
                shape_id_to_group[int(sid.item())] = group

        for shape_idx, shape in enumerate(shapes):
            if hasattr(shape, 'num_control_points') and hasattr(shape, 'points'):
                cubics = split_path_to_cubics(shape.points, shape.num_control_points)
                all_cubics.append(cubics)
                num_segments = cubics.shape[0]
                all_stroke_widths.extend([shape.stroke_width] * num_segments)

                # Extract opacity from stroke_color alpha channel
                group = shape_id_to_group.get(shape_idx)
                opacity = torch.tensor(1.0)
                if group is not None and hasattr(group, 'stroke_color') and group.stroke_color is not None:
                    sc = group.stroke_color
                    # Handle both raw tensor and SolidColor object
                    if hasattr(sc, 'color'):
                        sc = sc.color
                    if isinstance(sc, torch.Tensor) and sc.numel() >= 4:
                        opacity = sc[3]  # RGBA alpha
                all_opacities.extend([opacity] * num_segments)

        if len(all_cubics) == 0:
            return (canvas_width, canvas_height, None, None, None)

        # Stack all cubics: (total_segments, 4, 2)
        all_cubics = torch.cat(all_cubics, dim=0)
        all_stroke_widths = torch.stack(all_stroke_widths)
        all_opacities = torch.stack(all_opacities)

        return (canvas_width, canvas_height, all_cubics, all_stroke_widths, all_opacities)

    @staticmethod
    def apply(
        width: int,
        height: int,
        num_samples_x: int,  # Ignored (for interface compatibility)
        num_samples_y: int,  # Ignored
        seed: int,           # Ignored
        background_image,    # Ignored for now
        *scene_args,
    ) -> torch.Tensor:
        """Render the scene to an RGBA image.

        Returns:
            (H, W, 4) RGBA tensor, values in [0, 1]
        """
        canvas_width, canvas_height, all_cubics, all_stroke_widths, all_opacities = scene_args[:5]

        if all_cubics is None:
            return torch.ones(height, width, 4)

        # Ensure all tensors are on the same device
        device = all_cubics.device
        all_stroke_widths = all_stroke_widths.to(device)
        all_opacities = all_opacities.to(device)

        # Normalize coordinates: [0, canvas_size] -> [-1, 1]
        scale = max(canvas_width, canvas_height)
        cubics_normalized = (all_cubics / scale) * 2.0 - 1.0

        # Stroke widths stay in pixel space (splat_render_cubics expects pixels)
        # sigma_across = stroke_width / 2 for half-width Gaussian
        stroke_widths_px = all_stroke_widths / 2.0

        # Add batch dimension: (1, num_segments, 4, 2)
        cubics_batched = cubics_normalized.unsqueeze(0)
        widths_batched = stroke_widths_px.unsqueeze(0)
        opacities_batched = all_opacities.unsqueeze(0)

        # Render grayscale
        grayscale = splat_render_cubics(
            cubics_batched,
            widths_batched,
            opacities=opacities_batched,
            canvas_size=max(width, height),
            num_samples=20,
        )  # (1, H, W)

        # Crop to actual size if non-square
        grayscale = grayscale[0, :height, :width]  # (H, W)

        # Convert to RGBA
        rgb = grayscale.unsqueeze(-1).expand(-1, -1, 3)  # (H, W, 3)
        alpha = torch.ones_like(grayscale).unsqueeze(-1)  # (H, W, 1)
        rgba = torch.cat([rgb, alpha], dim=-1)  # (H, W, 4)

        return rgba
