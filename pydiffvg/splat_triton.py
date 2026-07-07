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
