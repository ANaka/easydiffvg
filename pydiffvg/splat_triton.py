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
    # Must match splat_render._MAHAL_SQ_CUTOFF (and the literal 20.0 in _splat_chunk).
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
