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
