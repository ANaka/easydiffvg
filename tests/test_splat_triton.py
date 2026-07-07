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
