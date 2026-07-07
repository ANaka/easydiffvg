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


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _scene(strokes, seed=0, device="cuda"):
    torch.manual_seed(seed)
    c = (torch.rand(1, strokes, 4, 2, device=device) * 2 - 1).requires_grad_(True)
    w = (torch.rand(1, strokes, device=device) * 3 + 0.5).requires_grad_(True)
    o = (torch.rand(1, strokes, device=device) * 0.5 + 0.5).requires_grad_(True)
    return c, w, o


@cuda_only
@pytest.mark.skip(reason="tiling='triton' wired in Task 4")
@pytest.mark.parametrize("tile", [16, 32])
def test_triton_forward_matches_dense(tile):
    from pydiffvg.splat_render import splat_render_cubics

    c, w, o = _scene(6)
    dense = splat_render_cubics(c, w, canvas_size=384, num_samples=16, opacities=o)
    tri = splat_render_cubics(c, w, canvas_size=384, num_samples=16, opacities=o,
                              tiling="triton", tile_size=tile)
    assert (dense - tri).abs().max().item() <= 1e-5


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
