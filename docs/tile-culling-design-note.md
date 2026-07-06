# Design note: tile-based gaussian culling for `splat_render_cubics`

*Status: design only — deliberately not implemented (out of scope for the speed pass).*

## Problem

The splat kernel is dense: every gaussian is evaluated against every pixel chunk,
O(G × P) pairs. At the scene shape (600 strokes × 16 samples = 9,600 gaussians,
384² = 147k pixels) that is ~1.4B pair evaluations per forward, ~670 ms/iter
fwd+bwd on an RTX 5090. Yet the renderer already zeroes every contribution beyond
the mahalanobis cutoff (`alpha * (mahal_sq < 20.0)`), so the overwhelming majority
of pairs are computed and then multiplied by zero: a gaussian with σ ≈ 1–4 px
influences a disc of radius `r = sqrt(2·20)·σ ≈ 6.3σ ≈ 8–25 px` — under 0.5% of
the canvas.

## Key exactness property

Culling is **exact**, not approximate. For `mahal_sq ≥ 20` the forward
contribution is exactly 0 (hard mask) *and* the gradient contribution is exactly
0 (the mask zeroes the value path, and `clamp(max=20)` kills the gradient through
the exp path). So skipping any gaussian–pixel pair with a *conservative* bound
`mahal_sq ≥ 20` changes neither outputs nor gradients — the same argument that
makes `pixel_box` exact. No gray-wash risk: the cutoff semantics are untouched.

## Sketch

1. **Tiles**: split the canvas (or `pixel_box`) into T×T pixel tiles, T = 32
   (384² → 144 tiles; one tile ≈ one pixel chunk today).
2. **Conservative gaussian AABB**: half-extents
   `ex = 6.33·(|cosθ|·σ_along + |sinθ|·σ_across)`, `ey` symmetric — cheap,
   vectorized over all G, recomputed every step (means move during optimization).
3. **Binning**: map each gaussian's AABB to a tile-index range; build per-tile
   gaussian lists. In pure PyTorch: `argsort` by tile id + segment offsets
   (`bucketize`), or a (G × n_tiles) boolean overlap matrix at small scale.
4. **Ragged execution** (the hard part without custom kernels): pad each tile's
   list to a per-batch cap K and run one dense (n_tiles, K, T²) kernel via
   `gather`. Bucket K to powers of two to keep shapes compile-cache-friendly.
   Overflowing tiles (> K gaussians) fall back to the dense path for correctness.
5. **Compositing**: per-tile `1 − prod(1 − α)` is safe because gaussians outside
   a tile's lists contribute α = 0 there (multiplying by 1 − 0 is identity).

## Expected win and costs

At the scene shape the average gaussian overlaps ~1–9 tiles out of 144, i.e.
~15–100× fewer pairs; realistically 5–15× end-to-end after binning overhead,
padding waste, and gather/scatter in backward. Overheads paid *every step*:
AABB + sort ≈ O(G log G) (cheap), gather materialization (the real cost).
Greedy-shape workloads gain nothing — `pixel_box` already cut them to 2.9 ms/iter
with only ~96 gaussians in play.

## Risks / open questions

- Ragged batching across B with different tile occupancies; simplest is B=1 per
  call (matches downstream) or shared lists with per-batch masks.
- Backward through `gather` produces scatter-adds — nondeterministic accumulation
  order on CUDA; still within fp32 noise, but bitwise tests must relax to ≤1e-6.
- Interaction with `torch.compile` (measured 8.6× on the dense kernel at this
  shape): compile may capture much of the same win with zero complexity. **Do the
  compile opt-in first**; only build culling if scene-shape time still dominates
  after that, and validate against the existing exactness test pattern
  (slice-equality + grad ≤1e-6).
