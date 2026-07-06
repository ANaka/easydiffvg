# Design note: a Triton tile kernel for the splat renderer

*Status: design only. Gate for this note (tiled path exact and ≥5× at G=10k)
was met: ≤4.2e-7 forward deviation, 14.6× at G=10,240 / canvas 384.*

## Where the remaining time goes

The PyTorch tiled path (`tiling="tiles"`) is memory-bandwidth-bound, not
compute-bound. Per (gaussian, tile) pair it materializes ~8 intermediate
tensors of `tile_size²` elements in global memory (dx, dy, d_along, d_across,
mahal, alpha, log1p, indices), scatter-adds through global-memory atomics, and
runs ~15 separate kernel launches per pair-chunk plus the same again in the
checkpointed backward. At G=10,240 that is ~50 ms/iter of mostly memory
traffic on arithmetic a 5090 could do in single-digit milliseconds.

## Kernel sketch

3DGS-style, one Triton program per tile (per batch element):

- **Forward**: each program loads its tile's gaussian list (from the same
  pair-building pass we already have — keep that in PyTorch, it is cheap and
  index-only), iterates gaussians in blocks, keeps the per-pixel
  log-transmittance accumulator in **registers**, and writes each output pixel
  once. No intermediates in global memory, no atomics in the forward, no
  log/exp roundtrip needed per gaussian (accumulate the product directly or
  in log space — register-resident either way).
- **Backward**: custom `autograd.Function`. Recompute alpha per gaussian in
  registers (checkpointing for free), accumulate ∂L/∂(mean, angle, inv_σ²,
  opacity) per gaussian in shared memory across the tile's pixels, then one
  atomic add per gaussian parameter per tile into global grads.

## Expected additional gain

Bandwidth accounting (≥8 global-memory round trips per pair-element today vs
~1 write per pixel + parameter reads) suggests **3–8× over the PyTorch tiled
path** at G ≥ 10k: ~50 ms → 7–15 ms at G=10,240; ~180 ms → 25–60 ms at
G=40,960. Reference points: 3DGS renderers rasterize 100k+ gaussians in
single-digit ms forward at similar resolutions. Below ~1k gaussians the fixed
launch overhead dominates either way and the gain shrinks toward 1×.

## Precision risks

- Register accumulation is deterministic per tile — *better* than today's
  CUDA scatter-add atomics (~1-ulp run-to-run noise). Backward atomics remain
  per-parameter, same noise class as now.
- The hard cutoff `alpha·(mahal² < 20)` must be replicated exactly — the
  gray-wash regression history makes this the first thing to test.
- Hand-derived backward is the real risk: a subtle gradient mismatch would
  pass eyeball tests and corrupt training. Mitigation: the existing
  tiled-vs-dense 1e-5 gates already compare full gradients; add fp64
  `gradcheck` on tiny scenes and run the adversarial suite (borders,
  off-canvas, σ→0, opacity 0) against the Triton path unchanged.

## Maintenance cost

Two Triton kernels + a custom autograd.Function (~400–600 lines), a CPU/eager
fallback path (Triton is CUDA/ROCm-only — the pure-PyTorch tiled path must
stay), pinning to the Triton that ships with torch (2.10 bundles a
sm_120-capable Triton; verified working on this 5090), and a doubled test
matrix. Debugging register-level kernels is markedly harder than the current
all-tensor code.

## Recommendation

Do it only if scene-scale rendering still dominates end-to-end pipeline time
*after* adopting `tiling="auto"` downstream — the PyTorch tiled path already
took G=10k from 728 ms to 50 ms. Measure the pipeline first; if rendering is
under ~30% of wall time, the 3–8× here buys little for its complexity.
