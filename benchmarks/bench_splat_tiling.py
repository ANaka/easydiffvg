"""Tiling sweep benchmark for pydiffvg.splat_render.splat_render_cubics.

Compares the dense splatting path (tiling="none") against the tile-culled
path (tiling="tiles", tile_size in {16, 32}) across gaussian counts, plus a
dense + use_compile=True context arm (so tiling can be compared against just
compiling). G = num_strokes * 16 samples per curve.

Sweep (float32, B=1, num_samples=16, fixed seeds, loss = MSE vs a fixed
random target over the full frame):
  canvas 384, G in {96, 512, 2048, 10240, 40960}  (6..2560 strokes)
  canvas 768, G = 10240

Before timing, each tiled arm is checked for forward exactness against the
dense arm (max abs diff <= 1e-5; the actual value is printed). A failure is
reported as FAILED EXACTNESS for that cell and the sweep continues. The
report shows ms/iter mean (median) per arm, peak GPU MB per arm, the
best-tiled-vs-dense speedup per row, and a log-log interpolated crossover
estimate suggesting a tiling="auto" threshold.

The tiling/tile_size kwargs are being implemented concurrently: if they do
not exist yet (TypeError on probe), the script prints a notice, runs the
dense arms only, and exits 0.

Note on the compile arm: it needs a working torch.compile toolchain (CPATH
must be set for Python.h on this machine). The renderer's own availability
probe (`_compile_available`) compiles a trivial function, which can succeed
even when the real splat kernel fails to build - and a failed in-process
torch.compile of the splat kernel poisons all later grad-mode eager calls
(verified: subsequent use_compile=False runs raise InductorError and
torch._dynamo.reset() does not recover). The script therefore (a) runs all
compile arms in a second phase after the dense/tiled sweep, and (b) first
verifies the real kernel compiles in a throwaway subprocess; if that
preflight fails, compile cells are reported as "compile-broken". When
`_compile_available` returns False the renderer falls back to eager and the
arm is labeled "eager-fallback".

Usage:
    uv run python benchmarks/bench_splat_tiling.py [--device cuda|cpu]
        [--quick] [--iters-scale FLOAT] [--skip-compile-arm]
"""

import argparse
import math
import statistics
import subprocess
import sys
import time

import torch

import pydiffvg.splat_render as splat_mod

NUM_SAMPLES = 16
EXACTNESS_TOL = 1e-5
BASE_ITERS = {96: 200, 512: 100, 2048: 50, 10240: 20, 40960: 10}
QUICK_ITERS = 4

# Run in a subprocess: does torch.compile of the real splat kernel work at
# all (fwd+bwd)? Kept tiny; only the build-toolchain outcome matters.
PREFLIGHT_CODE = """
import sys, torch
import pydiffvg.splat_render as sm
dev = torch.device(sys.argv[1])
torch.manual_seed(0)
cub = (torch.rand(1, 2, 4, 2, device=dev) * 2 - 1).requires_grad_(True)
sw = (torch.rand(1, 2, device=dev) * 3 + 1).requires_grad_(True)
out = sm.splat_render_cubics(cub, sw, canvas_size=32, num_samples=8,
                             use_compile=True)
out.mean().backward()
print("PREFLIGHT_COMPILE_OK")
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().mean()


def make_base_inputs(num_strokes: int, canvas: int):
    """Deterministic inputs, generated on CPU with a fixed seed."""
    torch.manual_seed(0)
    cubics = torch.rand(1, num_strokes, 4, 2) * 2.0 - 1.0        # [-1, 1]
    stroke_widths = torch.rand(1, num_strokes) * 3.0 + 1.0       # [1, 4] px
    opacities = torch.rand(1, num_strokes) * 0.5 + 0.5           # [0.5, 1]
    target = torch.rand(1, canvas, canvas)                       # [0, 1]
    return cubics, stroke_widths, opacities, target


def make_leaves(base_tensors, device):
    """Fresh leaf tensors (requires_grad) with identical values per arm."""
    return [t.detach().clone().to(device).requires_grad_(True) for t in base_tensors]


def timed_run(step, params, warmup: int, iters: int, device: torch.device):
    """Run `step` (fwd+bwd) warmup+iters times; return (mean_ms, median_ms, peak_mb).

    Warmup also absorbs first-call torch.compile latency for the compile arm.
    """
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        for p in params:
            p.grad = None
        step()
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times_ms = []
    for _ in range(iters):
        for p in params:
            p.grad = None
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step()
        if is_cuda:
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else float("nan")
    return statistics.mean(times_ms), statistics.median(times_ms), peak_mb


def print_device_info(device: torch.device):
    print(f"torch {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            print("nvidia-smi (current utilization, for idle/contended context):")
            for line in out.splitlines():
                print(f"  {line}")
        except Exception as exc:  # noqa: BLE001 - report-only benchmark
            print(f"nvidia-smi unavailable: {exc}")
    else:
        print("Device: CPU (no GPU memory / utilization reporting)")


def fmt_time(res) -> str:
    """Format an arm result for the timing table."""
    if isinstance(res, tuple):
        return f"{res[0]:.2f} ({res[1]:.2f})"
    return res  # "OOM", "FAILED EXACTNESS", "not implemented", "ERR(...)", ...


def fmt_mem(res) -> str:
    """Format an arm result for the memory table."""
    if isinstance(res, tuple):
        return "n/a" if res[2] != res[2] else f"{res[2]:.1f}"  # NaN check
    return res


def detect_tiling_support(fn) -> bool:
    """Probe whether splat_render_cubics accepts tiling/tile_size kwargs."""
    cubics = torch.zeros(1, 1, 4, 2)
    widths = torch.ones(1, 1)
    try:
        with torch.no_grad():
            fn(cubics, widths, canvas_size=8, num_samples=4,
               tiling="none", tile_size=32)
    except TypeError as exc:
        msg = str(exc)
        if "tiling" in msg or "tile_size" in msg or "unexpected keyword" in msg:
            return False
        raise
    return True


def preflight_compile(device: torch.device) -> tuple[bool, str]:
    """Check in a throwaway subprocess that use_compile=True actually works.

    A failed in-process compile poisons later eager grad-mode calls, so the
    check must not run in this process.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PREFLIGHT_CODE, device.type],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "preflight subprocess timed out (600s)"
    except Exception as exc:  # noqa: BLE001 - report-only benchmark
        return False, f"preflight subprocess failed to launch: {exc}"
    if proc.returncode == 0 and "PREFLIGHT_COMPILE_OK" in proc.stdout:
        return True, ""
    lines = [ln.strip() for ln in (proc.stderr + "\n" + proc.stdout).splitlines()
             if ln.strip()]
    reason = next((ln for ln in reversed(lines) if "rror" in ln),
                  lines[-1] if lines else f"exit code {proc.returncode}")
    return False, reason[:200]


def forward_no_grad(fn, base, device, canvas, extra_kwargs):
    """One forward pass with detached inputs (for exactness checks)."""
    cub, sw, op = [t.detach().to(device) for t in base[:3]]
    with torch.no_grad():
        return fn(cub, sw, canvas_size=canvas, num_samples=NUM_SAMPLES,
                  opacities=op, **extra_kwargs)


def run_arm(fn, base, target, device, canvas, extra_kwargs, warmup, iters):
    """Time one arm (fwd+bwd); returns (mean, median, peak_mb) or an error string."""
    params = make_leaves(base[:3], device)
    cub, sw, op = params

    def step():
        out = fn(cub, sw, canvas_size=canvas, num_samples=NUM_SAMPLES,
                 opacities=op, **extra_kwargs)
        mse(out, target).backward()

    try:
        return timed_run(step, params, warmup, iters, device)
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return "OOM"
    except Exception as exc:  # noqa: BLE001 - concurrent impl may be incomplete
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [arm error] {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:200] if str(exc) else '(no message)'}")
        return f"ERR({type(exc).__name__})"


def crossover_estimate(points):
    """Estimate the G where best-tiled ms/iter equals dense ms/iter.

    points: list of (G, speedup_ratio) sorted by G, where
    ratio = dense_mean_ms / best_tiled_mean_ms (>1 means tiled wins).
    Interpolates log(ratio) vs log(G) between the adjacent points where the
    ratio crosses 1.0. Returns (G_estimate, None) or (None, message).
    """
    pts = [(g, r) for g, r in points if isinstance(r, float) and r == r and r > 0]
    if len(pts) < 2:
        return None, "not enough data points for a crossover estimate"
    if all(r > 1.0 for _, r in pts):
        return None, (f"tiled wins at every measured G "
                      f"(min speedup {min(r for _, r in pts):.2f}x); "
                      f"tiling='auto' could always tile in this range")
    if all(r < 1.0 for _, r in pts):
        return None, (f"tiled never wins in this range "
                      f"(max speedup {max(r for _, r in pts):.2f}x); "
                      f"tiling='auto' should stay dense here")
    for (g1, r1), (g2, r2) in zip(pts, pts[1:]):
        y1, y2 = math.log(r1), math.log(r2)
        if y1 == 0.0:
            return float(g1), None
        if y2 == 0.0:
            return float(g2), None
        if y1 * y2 < 0:
            x1, x2 = math.log(g1), math.log(g2)
            x = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            return math.exp(x), None
    return None, "speedup crosses 1.0 non-monotonically; no clean crossover estimate"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quick", action="store_true",
                        help="Tiny validation run: canvas 96, G in {96, 512}, "
                             f"{QUICK_ITERS} timed iters, no 768 row")
    parser.add_argument("--iters-scale", type=float, default=1.0,
                        help="Multiply timed iteration counts by this factor")
    parser.add_argument("--skip-compile-arm", action="store_true",
                        help="Skip the dense + use_compile=True context arm")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; rerun with --device cpu")
    device = torch.device(args.device)

    if args.quick:
        primary_canvas = 96
        sweep_gs = [96, 512]
        rows = [(primary_canvas, g) for g in sweep_gs]
    else:
        primary_canvas = 384
        sweep_gs = [96, 512, 2048, 10240, 40960]
        rows = [(primary_canvas, g) for g in sweep_gs] + [(768, 10240)]

    def iters_for(g: int) -> tuple[int, int]:
        base = QUICK_ITERS if args.quick else BASE_ITERS[g]
        iters = max(1, round(base * args.iters_scale))
        warmup = max(3, iters // 4)
        return iters, warmup

    fn = splat_mod.splat_render_cubics
    tiling_supported = detect_tiling_support(fn)
    dense_kwargs = {"tiling": "none"} if tiling_supported else {}

    print("# splat_render_cubics tiling sweep benchmark")
    print_device_info(device)
    print(f"config: device={device.type} quick={args.quick} "
          f"iters_scale={args.iters_scale} num_samples={NUM_SAMPLES} "
          f"primary_canvas={primary_canvas} Gs={sweep_gs} "
          f"exactness_tol={EXACTNESS_TOL:g}")
    print("note: the dense+compile arm requires a working torch.compile "
          "toolchain (CPATH set for Python.h on this machine); it runs after "
          "the main sweep, gated by a subprocess preflight, because a failed "
          "in-process compile poisons later eager grad-mode calls.")

    if not tiling_supported:
        print()
        print("tiled path not implemented yet — skipping tiled arms "
              "(splat_render_cubics does not accept tiling/tile_size kwargs; "
              "re-run after the tiled implementation lands)")

    # Main-sweep arm specs: (key, column label, extra kwargs).
    arm_specs = [("dense", "dense", dict(dense_kwargs))]
    if tiling_supported:
        arm_specs.append(("tiled16", "tiled ts=16",
                          {"tiling": "tiles", "tile_size": 16}))
        arm_specs.append(("tiled32", "tiled ts=32",
                          {"tiling": "tiles", "tile_size": 32}))
    print()

    # -----------------------------------------------------------------
    # Phase 1: dense + tiled sweep
    # -----------------------------------------------------------------
    results = []
    for canvas, g in rows:
        num_strokes = g // NUM_SAMPLES
        iters, warmup = iters_for(g)
        print(f"-- canvas {canvas}, G={g} ({num_strokes} strokes): "
              f"iters={iters} warmup={warmup}")
        base = make_base_inputs(num_strokes, canvas)
        target = base[3].to(device)
        row = {"canvas": canvas, "g": g, "num_strokes": num_strokes, "arms": {}}

        # Dense forward reference for the exactness checks.
        dense_ref = None
        if tiling_supported:
            try:
                dense_ref = forward_no_grad(fn, base, device, canvas, dense_kwargs)
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("  [check] dense reference forward OOM; "
                      "exactness checks skipped for this row")
            except Exception as exc:  # noqa: BLE001
                print(f"  [check] dense reference forward failed "
                      f"({type(exc).__name__}); exactness checks skipped")

        for key, label, extra_kwargs in arm_specs:
            # Exactness gate for tiled arms.
            if key.startswith("tiled"):
                try:
                    out = forward_no_grad(fn, base, device, canvas, extra_kwargs)
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row["arms"][key] = "OOM"
                    print(f"  [{label}] OOM during exactness forward")
                    continue
                except Exception as exc:  # noqa: BLE001
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row["arms"][key] = f"ERR({type(exc).__name__})"
                    print(f"  [{label}] exactness forward failed: "
                          f"{type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:200] if str(exc) else ''}")
                    continue
                if dense_ref is not None:
                    if out.shape != dense_ref.shape:
                        row["arms"][key] = "FAILED EXACTNESS"
                        print(f"  FAILED EXACTNESS: {label} at canvas {canvas} "
                              f"G={g}: shape {tuple(out.shape)} != dense "
                              f"{tuple(dense_ref.shape)}")
                        continue
                    diff = (out - dense_ref).abs().max().item()
                    if diff > EXACTNESS_TOL:
                        row["arms"][key] = "FAILED EXACTNESS"
                        print(f"  FAILED EXACTNESS: {label} at canvas {canvas} "
                              f"G={g}: max abs fwd diff {diff:.3e} > "
                              f"{EXACTNESS_TOL:g}")
                        continue
                    print(f"  [check] {label} vs dense: max abs fwd diff "
                          f"{diff:.3e} (tol {EXACTNESS_TOL:g}) OK")
                del out

            row["arms"][key] = run_arm(fn, base, target, device, canvas,
                                       extra_kwargs, warmup, iters)
        results.append(row)
    print()

    # -----------------------------------------------------------------
    # Phase 2: dense + use_compile=True context arm
    # -----------------------------------------------------------------
    compile_col = None  # (key, column label)
    if not args.skip_compile_arm:
        compile_label = "dense+compile"
        try:
            compile_avail = splat_mod._compile_available(device)
        except AttributeError:
            compile_avail = None  # renamed upstream; let the preflight decide
        if compile_avail is False:
            compile_label = "dense+compile (eager-fallback)"
        run_compile = True
        broken_msg = None
        if compile_avail is not False:
            ok, why = preflight_compile(device)
            if not ok:
                run_compile = False
                broken_msg = why
        compile_col = ("compile", compile_label)

        if run_compile:
            print(f"-- compile context arm ({compile_label})")
            for row in results:
                canvas, g = row["canvas"], row["g"]
                iters, warmup = iters_for(g)
                base = make_base_inputs(row["num_strokes"], canvas)
                target = base[3].to(device)
                row["arms"]["compile"] = run_arm(
                    fn, base, target, device, canvas,
                    {**dense_kwargs, "use_compile": True}, warmup, iters)
                print(f"   canvas {canvas}, G={g}: "
                      f"{fmt_time(row['arms']['compile'])} ms/iter")
        else:
            print("-- compile context arm: torch.compile of the splat kernel "
                  "fails on this machine (preflight subprocess); not run "
                  "in-process to avoid poisoning. Reason:")
            print(f"   {broken_msg}")
            for row in results:
                row["arms"]["compile"] = "compile-broken"
        print()

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    def row_label(row):
        return (f"canvas {row['canvas']}, G={row['g']} "
                f"({row['num_strokes']} strokes)")

    def best_tiled(row):
        """(key, mean_ms) of the fastest timed tiled arm, or None."""
        cands = [(k, row["arms"][k][0]) for k in ("tiled16", "tiled32")
                 if isinstance(row["arms"].get(k), tuple)]
        return min(cands, key=lambda kv: kv[1]) if cands else None

    report_cols = [(key, label) for key, label, _ in arm_specs]
    if not tiling_supported:
        report_cols += [("tiled16", "tiled ts=16"), ("tiled32", "tiled ts=32")]
    if compile_col is not None:
        report_cols.append(compile_col)

    print("## Results  (ms/iter = forward+backward, mean (median))")
    print()
    print("| config | " + " ms/iter | ".join(lbl for _, lbl in report_cols)
          + " ms/iter | best-tiled speedup |")
    print("|" + "---|" * (len(report_cols) + 2))
    for row in results:
        cells = [fmt_time(row["arms"].get(key, "not implemented"))
                 for key, _ in report_cols]
        bt = best_tiled(row)
        dense_res = row["arms"].get("dense")
        if bt is not None and isinstance(dense_res, tuple):
            speedup = f"{dense_res[0] / bt[1]:.2f}x ({bt[0]})"
        else:
            speedup = "n/a"
        print(f"| {row_label(row)} | " + " | ".join(cells) + f" | {speedup} |")
    print()

    print("## Peak GPU memory (MB, per arm)")
    print()
    print("| config | " + " | ".join(lbl for _, lbl in report_cols) + " |")
    print("|" + "---|" * (len(report_cols) + 1))
    for row in results:
        cells = [fmt_mem(row["arms"].get(key, "not implemented"))
                 for key, _ in report_cols]
        print(f"| {row_label(row)} | " + " | ".join(cells) + " |")
    print()

    # -----------------------------------------------------------------
    # Crossover estimate (primary-canvas sweep only)
    # -----------------------------------------------------------------
    if tiling_supported:
        points = []
        for row in results:
            if row["canvas"] != primary_canvas:
                continue
            bt = best_tiled(row)
            dense_res = row["arms"].get("dense")
            if bt is not None and isinstance(dense_res, tuple):
                points.append((row["g"], dense_res[0] / bt[1]))
        points.sort()
        g_est, msg = crossover_estimate(points)
        if g_est is not None:
            print(f"suggested tiling='auto' threshold "
                  f"(canvas {primary_canvas}): G ≈ {round(g_est)}")
        else:
            print(f"crossover estimate (canvas {primary_canvas}): {msg}")
    else:
        print("crossover estimate: n/a (tiled path not implemented yet)")


if __name__ == "__main__":
    main()
