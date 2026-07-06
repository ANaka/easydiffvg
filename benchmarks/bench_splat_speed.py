"""Before/after speed benchmark for pydiffvg.splat_render.splat_render_cubics.

"Before" = the main-branch implementation (fetched at runtime via
`git show main:pydiffvg/splat_render.py`, imported as a separate module).
It lacks pixel_box, use_checkpoint and the module-level pixel-grid cache.

"After" = the installed pydiffvg.splat_render with the new kwargs.

Workloads (float32, B=1, canvas 384, num_samples=16, torch.manual_seed(0)):
  1. "greedy": 6 strokes; downstream only cares about a 96x96 window at
     (y0, x0) = (144, 144). Before: full 384^2 render + slice. After:
     pixel_box window render with use_checkpoint=False.
  2. "scene": 600 strokes, full 384^2 frame, defaults on both sides
     (pixel-grid cache is the only change). Plus an after-variant with
     use_checkpoint=False, guarded against CUDA OOM.

Usage:
    uv run python benchmarks/bench_splat_speed.py [--device cuda|cpu]
        [--quick] [--iters-scale FLOAT]
"""

import argparse
import atexit
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# "Before" implementation loading
# ---------------------------------------------------------------------------

def load_before_module():
    """Import main-branch pydiffvg/splat_render.py as a separate module."""
    src = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", "main:pydiffvg/splat_render.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    fd, path = tempfile.mkstemp(suffix="_splat_render_before.py")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    atexit.register(lambda: os.path.exists(path) and os.unlink(path))
    spec = importlib.util.spec_from_file_location("splat_render_before", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["splat_render_before"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().mean()


def make_base_inputs(num_strokes: int, target_shape: tuple[int, int]):
    """Deterministic inputs, generated on CPU with a fixed seed."""
    torch.manual_seed(0)
    cubics = torch.rand(1, num_strokes, 4, 2) * 2.0 - 1.0        # [-1, 1]
    stroke_widths = torch.rand(1, num_strokes) * 3.0 + 1.0       # [1, 4] px
    opacities = torch.rand(1, num_strokes) * 0.5 + 0.5           # [0.5, 1]
    target = torch.rand(1, *target_shape)                        # [0, 1]
    return cubics, stroke_widths, opacities, target


def make_leaves(base_tensors, device):
    """Fresh leaf tensors (requires_grad) with identical values per arm."""
    return [t.detach().clone().to(device).requires_grad_(True) for t in base_tensors]


def timed_run(step, params, warmup: int, iters: int, device: torch.device):
    """Run `step` (fwd+bwd) warmup+iters times; return (mean_ms, median_ms, peak_mb)."""
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


def fmt_ms(mean_ms, median_ms):
    return f"{mean_ms:.2f} ({median_ms:.2f})"


def fmt_mb(mb):
    return "n/a" if mb != mb else f"{mb:.1f}"  # NaN check


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quick", action="store_true",
                        help="Tiny validation run: canvas 96, window (24,24,32,32), "
                             "20 timed iters, 60 strokes for scene")
    parser.add_argument("--iters-scale", type=float, default=1.0,
                        help="Multiply timed iteration counts by this factor")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; rerun with --device cpu")
    device = torch.device(args.device)

    if args.quick:
        canvas = 96
        window = (24, 24, 32, 32)       # (y0, x0, h, w)
        greedy_iters = 20
        scene_iters = 20
        warmup = 5
        scene_strokes = 60
    else:
        canvas = 384
        window = (144, 144, 96, 96)
        greedy_iters = 200
        scene_iters = 50
        warmup = 20
        scene_strokes = 600
    greedy_iters = max(1, round(greedy_iters * args.iters_scale))
    scene_iters = max(1, round(scene_iters * args.iters_scale))
    num_samples = 16
    greedy_strokes = 6
    y0, x0, wh, ww = window

    import pydiffvg.splat_render as after_mod
    before_mod = load_before_module()
    before_fn = before_mod.splat_render_cubics
    after_fn = after_mod.splat_render_cubics

    print("# splat_render_cubics before/after speed benchmark")
    print_device_info(device)
    print(f"config: canvas={canvas} num_samples={num_samples} window={window} "
          f"greedy_strokes={greedy_strokes} scene_strokes={scene_strokes} "
          f"warmup={warmup} greedy_iters={greedy_iters} scene_iters={scene_iters} "
          f"quick={args.quick}")
    print()

    # -----------------------------------------------------------------
    # Workload 1: greedy (window optimization)
    # -----------------------------------------------------------------
    g_base = make_base_inputs(greedy_strokes, (wh, ww))
    g_target = g_base[3].to(device)

    def greedy_before_step_factory(params):
        cub, sw, op = params

        def step():
            out = before_fn(cub, sw, canvas_size=canvas, num_samples=num_samples,
                            opacities=op)
            win = out[:, y0:y0 + wh, x0:x0 + ww]
            mse(win, g_target).backward()
        return step

    def greedy_after_step_factory(params):
        cub, sw, op = params

        def step():
            out = after_fn(cub, sw, canvas_size=canvas, num_samples=num_samples,
                           opacities=op, pixel_box=window, use_checkpoint=False)
            mse(out, g_target).backward()
        return step

    # Correctness gate: after's window output must match before's slice.
    with torch.no_grad():
        cub, sw, op = [t.to(device) for t in g_base[:3]]
        before_full = before_fn(cub, sw, canvas_size=canvas,
                                num_samples=num_samples, opacities=op)
        before_slice = before_full[:, y0:y0 + wh, x0:x0 + ww]
        after_win = after_fn(cub, sw, canvas_size=canvas, num_samples=num_samples,
                             opacities=op, pixel_box=window, use_checkpoint=False)
        assert after_win.shape == (1, wh, ww), after_win.shape
        assert torch.allclose(before_slice, after_win, atol=1e-6), (
            f"greedy mismatch: max abs diff "
            f"{(before_slice - after_win).abs().max().item():.3e}")
    print(f"[check] greedy: after pixel_box output matches before slice "
          f"(max abs diff {(before_slice - after_win).abs().max().item():.3e})")

    g_before_params = make_leaves(g_base[:3], device)
    g_before = timed_run(greedy_before_step_factory(g_before_params),
                         g_before_params, warmup, greedy_iters, device)
    g_after_params = make_leaves(g_base[:3], device)
    g_after = timed_run(greedy_after_step_factory(g_after_params),
                        g_after_params, warmup, greedy_iters, device)

    # -----------------------------------------------------------------
    # Workload 2: scene (full frame)
    # -----------------------------------------------------------------
    s_base = make_base_inputs(scene_strokes, (canvas, canvas))
    s_target = s_base[3].to(device)

    def scene_step_factory(fn, params, **extra):
        cub, sw, op = params

        def step():
            out = fn(cub, sw, canvas_size=canvas, num_samples=num_samples,
                     opacities=op, **extra)
            mse(out, s_target).backward()
        return step

    with torch.no_grad():
        cub, sw, op = [t.to(device) for t in s_base[:3]]
        scene_before_out = before_fn(cub, sw, canvas_size=canvas,
                                     num_samples=num_samples, opacities=op)
        scene_after_out = after_fn(cub, sw, canvas_size=canvas,
                                   num_samples=num_samples, opacities=op)
        assert torch.equal(scene_before_out, scene_after_out), (
            f"scene mismatch: max abs diff "
            f"{(scene_before_out - scene_after_out).abs().max().item():.3e}")
    print("[check] scene: after output is torch.equal to before output")
    print()

    s_before_params = make_leaves(s_base[:3], device)
    s_before = timed_run(scene_step_factory(before_fn, s_before_params),
                         s_before_params, warmup, scene_iters, device)
    s_after_params = make_leaves(s_base[:3], device)
    s_after = timed_run(scene_step_factory(after_fn, s_after_params),
                        s_after_params, warmup, scene_iters, device)

    # After-variant: no gradient checkpointing (may OOM at 600x16 gaussians).
    s_nockpt = None
    s_nockpt_err = None
    try:
        s_nockpt_params = make_leaves(s_base[:3], device)
        s_nockpt = timed_run(
            scene_step_factory(after_fn, s_nockpt_params, use_checkpoint=False),
            s_nockpt_params, warmup, scene_iters, device)
    except torch.cuda.OutOfMemoryError as exc:
        s_nockpt_err = f"OOM ({exc})".splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("## Results  (ms/iter = forward+backward, mean (median))")
    print()
    print("| config | before ms/iter | after ms/iter | speedup |")
    print("|---|---|---|---|")
    print(f"| greedy: {greedy_strokes} strokes, {wh}x{ww} window "
          f"(pixel_box + no ckpt) | {fmt_ms(g_before[0], g_before[1])} "
          f"| {fmt_ms(g_after[0], g_after[1])} "
          f"| {g_before[0] / g_after[0]:.2f}x |")
    print(f"| scene: {scene_strokes} strokes, full {canvas}^2 (defaults) "
          f"| {fmt_ms(s_before[0], s_before[1])} "
          f"| {fmt_ms(s_after[0], s_after[1])} "
          f"| {s_before[0] / s_after[0]:.2f}x |")
    if s_nockpt is not None:
        print(f"| scene variant: use_checkpoint=False (vs before defaults) "
              f"| {fmt_ms(s_before[0], s_before[1])} "
              f"| {fmt_ms(s_nockpt[0], s_nockpt[1])} "
              f"| {s_before[0] / s_nockpt[0]:.2f}x |")
    else:
        print(f"| scene variant: use_checkpoint=False (vs before defaults) "
              f"| {fmt_ms(s_before[0], s_before[1])} | {s_nockpt_err} | n/a |")
    print()
    print("| config | before peak GPU MB | after peak GPU MB |")
    print("|---|---|---|")
    print(f"| greedy | {fmt_mb(g_before[2])} | {fmt_mb(g_after[2])} |")
    print(f"| scene | {fmt_mb(s_before[2])} | {fmt_mb(s_after[2])} |")
    nockpt_mem = fmt_mb(s_nockpt[2]) if s_nockpt is not None else (s_nockpt_err or "OOM")
    print(f"| scene no-checkpoint | - | {nockpt_mem} |")


if __name__ == "__main__":
    main()
