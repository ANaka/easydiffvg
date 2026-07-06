"""Report-only experiments for pydiffvg.splat_render.splat_render_cubics.

No library changes; everything is measured against the installed module.

(a) torch.compile: compile the splat kernel and compare eager vs compiled
    (cold-start first-call time, steady-state ms/iter, forward/grad deviation).
    `_splat_chunk` is module-level in this build, so the script compiles it
    and monkeypatches it into the module (the expected path, stated in the
    output). If it were ever nested again, the script falls back to compiling
    the whole `splat_render_cubics` function.
    Note: pixel chunk sizes vary across calls (the last chunk is smaller),
    which may trigger dynamo recompiles; recompile counters are reported
    when available.

(b) bf16 autocast: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    around forward+loss (backward outside autocast), vs plain fp32.
    Skipped with a message on CPU.

Workloads:
    small = "greedy" shape: 6 strokes x 16 samples, pixel_box window,
            use_checkpoint=False.
    large = 625 strokes x 16 samples = 10000 gaussians, full 384^2 frame,
            default kwargs (gradient checkpointing on).

Usage:
    uv run python benchmarks/bench_splat_experiments.py [--device cuda|cpu]
        [--quick]
"""

import argparse
import statistics
import subprocess
import sys
import time

import torch

import pydiffvg.splat_render as sr


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
    return [t.detach().clone().to(device).requires_grad_(True) for t in base_tensors]


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Workload:
    """A (fwd+bwd) benchmark step around splat_render_cubics-like callables."""

    def __init__(self, name, num_strokes, canvas, num_samples, window, device,
                 warmup, iters):
        self.name = name
        self.canvas = canvas
        self.num_samples = num_samples
        self.window = window  # None => full frame, defaults
        self.device = device
        self.warmup = warmup
        self.iters = iters
        tgt_shape = (window[2], window[3]) if window else (canvas, canvas)
        base = make_base_inputs(num_strokes, tgt_shape)
        self.base_params = base[:3]
        self.target = base[3].to(device)

    def render_kwargs(self):
        kw = dict(canvas_size=self.canvas, num_samples=self.num_samples)
        if self.window is not None:
            kw.update(pixel_box=self.window, use_checkpoint=False)
        return kw

    def make_step(self, fn):
        """Returns (step, params). step() runs fwd+bwd and returns the output."""
        params = make_leaves(self.base_params, self.device)
        cub, sw, op = params
        kw = self.render_kwargs()

        def step():
            for p in params:
                p.grad = None
            out = fn(cub, sw, opacities=op, **kw)
            loss = mse(out, self.target)
            loss.backward()
            return out
        return step, params

    def make_autocast_step(self, fn):
        """bf16 autocast around forward+loss; backward outside (CUDA only)."""
        params = make_leaves(self.base_params, self.device)
        cub, sw, op = params
        kw = self.render_kwargs()

        def step():
            for p in params:
                p.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = fn(cub, sw, opacities=op, **kw)
                loss = mse(out.float(), self.target)
            loss.backward()
            return out
        return step, params


def run_arm(step, workload, measure_cold=False):
    """Time step() over warmup+iters. Returns (cold_s, mean_ms, median_ms)."""
    device = workload.device
    cold_s = None
    warmup = workload.warmup
    if measure_cold:
        sync(device)
        t0 = time.perf_counter()
        step()
        sync(device)
        cold_s = time.perf_counter() - t0
        warmup = max(0, warmup - 1)
    for _ in range(warmup):
        step()
    times_ms = []
    for _ in range(workload.iters):
        sync(device)
        t0 = time.perf_counter()
        step()
        sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return cold_s, statistics.mean(times_ms), statistics.median(times_ms)


def fwd_grads(step, params):
    """One fwd+bwd; returns detached output and grads (cloned immediately)."""
    out = step()
    grads = [p.grad.detach().clone() for p in params]
    return out.detach().clone(), grads


def max_abs_dev(a, b):
    return (a - b).abs().max().item()


def dynamo_counters_snapshot():
    try:
        from torch._dynamo.utils import counters
        return {k: dict(v) for k, v in counters.items() if k in ("stats", "frames")}
    except Exception:  # noqa: BLE001
        return None


def dynamo_reset():
    try:
        torch._dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:  # noqa: BLE001
        pass


def build_compiled(mode):
    """Compiled callable + restore fn + description of how it was built."""
    if hasattr(sr, "_splat_chunk"):
        orig = sr._splat_chunk
        sr._splat_chunk = torch.compile(orig, mode=mode)

        def restore():
            sr._splat_chunk = orig
        return sr.splat_render_cubics, restore, "monkeypatched module-level _splat_chunk"

    compiled = torch.compile(sr.splat_render_cubics, mode=mode)
    return compiled, (lambda: None), (
        "whole-function torch.compile(splat_render_cubics) - _splat_chunk is a "
        "nested closure in this build (not a module attribute), so it cannot "
        "be monkeypatched")


def fmt_dev(x):
    return f"{x:.3e}"


# ---------------------------------------------------------------------------
# Experiment (a): torch.compile
# ---------------------------------------------------------------------------

def compile_experiment(workload: Workload):
    print(f"### torch.compile - workload '{workload.name}'")
    eager_step, _ = workload.make_step(sr.splat_render_cubics)
    _, e_mean, e_median = run_arm(eager_step, workload)
    ref_step, ref_params = workload.make_step(sr.splat_render_cubics)
    ref_out, ref_grads = fwd_grads(ref_step, ref_params)

    rows = [("eager", "-", f"{e_mean:.2f} ({e_median:.2f})", "-", "-")]
    notes = []
    default_ok = False

    for mode in (None, "reduce-overhead"):
        label = f"compiled ({mode or 'default'})"
        if mode == "reduce-overhead" and not default_ok:
            rows.append((label, "-", "skipped (default mode failed)", "-", "-"))
            continue
        dynamo_reset()
        fn, restore, how = build_compiled(mode)
        if mode is None:
            notes.append(f"compile strategy: {how}")
        try:
            step, _ = workload.make_step(fn)
            cold_s, c_mean, c_median = run_arm(step, workload, measure_cold=True)
            dev_step, dev_params = workload.make_step(fn)
            c_out, c_grads = fwd_grads(dev_step, dev_params)
            fwd_dev = max_abs_dev(ref_out, c_out)
            grad_dev = max(max_abs_dev(g0, g1)
                           for g0, g1 in zip(ref_grads, c_grads))
            rows.append((label, f"{cold_s:.2f}",
                         f"{c_mean:.2f} ({c_median:.2f})",
                         fmt_dev(fwd_dev), fmt_dev(grad_dev)))
            stats = dynamo_counters_snapshot()
            if stats:
                notes.append(f"{label} dynamo counters: {stats}")
            if mode is None:
                default_ok = True
        except Exception as exc:  # noqa: BLE001 - report-only experiment
            msg = str(exc).splitlines()[0][:160]
            rows.append((label, "-", f"FAILED: {type(exc).__name__}: {msg}", "-", "-"))
            notes.append(f"{label} failed; full error type: {type(exc).__name__}")
        finally:
            restore()
    dynamo_reset()

    print("| arm | cold-start first-call s | steady ms/iter mean (median) "
          "| max abs fwd dev | max abs grad dev |")
    print("|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()
    print("Notes:")
    print("- chunk sizes vary across calls (last pixel chunk is smaller), which "
          "can trigger dynamo recompiles; compare first-call vs steady-state "
          "times and see counters above.")
    for n in notes:
        print(f"- {n}")
    print()


# ---------------------------------------------------------------------------
# Experiment (b): bf16 autocast
# ---------------------------------------------------------------------------

def bf16_experiment(workload: Workload):
    print(f"### bf16 autocast - workload '{workload.name}'")
    if workload.device.type != "cuda":
        print("Skipped: bf16 autocast experiment requires CUDA "
              "(torch.autocast(device_type='cuda')); running on CPU.")
        print()
        return

    fp32_step, _ = workload.make_step(sr.splat_render_cubics)
    _, f_mean, f_median = run_arm(fp32_step, workload)
    ref_step, ref_params = workload.make_step(sr.splat_render_cubics)
    _, ref_grads = fwd_grads(ref_step, ref_params)

    bf16_step, _ = workload.make_autocast_step(sr.splat_render_cubics)
    _, b_mean, b_median = run_arm(bf16_step, workload)
    dev_step, dev_params = workload.make_autocast_step(sr.splat_render_cubics)
    _, b_grads = fwd_grads(dev_step, dev_params)

    devs = [max_abs_dev(g0, g1) for g0, g1 in zip(ref_grads, b_grads)]
    print("| arm | ms/iter mean (median) | speedup | max abs grad dev "
          "cubics | stroke_widths | opacities |")
    print("|---|---|---|---|---|---|")
    print(f"| fp32 | {f_mean:.2f} ({f_median:.2f}) | 1.00x | - | - | - |")
    print(f"| bf16 autocast | {b_mean:.2f} ({b_median:.2f}) "
          f"| {f_mean / b_mean:.2f}x | {fmt_dev(devs[0])} | {fmt_dev(devs[1])} "
          f"| {fmt_dev(devs[2])} |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        except Exception as exc:  # noqa: BLE001
            print(f"nvidia-smi unavailable: {exc}")
    else:
        print("Device: CPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quick", action="store_true",
                        help="Tiny validation run: canvas 96, window "
                             "(24,24,32,32), few iters, 60 strokes for large")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; rerun with --device cpu")
    device = torch.device(args.device)

    if args.quick:
        canvas = 96
        window = (24, 24, 32, 32)
        large_strokes = 60
        small_warmup, small_iters = 3, 10
        large_warmup, large_iters = 2, 5
    else:
        canvas = 384
        window = (144, 144, 96, 96)
        large_strokes = 625
        small_warmup, small_iters = 10, 50
        large_warmup, large_iters = 3, 10
    num_samples = 16

    small = Workload("small (6 strokes, window, no ckpt)", 6, canvas,
                     num_samples, window, device, small_warmup, small_iters)
    large = Workload(f"large ({large_strokes} strokes, full frame, defaults)",
                     large_strokes, canvas, num_samples, None, device,
                     large_warmup, large_iters)

    print("# splat_render_cubics experiments (torch.compile, bf16 autocast)")
    print_device_info(device)
    print(f"config: canvas={canvas} num_samples={num_samples} window={window} "
          f"large_strokes={large_strokes} quick={args.quick}")
    print()

    compile_experiment(small)
    compile_experiment(large)
    bf16_experiment(small)
    bf16_experiment(large)


if __name__ == "__main__":
    main()
