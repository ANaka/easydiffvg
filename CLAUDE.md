# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pydiffvg is a pure PyTorch reimplementation of diffvg (differentiable vector graphics rasterizer) designed as a drop-in replacement for the original. The goal is `pip install pydiffvg` just works - no C++ compilation, no CUDA toolkit required.

The `.original_diffvg/` directory contains the original diffvg repository as a reference implementation. The new pure PyTorch implementation lives in `pydiffvg/`.

## Build & Development Commands

```bash
# Install dependencies and project in development mode
uv sync

# Run tests
uv run pytest

# Add new dependencies (always use uv add, never pip)
uv add <package>

# Add dev dependencies
uv add --dev <package>
```

**Important**: Always use `uv` for dependency management. Avoid `pip` entirely.

## Architecture

### Original diffvg (reference in `.original_diffvg/`)

The original diffvg has two main components:

1. **C++/CUDA core** (`.original_diffvg/*.cpp`, `.original_diffvg/*.h`): Low-level rasterization with CUDA support

2. **Python bindings** (`.original_diffvg/pydiffvg/`):
   - `shape.py`: Python shape classes (Circle, Ellipse, Path, Polygon, Rect, ShapeGroup)
   - `render_pytorch.py`: PyTorch autograd function (`RenderFunction`) that wraps the C++ renderer
   - `parse_svg.py`, `save_svg.py`: SVG I/O utilities
   - `color.py`: LinearGradient, RadialGradient classes

### New Implementation (`pydiffvg/`)

Pure PyTorch implementation matching the original API:

- `shapes.py`: Shape primitives (Circle, Ellipse, Path, Polygon, Rect)
- `groups.py`: ShapeGroup with fill/stroke/transform
- `color.py`: SolidColor, LinearGradient, RadialGradient
- `render.py`: RenderFunction (torch.autograd.Function)
- `rasterize.py`: Core rasterization logic
- `gradients.py`: Boundary sampling for backward pass
- `bvh.py`: Bounding volume hierarchy for acceleration
- `svg/`: SVG parsing and saving
- `utils/`: Bezier math, winding number, distance fields

### Key Concepts

- **Shapes**: Primitives like Circle, Ellipse, Path (bezier curves), Polygon, Rect
- **ShapeGroups**: Groups shapes together with fill/stroke colors and transformations
- **RenderFunction**: PyTorch autograd.Function that enables gradient computation through rasterization
- **Backward pass**: Uses boundary sampling (Reynolds transport theorem) for gradients at shape edges

## Python Version

Python 3.12+ (see `.python-version`)
