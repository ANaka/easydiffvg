# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

easydiffvg is a modern reimplementation of diffvg (differentiable vector graphics rasterizer) designed for easy installation as a Python dependency. The goal is to provide the same differentiable rendering capabilities as the original diffvg but with a simpler, pip-installable package.

The `diffvg/` directory contains the original diffvg repository as a reference implementation. The new implementation lives in `src/easydiffvg/`.

## Build & Development Commands

```bash
# Install in development mode (using uv, as indicated by .python-version)
uv pip install -e .

# Install dependencies
uv sync
```

## Architecture

### Original diffvg (reference in `diffvg/`)

The original diffvg has two main components:

1. **C++/CUDA core** (`diffvg/*.cpp`, `diffvg/*.h`): Low-level rasterization with CUDA support
   - `scene.cpp/h`: Scene graph management
   - `shape.cpp/h`: Shape primitives (circles, paths, rects)
   - `diffvg.cpp/h`: Main rendering entry point
   - `color.cpp/h`: Color handling including gradients

2. **Python bindings** (`diffvg/pydiffvg/`):
   - `shape.py`: Python shape classes (Circle, Ellipse, Path, Polygon, Rect, ShapeGroup)
   - `render_pytorch.py`: PyTorch autograd function (`RenderFunction`) that wraps the C++ renderer
   - `parse_svg.py`, `save_svg.py`: SVG I/O utilities
   - `color.py`: LinearGradient, RadialGradient classes

### Key Concepts

- **Shapes**: Primitives like Circle, Ellipse, Path (bezier curves), Polygon, Rect
- **ShapeGroups**: Groups shapes together with fill/stroke colors and transformations
- **RenderFunction**: PyTorch autograd.Function that enables gradient computation through the rasterization process
- **Scene**: Container for all shapes and groups to be rendered

### Rendering Pipeline

1. Shapes and ShapeGroups are serialized into a flat argument list
2. Forward pass: C++ renderer produces an image tensor
3. Backward pass: Gradients flow back to shape parameters (positions, colors, etc.)

## Python Version

Python 3.12+ (see `.python-version`)
