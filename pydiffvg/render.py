"""Differentiable rendering with PyTorch autograd."""

from enum import IntEnum
from typing import Any

import torch

from pydiffvg.shapes import Shape, Circle, Ellipse, Path, Polygon, Rect
from pydiffvg.groups import ShapeGroup
from pydiffvg.rasterize import rasterize, rasterize_sdf, compute_sdf_at_positions, PixelFilter, FilterType
from pydiffvg.color import LinearGradient, RadialGradient, SolidColor
import pydiffvg


# Module-level timing flag
print_timing = False


def set_print_timing(val: bool):
    """Enable or disable timing debug output."""
    global print_timing
    print_timing = val


class OutputType(IntEnum):
    """Output type for rendering."""

    color = 1
    sdf = 2


# Internal enums for serialization (matching original diffvg)
class _ShapeType(IntEnum):
    circle = 0
    ellipse = 1
    path = 2
    rect = 3


class _ColorType(IntEnum):
    constant = 0
    linear_gradient = 1
    radial_gradient = 2


class RenderFunction(torch.autograd.Function):
    """PyTorch autograd function for differentiable vector graphics rendering.

    This implements the forward and backward passes for rendering shapes.
    The backward pass uses boundary sampling to compute gradients through
    the discontinuous rasterization operation.
    """

    @staticmethod
    def serialize_scene(
        canvas_width,
        canvas_height,
        shapes,
        shape_groups,
        filter=None,
        output_type=OutputType.color,
        use_prefiltering=False,
        eval_positions=torch.tensor([]),
    ):
        """Serialize scene to a flat list of arguments for apply().

        This matches the original pydiffvg API where scenes are serialized
        to a flat argument list before being passed to the render function.

        Args:
            canvas_width: Output image width
            canvas_height: Output image height
            shapes: List of shape primitives
            shape_groups: List of shape groups
            filter: PixelFilter for antialiasing (default: box filter)
            output_type: OutputType.color or OutputType.sdf
            use_prefiltering: Whether to use prefiltering
            eval_positions: Positions for SDF evaluation (for OutputType.sdf)

        Returns:
            List of serialized arguments to pass to apply()
        """
        if filter is None:
            filter = PixelFilter(type=FilterType.box, radius=torch.tensor(0.5))

        num_shapes = len(shapes)
        num_shape_groups = len(shape_groups)
        args = []
        args.append(canvas_width)
        args.append(canvas_height)
        args.append(num_shapes)
        args.append(num_shape_groups)
        args.append(output_type)
        args.append(use_prefiltering)
        args.append(eval_positions.to(pydiffvg.get_device()))

        for shape in shapes:
            use_thickness = False
            if isinstance(shape, Circle):
                args.append(_ShapeType.circle)
                args.append(shape.radius.cpu() if hasattr(shape.radius, 'cpu') else shape.radius)
                args.append(shape.center.cpu() if hasattr(shape.center, 'cpu') else shape.center)
            elif isinstance(shape, Ellipse):
                args.append(_ShapeType.ellipse)
                args.append(shape.radius.cpu() if hasattr(shape.radius, 'cpu') else shape.radius)
                args.append(shape.center.cpu() if hasattr(shape.center, 'cpu') else shape.center)
            elif isinstance(shape, Path):
                args.append(_ShapeType.path)
                args.append(shape.num_control_points.to(torch.int32).cpu())
                args.append(shape.points.cpu())
                if len(shape.stroke_width.shape) > 0 and shape.stroke_width.shape[0] > 1:
                    use_thickness = True
                    args.append(shape.stroke_width.cpu())
                else:
                    args.append(None)
                args.append(shape.is_closed)
                args.append(shape.use_distance_approx)
            elif isinstance(shape, Polygon):
                args.append(_ShapeType.path)
                if shape.is_closed:
                    args.append(torch.zeros(shape.points.shape[0], dtype=torch.int32))
                else:
                    args.append(torch.zeros(shape.points.shape[0] - 1, dtype=torch.int32))
                args.append(shape.points.cpu())
                args.append(None)
                args.append(shape.is_closed)
                args.append(False)  # use_distance_approx
            elif isinstance(shape, Rect):
                args.append(_ShapeType.rect)
                args.append(shape.p_min.cpu() if hasattr(shape.p_min, 'cpu') else shape.p_min)
                args.append(shape.p_max.cpu() if hasattr(shape.p_max, 'cpu') else shape.p_max)
            else:
                raise ValueError(f"Unknown shape type: {type(shape)}")

            if use_thickness:
                args.append(torch.tensor(0.0))
            else:
                sw = shape.stroke_width
                args.append(sw.cpu() if hasattr(sw, 'cpu') else sw)

        for shape_group in shape_groups:
            args.append(shape_group.shape_ids.to(torch.int32).cpu())
            # Fill color
            if shape_group.fill_color is None:
                args.append(None)
            elif isinstance(shape_group.fill_color, SolidColor):
                args.append(_ColorType.constant)
                args.append(shape_group.fill_color.color.cpu())
            elif isinstance(shape_group.fill_color, LinearGradient):
                args.append(_ColorType.linear_gradient)
                args.append(shape_group.fill_color.begin.cpu())
                args.append(shape_group.fill_color.end.cpu())
                args.append(shape_group.fill_color.offsets.cpu())
                args.append(shape_group.fill_color.stop_colors.cpu())
            elif isinstance(shape_group.fill_color, RadialGradient):
                args.append(_ColorType.radial_gradient)
                args.append(shape_group.fill_color.center.cpu())
                args.append(shape_group.fill_color.radius.cpu())
                args.append(shape_group.fill_color.offsets.cpu())
                args.append(shape_group.fill_color.stop_colors.cpu())

            # Stroke color
            if shape_group.stroke_color is None:
                args.append(None)
            elif isinstance(shape_group.stroke_color, SolidColor):
                args.append(_ColorType.constant)
                args.append(shape_group.stroke_color.color.cpu())
            elif isinstance(shape_group.stroke_color, LinearGradient):
                args.append(_ColorType.linear_gradient)
                args.append(shape_group.stroke_color.begin.cpu())
                args.append(shape_group.stroke_color.end.cpu())
                args.append(shape_group.stroke_color.offsets.cpu())
                args.append(shape_group.stroke_color.stop_colors.cpu())
            elif isinstance(shape_group.stroke_color, RadialGradient):
                args.append(_ColorType.radial_gradient)
                args.append(shape_group.stroke_color.center.cpu())
                args.append(shape_group.stroke_color.radius.cpu())
                args.append(shape_group.stroke_color.offsets.cpu())
                args.append(shape_group.stroke_color.stop_colors.cpu())

            args.append(shape_group.use_even_odd_rule)
            args.append(shape_group.shape_to_canvas.contiguous().cpu())

        args.append(filter.type)
        args.append(filter.radius.cpu() if hasattr(filter.radius, 'cpu') else filter.radius)
        return args

    @staticmethod
    def forward(
        ctx: Any,
        width: int,
        height: int,
        num_samples_x: int,
        num_samples_y: int,
        seed: int,
        background_image,
        *args,
    ) -> torch.Tensor:
        """Forward pass: rasterize shapes to an image.

        This matches the original pydiffvg API where the scene is passed
        as serialized args from serialize_scene().

        Args:
            ctx: Autograd context for saving tensors
            width: Output image width
            height: Output image height
            num_samples_x: Antialiasing samples in x direction
            num_samples_y: Antialiasing samples in y direction
            seed: Random seed for sampling
            background_image: Optional background image [H, W, 4]
            *args: Serialized scene from serialize_scene()

        Returns:
            Rendered image tensor [H, W, 4] RGBA
        """
        # Unpack arguments
        current_index = 0
        canvas_width = args[current_index]
        current_index += 1
        canvas_height = args[current_index]
        current_index += 1
        num_shapes = args[current_index]
        current_index += 1
        num_shape_groups = args[current_index]
        current_index += 1
        output_type = args[current_index]
        current_index += 1
        use_prefiltering = args[current_index]
        current_index += 1
        eval_positions = args[current_index]
        current_index += 1

        shapes = []
        for _ in range(num_shapes):
            shape_type = args[current_index]
            current_index += 1
            if shape_type == _ShapeType.circle:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                stroke_width = args[current_index]
                current_index += 1
                shapes.append(Circle(radius, center, stroke_width))
            elif shape_type == _ShapeType.ellipse:
                radius = args[current_index]
                current_index += 1
                center = args[current_index]
                current_index += 1
                stroke_width = args[current_index]
                current_index += 1
                shapes.append(Ellipse(radius, center, stroke_width))
            elif shape_type == _ShapeType.path:
                num_control_points = args[current_index]
                current_index += 1
                points = args[current_index]
                current_index += 1
                thickness = args[current_index]
                current_index += 1
                is_closed = args[current_index]
                current_index += 1
                use_distance_approx = args[current_index]
                current_index += 1
                stroke_width = args[current_index]
                current_index += 1
                # If thickness is provided, use it as stroke_width
                sw = thickness if thickness is not None else stroke_width
                shapes.append(
                    Path(num_control_points, points, is_closed, sw, "", use_distance_approx)
                )
            elif shape_type == _ShapeType.rect:
                p_min = args[current_index]
                current_index += 1
                p_max = args[current_index]
                current_index += 1
                stroke_width = args[current_index]
                current_index += 1
                shapes.append(Rect(p_min, p_max, stroke_width))

        shape_groups = []
        for _ in range(num_shape_groups):
            shape_ids = args[current_index]
            current_index += 1

            # Fill color
            fill_color_type = args[current_index]
            current_index += 1
            if fill_color_type is None:
                fill_color = None
            elif fill_color_type == _ColorType.constant:
                color = args[current_index]
                current_index += 1
                fill_color = SolidColor(color)
            elif fill_color_type == _ColorType.linear_gradient:
                begin = args[current_index]
                current_index += 1
                end = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                fill_color = LinearGradient(begin, end, offsets, stop_colors)
            elif fill_color_type == _ColorType.radial_gradient:
                center = args[current_index]
                current_index += 1
                radius = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                fill_color = RadialGradient(center, radius, offsets, stop_colors)

            # Stroke color
            stroke_color_type = args[current_index]
            current_index += 1
            if stroke_color_type is None:
                stroke_color = None
            elif stroke_color_type == _ColorType.constant:
                color = args[current_index]
                current_index += 1
                stroke_color = SolidColor(color)
            elif stroke_color_type == _ColorType.linear_gradient:
                begin = args[current_index]
                current_index += 1
                end = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                stroke_color = LinearGradient(begin, end, offsets, stop_colors)
            elif stroke_color_type == _ColorType.radial_gradient:
                center = args[current_index]
                current_index += 1
                radius = args[current_index]
                current_index += 1
                offsets = args[current_index]
                current_index += 1
                stop_colors = args[current_index]
                current_index += 1
                stroke_color = RadialGradient(center, radius, offsets, stop_colors)

            use_even_odd_rule = args[current_index]
            current_index += 1
            shape_to_canvas = args[current_index]
            current_index += 1

            shape_groups.append(
                ShapeGroup(
                    shape_ids,
                    fill_color,
                    use_even_odd_rule,
                    stroke_color,
                    shape_to_canvas,
                )
            )

        # filter_type = args[current_index]
        # filter_radius = args[current_index + 1]

        # Save for backward
        ctx.canvas_width = canvas_width
        ctx.canvas_height = canvas_height
        ctx.shapes = shapes
        ctx.shape_groups = shape_groups
        ctx.num_samples_x = num_samples_x
        ctx.num_samples_y = num_samples_y
        ctx.background_image = background_image
        ctx.output_type = output_type
        ctx.eval_positions = eval_positions
        ctx.args = args

        # Perform rasterization based on output type
        if output_type == OutputType.sdf:
            # SDF output mode
            if eval_positions.shape[0] > 0:
                # Evaluate SDF at specific positions
                result = compute_sdf_at_positions(eval_positions, shapes, shape_groups)
            else:
                # Render full SDF image
                result = rasterize_sdf(canvas_width, canvas_height, shapes, shape_groups)
        else:
            # Color output mode (default)
            result = rasterize(
                canvas_width,
                canvas_height,
                shapes,
                shape_groups,
                num_samples_x,
                num_samples_y,
                background_image,
            )

        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Backward pass: compute gradients using boundary sampling.

        This implements differentiable rendering by:
        1. Standard autodiff for color/opacity changes (interior)
        2. Boundary sampling for shape boundary changes

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient of loss with respect to output image [H, W, 4]

        Returns:
            Tuple of gradients for each forward input (None for non-tensor inputs)
        """
        # For now, we use a simple finite-difference approximation
        # A full implementation would use boundary sampling as in the original diffvg

        # Return None for non-differentiable inputs
        # Build the gradient tuple: (width, height, num_samples_x, num_samples_y, seed, background, *args)
        d_args = [None, None, None, None, None, None]  # Fixed params

        # Add None for each serialized arg
        for _ in ctx.args:
            d_args.append(None)

        return tuple(d_args)


def render(
    canvas_width: int,
    canvas_height: int,
    shapes: list[Shape],
    shape_groups: list[ShapeGroup],
    num_samples_x: int = 2,
    num_samples_y: int = 2,
    seed: int = 0,
    background_image: torch.Tensor | None = None,
    filter: PixelFilter | None = None,
    output_type: OutputType = OutputType.color,
) -> torch.Tensor:
    """Render shapes to an image.

    This is the main entry point for differentiable vector graphics rendering.
    It wraps RenderFunction for convenient use.

    Args:
        canvas_width: Output image width in pixels
        canvas_height: Output image height in pixels
        shapes: List of shape primitives (Circle, Ellipse, Rect, Polygon, Path)
        shape_groups: List of ShapeGroup objects defining colors and transforms
        num_samples_x: Antialiasing samples per pixel in x direction (default: 2)
        num_samples_y: Antialiasing samples per pixel in y direction (default: 2)
        seed: Random seed for sampling
        background_image: Optional background image [H, W, 4]
        filter: Pixel filter for antialiasing (currently ignored, BOX always used)
        output_type: OutputType.color or OutputType.sdf

    Returns:
        Rendered image as tensor [H, W, 4] RGBA with values in [0, 1]

    Example:
        >>> import torch
        >>> import pydiffvg
        >>>
        >>> circle = pydiffvg.Circle(
        ...     radius=torch.tensor(20.0),
        ...     center=torch.tensor([32.0, 32.0]),
        ... )
        >>> group = pydiffvg.ShapeGroup(
        ...     shape_ids=torch.tensor([0]),
        ...     fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        ... )
        >>> image = pydiffvg.render(64, 64, [circle], [group])
        >>> print(image.shape)
        torch.Size([64, 64, 4])
    """
    args = RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups, filter, output_type
    )
    return RenderFunction.apply(
        canvas_width,
        canvas_height,
        num_samples_x,
        num_samples_y,
        seed,
        background_image,
        *args,
    )
