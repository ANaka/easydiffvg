"""Bounding volume hierarchy for acceleration."""

from dataclasses import dataclass

import torch

from pydiffvg.shapes import Shape, Circle, Ellipse, Rect, Polygon, Path
from pydiffvg.utils.bezier import cubic_bounding_box


@dataclass
class BVHNode:
    """A node in the bounding volume hierarchy."""

    bbox: torch.Tensor  # [4] min_x, min_y, max_x, max_y
    left: int | None  # Child index or None if leaf
    right: int | None  # Child index or None if leaf
    shape_id: int | None  # Only set for leaf nodes


def compute_shape_bbox(shape: Shape) -> torch.Tensor:
    """Compute axis-aligned bounding box for a shape.

    Args:
        shape: Any shape primitive

    Returns:
        Bounding box [4]: (min_x, min_y, max_x, max_y)
    """
    if isinstance(shape, Circle):
        center = shape.center
        r = shape.radius
        return torch.stack(
            [center[0] - r, center[1] - r, center[0] + r, center[1] + r]
        )

    elif isinstance(shape, Ellipse):
        center = shape.center
        rx, ry = shape.radius[0], shape.radius[1]
        return torch.stack(
            [center[0] - rx, center[1] - ry, center[0] + rx, center[1] + ry]
        )

    elif isinstance(shape, Rect):
        return torch.stack(
            [shape.p_min[0], shape.p_min[1], shape.p_max[0], shape.p_max[1]]
        )

    elif isinstance(shape, Polygon):
        points = shape.points
        min_pt = points.min(dim=0).values
        max_pt = points.max(dim=0).values
        return torch.stack([min_pt[0], min_pt[1], max_pt[0], max_pt[1]])

    elif isinstance(shape, Path):
        # For paths, we need to compute bbox over all segments
        points = shape.points
        num_control = shape.num_control_points

        min_x = points[:, 0].min()
        min_y = points[:, 1].min()
        max_x = points[:, 0].max()
        max_y = points[:, 1].max()

        # For bezier curves, control points might not bound the curve
        # We need to check extrema for cubic segments
        idx = 0
        for i, n_ctrl in enumerate(num_control):
            n_ctrl_val = int(n_ctrl.item())
            if n_ctrl_val == 2:  # Cubic bezier
                if idx + 3 < len(points):
                    p0 = points[idx]
                    p1 = points[idx + 1]
                    p2 = points[idx + 2]
                    p3 = points[idx + 3]
                    bbox_min, bbox_max = cubic_bounding_box(p0, p1, p2, p3)
                    min_x = torch.minimum(min_x, bbox_min[0])
                    min_y = torch.minimum(min_y, bbox_min[1])
                    max_x = torch.maximum(max_x, bbox_max[0])
                    max_y = torch.maximum(max_y, bbox_max[1])
            idx += n_ctrl_val + 1

        return torch.stack([min_x, min_y, max_x, max_y])

    else:
        raise ValueError(f"Unknown shape type: {type(shape)}")


def bbox_union(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Compute the union of two bounding boxes."""
    return torch.stack(
        [
            torch.minimum(bbox1[0], bbox2[0]),
            torch.minimum(bbox1[1], bbox2[1]),
            torch.maximum(bbox1[2], bbox2[2]),
            torch.maximum(bbox1[3], bbox2[3]),
        ]
    )


def bbox_area(bbox: torch.Tensor) -> torch.Tensor:
    """Compute the area of a bounding box."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height


def bbox_contains_point(bbox: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Check if a bounding box contains a point.

    Args:
        bbox: Bounding box [4]: (min_x, min_y, max_x, max_y)
        point: Query point [2]

    Returns:
        Boolean tensor
    """
    return (
        (point[0] >= bbox[0])
        & (point[0] <= bbox[2])
        & (point[1] >= bbox[1])
        & (point[1] <= bbox[3])
    )


def build_bvh(shapes: list[Shape]) -> list[BVHNode]:
    """Build a BVH using surface area heuristic.

    Args:
        shapes: List of shapes to organize

    Returns:
        List of BVH nodes (root is at index 0)
    """
    if len(shapes) == 0:
        return []

    # Compute bounding boxes for all shapes
    bboxes = [compute_shape_bbox(shape) for shape in shapes]

    # Build leaf nodes
    nodes: list[BVHNode] = []

    def build_recursive(indices: list[int]) -> int:
        """Recursively build BVH, return index of created node."""
        if len(indices) == 1:
            # Leaf node
            idx = indices[0]
            node = BVHNode(
                bbox=bboxes[idx], left=None, right=None, shape_id=idx
            )
            nodes.append(node)
            return len(nodes) - 1

        # Compute combined bounding box
        combined_bbox = bboxes[indices[0]]
        for idx in indices[1:]:
            combined_bbox = bbox_union(combined_bbox, bboxes[idx])

        # Find best split using SAH (simplified: split at median of largest axis)
        extent = combined_bbox[2:] - combined_bbox[:2]
        axis = 0 if extent[0] > extent[1] else 1

        # Sort by centroid along chosen axis
        def centroid(idx: int) -> float:
            bb = bboxes[idx]
            return float((bb[axis] + bb[axis + 2]) / 2)

        indices_sorted = sorted(indices, key=centroid)

        # Split at median
        mid = len(indices_sorted) // 2
        left_indices = indices_sorted[:mid]
        right_indices = indices_sorted[mid:]

        # Create placeholder for this node
        node_idx = len(nodes)
        nodes.append(BVHNode(bbox=combined_bbox, left=None, right=None, shape_id=None))

        # Build children
        left_idx = build_recursive(left_indices)
        right_idx = build_recursive(right_indices)

        # Update node with child indices
        nodes[node_idx].left = left_idx
        nodes[node_idx].right = right_idx

        return node_idx

    build_recursive(list(range(len(shapes))))
    return nodes


def query_bvh(nodes: list[BVHNode], point: torch.Tensor) -> list[int]:
    """Query BVH for shapes whose bounding boxes contain a point.

    Args:
        nodes: BVH node list
        point: Query point [2]

    Returns:
        List of shape IDs whose bboxes contain the point
    """
    if len(nodes) == 0:
        return []

    result: list[int] = []

    def query_recursive(node_idx: int) -> None:
        node = nodes[node_idx]

        if not bbox_contains_point(node.bbox, point):
            return

        if node.shape_id is not None:
            # Leaf node
            result.append(node.shape_id)
        else:
            # Internal node
            if node.left is not None:
                query_recursive(node.left)
            if node.right is not None:
                query_recursive(node.right)

    query_recursive(0)
    return result
