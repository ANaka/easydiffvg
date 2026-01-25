"""SVG optimization utilities for pydiffvg.

This module provides classes for optimizing SVG shapes through gradient descent.
It's a simplified version of the original diffvg optimize_svg module.
"""

import copy
import json
import math
import xml.etree.ElementTree as etree
from collections import namedtuple
from xml.dom import minidom

import numpy as np
import torch

import pydiffvg


class SvgOptimizationSettings:
    """Settings for SVG optimization.

    Controls which parameters to optimize and their learning rates.
    """

    default_params = {
        "optimize_color": True,
        "color_lr": 2e-3,
        "optimize_alpha": False,
        "alpha_lr": 2e-3,
        "optimizer": "Adam",
        "transforms": {
            "optimize_transforms": True,
            "transform_mode": "rigid",
            "translation_mult": 1e-3,
            "transform_lr": 2e-3,
        },
        "circles": {
            "optimize_center": True,
            "optimize_radius": True,
            "shape_lr": 2e-1,
        },
        "paths": {
            "optimize_points": True,
            "shape_lr": 2e-1,
        },
        "gradients": {
            "optimize_stops": True,
            "stop_lr": 2e-3,
            "optimize_color": True,
            "color_lr": 2e-3,
            "optimize_alpha": False,
            "alpha_lr": 2e-3,
            "optimize_location": True,
            "location_lr": 2e-1,
        },
    }

    optims = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "ASGD": torch.optim.ASGD,
    }

    def __init__(self, f=None):
        """Initialize optimization settings.

        Args:
            f: Optional file handle to load settings from JSON
        """
        self.store = {}
        self.dname = "default"
        if f is None:
            self.store["default"] = copy.deepcopy(SvgOptimizationSettings.default_params)
        else:
            self.store = json.load(f)

    def default_name(self, dname):
        """Set the default name for this settings object."""
        self.dname = dname
        if dname not in self.store:
            self.store[dname] = self.store["default"]

    def retrieve(self, node_id):
        """Retrieve settings for a node.

        Returns:
            Tuple of (settings_dict, is_custom)
        """
        if node_id not in self.store:
            return (self.store["default"], False)
        else:
            return (self.store[node_id], True)

    def reset_to_defaults(self, node_id):
        """Reset a node's settings to defaults."""
        if node_id in self.store:
            del self.store[node_id]
        return self.store["default"]

    def undefault(self, node_id):
        """Create custom settings for a node if not already present."""
        if node_id not in self.store:
            self.store[node_id] = copy.deepcopy(self.store["default"])
        return self.store[node_id]

    def override_optimizer(self, optimizer):
        """Override the optimizer for all settings."""
        if optimizer is not None:
            for v in self.store.values():
                v["optimizer"] = optimizer

    def global_override(self, path, value):
        """Override a setting globally across all nodes.

        Args:
            path: List of keys to navigate to the setting
            value: New value to set
        """
        for store in self.store.values():
            d = store
            for key in path[:-1]:
                d = d[key]
            d[path[-1]] = value

    def save(self, file):
        """Save settings to a JSON file."""
        self.store["default"] = self.store[self.dname]
        json.dump(self.store, file, indent="\t")


class OptimizableSvg:
    """An SVG document that can be optimized through gradient descent.

    This class loads an SVG file and makes its parameters (colors, positions, etc.)
    differentiable for optimization.
    """

    class TransformTools:
        """Utilities for working with 2D affine transforms."""

        TransformDecomposition = namedtuple(
            "TransformDecomposition", "theta scale shear translate"
        )
        TransformProperties = namedtuple(
            "TransformProperties",
            "has_rotation has_scale has_mirror scale_uniform has_shear has_translation",
        )

        @staticmethod
        def parse_matrix(vals):
            """Parse SVG matrix transform values."""
            assert len(vals) == 6
            return np.array(
                [[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]], [0, 0, 1]]
            )

        @staticmethod
        def parse_translate(vals):
            """Parse SVG translate transform values."""
            assert len(vals) >= 1 and len(vals) <= 2
            mat = np.eye(3)
            mat[0, 2] = vals[0]
            if len(vals) > 1:
                mat[1, 2] = vals[1]
            return mat

        @staticmethod
        def parse_rotate(vals):
            """Parse SVG rotate transform values."""
            assert len(vals) == 1 or len(vals) == 3
            mat = np.eye(3)
            rads = math.radians(vals[0])
            sint = math.sin(rads)
            cost = math.cos(rads)
            mat[0:2, 0:2] = np.array([[cost, -sint], [sint, cost]])
            if len(vals) > 1:
                tr1 = OptimizableSvg.TransformTools.parse_translate(vals[1:3])
                tr2 = OptimizableSvg.TransformTools.parse_translate([-vals[1], -vals[2]])
                mat = tr1 @ mat @ tr2
            return mat

        @staticmethod
        def parse_scale(vals):
            """Parse SVG scale transform values."""
            assert len(vals) >= 1 and len(vals) <= 2
            d = np.array([vals[0], vals[1] if len(vals) > 1 else vals[0], 1])
            return np.diag(d)

        @staticmethod
        def decompose(M):
            """Decompose a 2D affine transform matrix.

            Returns:
                TransformDecomposition with theta, scale, shear, translate
            """
            m = M[0:2, 0:2]
            t0 = M[0:2, 2]
            # Get translation so that we can post-multiply with it
            TXY = np.linalg.solve(m, t0)

            q, r = np.linalg.qr(m)
            ref = np.array([[1, 0], [0, np.sign(np.linalg.det(q))]])
            Rot = np.dot(q, ref)
            ref2 = np.array([[1, 0], [0, np.sign(np.linalg.det(r))]])
            r2 = np.dot(ref2, r)
            Ref = np.dot(ref, ref2)

            sc = np.diag(r2)
            Scale = np.diagflat(sc)
            ShearX = r2[0, 1] / sc[0] if sc[0] != 0 else 0

            if np.sum(sc) < 0:
                Rot = np.dot(Rot, -np.eye(2))
                Scale = -Scale

            Theta = math.atan2(Rot[1, 0], Rot[0, 0])
            ScaleXY = np.array([Scale[0, 0], Scale[1, 1] * Ref[1, 1]])

            return OptimizableSvg.TransformTools.TransformDecomposition(
                theta=Theta, scale=ScaleXY, shear=ShearX, translate=TXY
            )

        @staticmethod
        def recompose(Theta, ScaleXY, ShearX, TXY):
            """Recompose a 2D affine transform from components."""
            cost = math.cos(Theta)
            sint = math.sin(Theta)
            Rot = np.array([[cost, -sint], [sint, cost]])
            Scale = np.diag(ScaleXY) if hasattr(ScaleXY, "__len__") else np.diag([ScaleXY, ScaleXY])
            Shear = np.eye(2)
            Shear[0, 1] = ShearX

            Translate = np.eye(3)
            Translate[0:2, 2] = TXY

            M = np.eye(3)
            M[0:2, 0:2] = Rot @ Scale @ Shear
            return M @ Translate

    def __init__(
        self,
        filename,
        settings=None,
        optimize_background=False,
        verbose=False,
        device=torch.device("cpu"),
    ):
        """Load an SVG file for optimization.

        Args:
            filename: Path to SVG file
            settings: SvgOptimizationSettings instance
            optimize_background: Whether to optimize background color
            verbose: Print parsing progress
            device: Torch device to use
        """
        if settings is None:
            settings = SvgOptimizationSettings()

        self.settings = settings
        self.verbose = verbose
        self.device = device
        self.settings.device = device
        self.optimizers = []

        self.background = torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float32,
            requires_grad=optimize_background,
            device=self.device,
        )

        # Parse the SVG file
        self._parse_svg(filename)

        self.dirty = True
        self.scene = None

    def _parse_svg(self, filename):
        """Parse SVG file and extract shapes."""
        # Use pydiffvg's parse_svg for the heavy lifting
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.parse_svg(
            filename, device=self.device
        )
        self.canvas = (canvas_width, canvas_height)
        self.shapes = shapes
        self.shape_groups = shape_groups

    def build_scene(self):
        """Build the scene for rendering.

        Returns:
            Tuple of (canvas_width, canvas_height, shapes, shape_groups)
        """
        if self.dirty:
            self.scene = (
                self.canvas[0],
                self.canvas[1],
                self.shapes,
                self.shape_groups,
            )
            self.dirty = False
        return self.scene

    def render(self, scale=None, seed=0):
        """Render the SVG to an image.

        Args:
            scale: Optional scale factor for output size
            seed: Random seed for sampling

        Returns:
            Rendered image tensor [H, W, 4]
        """
        scene = self.build_scene()
        scene_args = pydiffvg.RenderFunction.serialize_scene(*scene)
        render = pydiffvg.RenderFunction.apply
        out_size = (
            (scene[0], scene[1])
            if scale is None
            else (int(scene[0] * scale), int(scene[1] * scale))
        )
        img = render(
            out_size[0],  # width
            out_size[1],  # height
            2,  # num_samples_x
            2,  # num_samples_y
            seed,  # seed
            None,  # background_image
            *scene_args,
        )
        return img

    def zero_grad(self):
        """Zero gradients for all optimizers."""
        for optim in self.optimizers:
            optim.zero_grad()

    def step(self):
        """Take an optimization step."""
        self.dirty = True
        for optim in self.optimizers:
            optim.step()
