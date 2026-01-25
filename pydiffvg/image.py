"""Image I/O utilities for pydiffvg API compatibility."""

import os
from pathlib import Path

import numpy as np
import torch


def imwrite(
    img: torch.Tensor | np.ndarray,
    filename: str,
    gamma: float = 2.2,
    normalize: bool = False,
) -> None:
    """Write image to file with gamma correction.

    Args:
        img: Image tensor [H, W, C] or [H, W], values in [0, 1]
        filename: Output file path
        gamma: Gamma correction value (default 2.2)
        normalize: Whether to normalize image to [0, 1] range
    """
    # Create directory if needed
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    # Convert to numpy if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # Normalize if requested
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng

    # Clip to valid range
    img = np.clip(img, 0.0, 1.0)

    # Add channel dimension if grayscale
    if img.ndim == 2:
        img = np.expand_dims(img, 2)

    # Apply gamma correction to RGB channels
    img_out = img.copy()
    img_out[:, :, :3] = np.power(img[:, :, :3], 1.0 / gamma)

    # Convert to uint8
    img_uint8 = (img_out * 255).astype(np.uint8)

    # Try PIL first, fall back to imageio
    try:
        from PIL import Image
        if img_uint8.shape[2] == 4:
            pil_img = Image.fromarray(img_uint8, mode='RGBA')
        elif img_uint8.shape[2] == 3:
            pil_img = Image.fromarray(img_uint8, mode='RGB')
        else:
            pil_img = Image.fromarray(img_uint8[:, :, 0], mode='L')
        pil_img.save(filename)
    except ImportError:
        try:
            import imageio
            imageio.imwrite(filename, img_uint8)
        except ImportError:
            raise ImportError(
                "Either PIL (pillow) or imageio is required for imwrite. "
                "Install with: pip install pillow"
            )
