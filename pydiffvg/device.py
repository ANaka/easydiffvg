"""Device management utilities for pydiffvg API compatibility."""

import torch

# Global device state
_use_gpu = torch.cuda.is_available()
_device = torch.device('cuda') if _use_gpu else torch.device('cpu')


def set_use_gpu(v: bool) -> None:
    """Set whether to use GPU.

    Args:
        v: True to use GPU, False to use CPU
    """
    global _use_gpu, _device
    _use_gpu = v
    if not _use_gpu:
        _device = torch.device('cpu')
    elif torch.cuda.is_available():
        _device = torch.device('cuda')


def get_use_gpu() -> bool:
    """Get whether GPU is being used.

    Returns:
        True if using GPU, False otherwise
    """
    return _use_gpu


def set_device(d: torch.device) -> None:
    """Set the device to use.

    Args:
        d: PyTorch device
    """
    global _device, _use_gpu
    _device = d
    _use_gpu = d.type == 'cuda'


def get_device() -> torch.device:
    """Get the current device.

    Returns:
        Current PyTorch device
    """
    return _device
