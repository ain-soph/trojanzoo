#!/usr/bin/env python3

from trojanzoo.environ import env

import torch
import torchvision.transforms.functional as F
import numpy as np
import math
from PIL import Image
from typing import Any, Union    # TODO: python 3.10

__all__ = ['tanh_func', 'atan_func',
           'to_tensor', 'to_list',
           'float2byte', 'repeat_to_batch', 'add_noise']

_map = {'int': torch.int, 'long': torch.long,
        'byte': torch.uint8, 'uint8': torch.uint8,
        'float': torch.float, 'double': torch.double}


def tanh_func(x: torch.Tensor) -> torch.Tensor:
    r"""tanh object function.

    .. code-block:: python

        return x.tanh().add(1).mul(0.5)

    Args:
        x (torch.Tensor): The tensor ranging from
            :math:`[-\infty, +\infty]`.
    Returns:
        torch.Tensor: The tensor ranging in ``[0, 1]``
    """
    return x.tanh().add(1).mul(0.5)


def atan_func(x: torch.Tensor) -> torch.Tensor:
    r"""arctan object function.

    .. code-block:: python

        return x.atan().div(math.pi).add(0.5)

    Args:
        x (torch.Tensor): The tensor ranging from
            :math:`[-\infty], +\infty]`.
    Returns:
        torch.Tensor: The tensor ranging in ``[0, 1]``
    """
    return x.atan().div(math.pi).add(0.5)
# ------------------- Format Transform --------------------------- #


def to_tensor(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
              dtype: Union[str, torch.dtype] = None,
              device: Union[str, torch.device] = 'default',
              **kwargs) -> torch.Tensor:
    r"""transform a (batched) image to :any:`torch.Tensor`.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image.
        dtype (str | torch.dtype): Data type of tensor.
            If :class:`str`, choose from:

                * ``'int'``
                * ``'long'``
                * ``'byte' | 'uint8'``
                * ``'float'``
                * ``'double'``
        device (str | ~torch.torch.device):
            Passed to :any:`torch.as_tensor`.
            If ``'default'``, use ``env['device']``.
        **kwargs: Keyword arguments passed to
            :any:`torch.as_tensor`.

    Returns:
        torch.Tensor:
    """
    if x is None:
        return None
    if isinstance(dtype, str):
        dtype = _map[dtype]

    if device == 'default':
        device = env['device']

    if isinstance(x, (list, tuple)):
        try:
            x = torch.stack(x)
        except TypeError:
            pass
    elif isinstance(x, Image.Image):
        x = F.to_tensor(x)
    try:
        x = torch.as_tensor(x, dtype=dtype).to(device=device, **kwargs)
    except Exception:
        print('tensor: ', x)
        if torch.is_tensor(x):
            print('shape: ', x.shape)
            print('device: ', x.device)
        raise
    return x


def to_list(x: Any) -> list:
    r"""transform a (batched) image to :class:`list`.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image.

    Returns:
        list:
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

# ----------------------- Image Utils ------------------------------ #

# def byte2float(img) -> torch.Tensor:
#     img = to_tensor(img).float()
#     if img.dim() == 2:
#         img.unsqueeze_(dim=0)
#     else:
#         img = img.permute(2, 0, 1).contiguous()
#     img.div_(255.0)
#     return img


def float2byte(img: torch.Tensor) -> torch.Tensor:
    r"""transform a ``torch.FloatTensor`` ranging in ``[0, 1]``
    with shape ``[(1), (C), H, W]``
    to ``torch.ByteTensor`` ranging from ``[0, 255]``
    with shape ``[H, W, (C)]``.

    Args:
        img (torch.Tensor): ``torch.FloatTensor`` ranging in ``[0, 1]``
            with shape ``[(1), (C), H, W]``.

    Returns:
        torch.Tensor: ``torch.ByteTensor`` ranging from ``[0, 255]``
            with shape ``[H, W, (C)]``..
    """
    img = torch.as_tensor(img)
    if img.dim() == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    elif img.dim() == 3:
        img = img.permute(1, 2, 0).contiguous()
    # img = (((img - img.min()) / (img.max() - img.min())) * 255
    #       ).astype(np.uint8).squeeze()
    return img.mul(255).byte()

# --------------------------------------------------------------------- #


def repeat_to_batch(x: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    r"""Repeat a single input tensor to a batch.

    Args:
        x (torch.Tensor): The single input tensor to process
            with shape ``(*)``.
        batch_size (int): Batch size. Defaults to ``1``.
    Returns:
        torch.Tensor:
            The batched input tensor with shape ``(batch_size, *)``
    """
    return x.expand([batch_size] + [-1] * x.dim())


def add_noise(x: torch.Tensor, noise: torch.Tensor = None,
              mean: float = 0.0, std: float = 1.0, universal: bool = False,
              clip_min: Union[float, torch.Tensor] = 0.0,
              clip_max: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    r"""Add noise to a batched input tensor.

    Args:
        x (torch.Tensor): The input tensor to process
            with shape ``(N, *)``.
        noise (torch.Tensor | None): The pre-defined noise.
            If ``None``, generate Gaussian noise
            using :any:`torch.normal`.
            Defaults to be ``None``.
        mean (float): The mean of generated Gaussian noise.
            Defaults to ``0.0``.
        std (float): The std of generated Gaussian noise.
            Defaults to ``1.0``.
        universal (bool): Whether the noise is universal
            for all samples in the batch.
            Defaults to ``False``.
        clip_min (float | torch.Tensor):
            The min value of available input region.
            Defaults to ``0.0``.
        clip_max (float | torch.Tensor):
            The max value of available input region.
            Defaults to ``1.0``.

    Returns:
        torch.Tensor:
            The noisy batched input tensor
            with shape ``(N, *)`` (``(*)`` when ``universal=True``).
    """
    if noise is None:
        shape = x.shape
        if universal:
            shape = shape[1:]
        noise = torch.normal(mean=mean, std=std,
                             size=shape, device=x.device)
    batch_noise = noise
    if universal:
        batch_noise = repeat_to_batch(noise, x.shape[0])
    noisy_input: torch.Tensor = (
        x + batch_noise).clamp(clip_min, clip_max)
    return noisy_input
