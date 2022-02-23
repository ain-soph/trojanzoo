#!/usr/bin/env python3

from trojanzoo.environ import env

import torch
import torchvision.transforms.functional as F
import numpy as np
import math
import os
from PIL import Image
from typing import Any, Union    # TODO: python 3.10

__all__ = ['tanh_func', 'atan_func',
           'to_tensor', 'to_numpy', 'to_list',
           'to_pil_image', 'gray_img', 'gray_tensor',
           'byte2float', 'float2byte',
           'save_as_img', 'read_img_as_tensor',
           'repeat_to_batch', 'add_noise']

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
        x = byte2float(x)
    try:
        x = torch.as_tensor(x, dtype=dtype).to(device=device, **kwargs)
    except Exception:
        print('tensor: ', x)
        if torch.is_tensor(x):
            print('shape: ', x.shape)
            print('device: ', x.device)
        raise
    return x


def to_numpy(x: Any, **kwargs) -> np.ndarray:
    r"""transform a (batched) image to :any:`numpy.ndarray`.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image.
        **kwargs: Keyword arguments passed to :any:`numpy.array`.

    Returns:
        numpy.ndarray:
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)


def to_list(x: Any) -> list:
    r"""transform a (batched) image to :class:`list`.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image.

    Returns:
        list:
    """
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.tolist()
    return list(x)

# ----------------------- Image Utils ------------------------------ #


def to_pil_image(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
                 mode=None) -> Image.Image:
    r"""transform an image to :any:`PIL.Image.Image`.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL.Image.Image: Pillow image instance.
    """
    if isinstance(x, Image.Image):
        return x
    x = to_tensor(x, device='cpu')
    return F.to_pil_image(x, mode=mode)


def gray_img(x: Union[torch.Tensor, np.ndarray, Image.Image],
             num_output_channels: int = 1) -> Image.Image:
    r"""transform an image to :any:`PIL.Image.Image` with gray scale.

    Args:
        x (torch.Tensor | np.ndarray | Image.Image):
            The input image with RGB channels.
        num_output_channels (int): Passed to
            :any:`torchvision.transforms.functional.to_grayscale`.
            Defaults to ``1``.

    Returns:
        PIL.Image.Image: Gray scale image instance.
    """
    if not isinstance(x, Image.Image):
        x = to_pil_image(x)
    return F.to_grayscale(x, num_output_channels=num_output_channels)


def gray_tensor(x: Union[torch.Tensor, np.ndarray, Image.Image],
                num_output_channels: int = 1, **kwargs) -> torch.Tensor:
    r"""transform a (batched) :any:`torch.Tensor`
    with shape ``([N], 3, H, W)`` to gray scale ``([N], 1, H, W)``.

    Args:
        img (torch.Tensor): ``torch.FloatTensor`` ranging in ``[0, 1]``
            with shape ``([N], 3, H, W)``.
        num_output_channels (int): Passed to
            :any:`torchvision.transforms.functional.rgb_to_grayscale`.
            Defaults to ``1``.

    Returns:
        torch.Tensor: Gray scale tensor with shape ``([N], 1, H, W)``.
    """
    img = F.rgb_to_grayscale(x, num_output_channels=num_output_channels)
    return to_tensor(img, **kwargs)


def byte2float(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    r"""transform an image ranging from ``[0, 255]``
    to ``torch.FloatTensor`` ranging in ``[0, 1]``.

    Args:
        img (torch.Tensor | numpy.ndarray | PIL.Image.Image):
            image ranging from ``[0, 255]``.

    Returns:
        torch.Tensor: ``torch.FloatTensor`` ranging in ``[0, 1]``.
    """
    if isinstance(img, torch.Tensor):
        img = to_numpy(img)
    return F.to_tensor(img)

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
    to ``torch.ByteTensor`` ranging from ``[0, 255]``.

    Args:
        img (torch.Tensor): ``torch.FloatTensor`` ranging in ``[0, 1]``.

    Returns:
        torch.Tensor: ``torch.ByteTensor`` ranging from ``[0, 255]``.
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


def tensor_to_img(_tensor: torch.Tensor) -> Image.Image:
    r"""transform a :any:`torch.Tensor` to :any:`PIL.Image.Image`.

    Args:
        path (str): The path to save.
        _tensor (torch.Tensor): The tensor of the image.

    Returns:
        PIL.Image.Image: The image instance.
    """
    if _tensor.dim() == 4:
        assert _tensor.shape[0] == 1
        _tensor = _tensor[0]
    if _tensor.dim() == 3 and _tensor.shape[0] == 1:
        _tensor = _tensor[0]
    if _tensor.dtype in [torch.float, torch.double]:
        _tensor = float2byte(_tensor)
    img = to_numpy(_tensor)
    return Image.fromarray(img)


def save_as_img(path: str, arr: Union[torch.Tensor, np.ndarray]):
    r"""Save a :any:`torch.Tensor` or :any:`numpy.ndarray` as image.

    Args:
        path (str): The path to save.
        arr (torch.Tensor | numpy.ndarray): The tensor of the image.
    """
    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    img = tensor_to_img(to_tensor(arr, device='cpu'))
    img.save(path)


def read_img_as_tensor(path: str) -> torch.Tensor:
    r"""Read image from a local file as :any:`torch.Tensor`.

    Args:
        path (str): The path to an image.
    Returns:
        torch.Tensor: The image tensor in ``[0, 1]``.
    """
    img: Image.Image = Image.open(path)
    return byte2float(img)

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
              mean: float = 0.0, std: float = 1.0, batch: bool = False,
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
        batch (bool): Whether the noise is universal
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
            with shape ``(N, *)`` (``(*)`` when ``batch=True``).
    """
    if noise is None:
        shape = x.shape
        if batch:
            shape = shape[1:]
        noise = torch.normal(mean=mean, std=std,
                             size=shape, device=x.device)
    batch_noise = noise
    if batch:
        batch_noise = repeat_to_batch(noise, x.shape[0])
    noisy_input: torch.Tensor = (
        x + batch_noise).clamp(clip_min, clip_max)
    return noisy_input
