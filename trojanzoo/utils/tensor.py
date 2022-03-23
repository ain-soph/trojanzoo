#!/usr/bin/env python3

import torch
import math

__all__ = ['tanh_func', 'atan_func',
           'repeat_to_batch', 'add_noise']


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
              clip_min: float | torch.Tensor = 0.0,
              clip_max: float | torch.Tensor = 1.0) -> torch.Tensor:
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
