#!/usr/bin/env python3

from .environ import env

import torch
import torchvision.transforms.functional as F
import numpy as np
import math
import os
from PIL import Image
from typing import Any, Union    # TODO: python 3.10

__all__ = ['cos_sim', 'tanh_func', 'atan_func',
           'to_tensor', 'to_numpy', 'to_list',
           'to_pil_image', 'gray_img', 'gray_tensor',
           'byte2float', 'float2byte',
           'save_tensor_as_img', 'save_numpy_as_img', 'read_img_as_tensor',
           'repeat_to_batch', 'add_noise']

_map = {'int': torch.int, 'float': torch.float,
        'double': torch.double, 'long': torch.long}
byte2float = F.to_tensor


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum() / a.norm(p=2) / b.norm(p=2)


def tanh_func(x: torch.Tensor) -> torch.Tensor:
    return x.tanh().add(1).mul(0.5)


def atan_func(x: torch.Tensor) -> torch.Tensor:
    return x.atan().div(math.pi).add(0.5)
# ------------------- Format Transform --------------------------- #


def to_tensor(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
              dtype: Union[str, torch.dtype] = None,
              device: Union[str, torch.device] = 'default',
              **kwargs) -> torch.Tensor:
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
    except Exception as e:
        print('tensor: ', x)
        if torch.is_tensor(x):
            print('shape: ', x.shape)
            print('device: ', x.device)
        raise e
    return x


def to_numpy(x: Any, **kwargs) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, **kwargs)


def to_list(x: Any) -> list:
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.tolist()
    return list(x)

# ----------------------- Image Utils ------------------------------ #


def to_pil_image(x: Union[torch.Tensor, np.ndarray, list, Image.Image], mode=None) -> Image.Image:
    # TODO: Linting for mode
    if isinstance(x, Image.Image):
        return x
    x = to_tensor(x, device='cpu')
    return F.to_pil_image(x, mode=mode)


def gray_img(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1) -> Image.Image:
    if not isinstance(x, Image.Image):
        x = to_pil_image(x)
    return F.to_grayscale(x, num_output_channels=num_output_channels)


def gray_tensor(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1, **kwargs) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if 'device' not in kwargs.keys():
            kwargs['device'] = x.device
    img = gray_img(x, num_output_channels=num_output_channels)
    return to_tensor(img, **kwargs)


def float2byte(img: torch.Tensor) -> torch.Tensor:
    img = torch.as_tensor(img)
    if img.dim() == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    elif img.dim() == 3:
        img = img.permute(1, 2, 0).contiguous()
    # img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8).squeeze()
    return img.mul(255).byte()

# def byte2float(img) -> torch.Tensor:
#     img = to_tensor(img).float()
#     if img.dim() == 2:
#         img.unsqueeze_(dim=0)
#     else:
#         img = img.permute(2, 0, 1).contiguous()
#     img.div_(255.0)
#     return img


def tensor_to_img(_tensor: torch.Tensor) -> Image.Image:
    if _tensor.dim() == 4:
        assert _tensor.shape[0] == 1
        _tensor = _tensor[0]
    if _tensor.dim() == 3 and _tensor.shape[0] == 1:
        _tensor = _tensor[0]
    if _tensor.dtype in [torch.float, torch.double]:
        _tensor = float2byte(_tensor)
    img = to_numpy(_tensor)
    return Image.fromarray(img)


def save_tensor_as_img(path: str, _tensor: torch.Tensor):
    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    img = tensor_to_img(_tensor)
    img.save(path)


def save_numpy_as_img(path: str, arr: np.ndarray):
    save_tensor_as_img(path, torch.as_tensor(arr))


def read_img_as_tensor(path: str) -> torch.Tensor:
    img: Image.Image = Image.open(path)
    return byte2float(img)

# --------------------------------------------------------------------- #


def repeat_to_batch(x: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    return x.expand([batch_size] + [-1] * x.dim())


def add_noise(_input: torch.Tensor, noise: torch.Tensor = None,
              mean: float = 0.0, std: float = 1.0, batch: bool = False,
              clip_min: Union[float, torch.Tensor] = 0.0,
              clip_max: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    if noise is None:
        shape = _input.shape
        if batch:
            shape = shape[1:]
        noise = torch.normal(mean=mean, std=std, size=shape, device=_input.device)
    batch_noise = noise
    if batch:
        batch_noise = repeat_to_batch(noise, _input.shape[0])
    noisy_input: torch.Tensor = (_input + batch_noise).clamp(clip_min, clip_max)
    return noisy_input
