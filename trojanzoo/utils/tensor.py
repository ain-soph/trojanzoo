# -*- coding: utf-8 -*-

import os
from PIL import Image
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as functional

from trojanzoo.config import Config
env = Config.env

_map = {'int': torch.int, 'float': torch.float,
        'double': torch.double, 'long': torch.long}

byte2float = functional.to_tensor


def to_tensor(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
              dtype=None, device='default', **kwargs) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(dtype, str):
        dtype = _map[dtype]

    if device == 'default':
        device = env['device']
        if 'non_blocking' not in kwargs.keys():
            kwargs['non_blocking'] = True

    if isinstance(x, list):
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


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if x is None:
        return None
    if type(x).__module__ == np.__name__:
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def to_list(x: Union[torch.Tensor, np.ndarray]) -> list:
    if x is None:
        return None
    if type(x).__module__ == np.__name__ or torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, list):
        return x
    else:
        return list(x)


def to_img(x: Union[torch.Tensor, np.ndarray, list, Image.Image], mode=None) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    x = to_tensor(x, device='cpu')
    return functional.to_pil_image(x, mode=mode)


def repeat_to_batch(x: torch.Tensor, batch_size=1) -> torch.Tensor:
    try:
        size = batch_size + [1]*len(x.shape)
        x = x.repeat(list(size))
    except Exception as e:
        print('tensor shape: ', x.shape)
        print('batch_size: ', batch_size)
        raise e
    return x


def gray_img(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1) -> Image.Image:
    if not isinstance(x, Image.Image):
        x = to_img(x)
    return functional.to_grayscale(x, num_output_channels=num_output_channels)


def gray_tensor(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1, **kwargs) -> torch.Tensor:
    if torch.is_tensor(x):
        if 'dtype' not in kwargs.keys():
            kwargs['dtype'] = x.dtype
        if 'device' not in kwargs.keys():
            kwargs['device'] = x.device
    img = gray_img(x, num_output_channels=num_output_channels)
    return to_tensor(img, **kwargs)


def add_noise(x: torch.Tensor, noise=None, mean=0.0, std=1.0, batch=False):
    if noise is None:
        shape = x.shape
        if batch:
            shape = shape[1:]
        noise = torch.normal(mean=mean, std=std, size=shape, device=x.device)
    batch_noise = noise
    if batch:
        batch_noise = repeat_to_batch(noise, x.shape[0])
    noisy_input = (x+batch_noise).clamp()
    return noisy_input


def arctanh(x, epsilon=1e-7):
    x = x-epsilon*x.sign()
    return torch.log(2/(1-x)-1)/2


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def float2byte(img) -> torch.ByteTensor:
    img = torch.as_tensor(img)
    if len(img.shape) == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    elif len(img.shape) == 3:
        img = img.transpose(0, 1).transpose(1, 2).contiguous()
    # img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8).squeeze()
    return img.mul(255).byte()


# def byte2float(img) -> torch.FloatTensor:
#     img = to_tensor(img).float()
#     if len(img.shape) == 2:
#         img.unsqueeze_(dim=0)
#     else:
#         img = img.transpose(1, 2).transpose(0, 1).contiguous()
#     img.div_(255.0)
#     return img


def save_tensor_as_img(path: str, _tensor: torch.Tensor):
    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    _tensor = _tensor.squeeze()
    img = to_numpy(float2byte(_tensor))
    # image.imsave(path, img)
    I = Image.fromarray(img)
    I.save(path)


def save_numpy_as_img(path: str, arr: np.ndarray):
    save_tensor_as_img(path, torch.as_tensor(arr))


def read_img_as_tensor(path: str) -> torch.Tensor:
    I: Image.Image = Image.open(path)
    return byte2float(I)
