# -*- coding: utf-8 -*-

import os
from PIL import Image
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torchvision

from trojanzoo.config import Config

_map = {'int': torch.int, 'float': torch.float,
        'double': torch.double, 'long': torch.long}

byte2float = torchvision.transforms.ToTensor()


def to_tensor(x, dtype=None, device='default') -> torch.Tensor:
    if x is None:
        return None
    _dtype = _map[dtype] if isinstance(dtype, str) else dtype

    if device == 'default' and Config.env['num_gpus']:
        device = 'cuda'

    if isinstance(x, list):
        try:
            x = torch.stack(x)
        except TypeError:
            pass
    try:
        x = torch.as_tensor(x, dtype=_dtype, device=device)
    except Exception as e:
        print('tensor: ', x)
        raise e
    return x


def to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if type(x).__module__ == np.__name__:
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def to_list(x) -> list:
    if x is None:
        return None
    if type(x).__module__ == np.__name__ or torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, list):
        return x
    else:
        return list(x)


def to_valid_img(img, min=0.0, max=1.0):
    return to_tensor(torch.clamp(img, min, max))


def repeat_to_batch(X, batch_size=1):
    X = to_tensor(X)
    try:
        size = torch.cat(
            (torch.as_tensor([batch_size]).int(), torch.ones(len(X.shape)).int()))
        X = X.repeat(list(size))
    except Exception:
        print('tensor shape: ', X.shape)
        print('batch_size: ', batch_size)
        raise ValueError()
    return X


def add_noise(x, noise=None, mean=0.0, std=1.0, batch=False, detach=True):
    if noise is None:
        shape = x.shape
        if batch:
            shape = shape[1:]
        noise = to_tensor(torch.normal(mean=mean, std=std, size=shape))
    batch_noise = noise
    if batch:
        batch_noise = repeat_to_batch(noise, x.shape[0])
    noisy_input = to_valid_img(x+batch_noise)
    if detach:
        noisy_input = noisy_input.detach()
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
    img = to_tensor(img)
    if len(img.shape) == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    elif len(img.shape) == 3:
        img = img.transpose(0, 1).transpose(1, 2).contiguous()
    img.mul_(255.0)
    # img = (((img - img.min()) / (img.max() - img.min())) * 255.9).astype(np.uint8).squeeze()
    return img.byte()


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


def save_numpy_as_img(path, arr):
    save_tensor_as_img(path, torch.as_tensor(arr))


def read_img_as_tensor(path):
    I: Image.Image = Image.open(path)
    return to_tensor(byte2float(I))
