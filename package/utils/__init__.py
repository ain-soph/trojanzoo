# -*- coding: utf-8 -*-

import sys

import numpy as np

import torch
import torch.nn as nn

import os
from PIL import Image

from typing import Union


_map = {'int': torch.int, 'float': torch.float,
        'double': torch.double, 'long': torch.long}

_cuda = torch.cuda.is_available()


def to_tensor(x, dtype=None, device=None, copy=False) -> torch.Tensor:
    if x is None:
        return None
    _dtype = _map[dtype] if isinstance(dtype, str) else dtype

    if isinstance(x, list):
        try:
            x = torch.stack(x)
        except Exception:
            pass
    try:
        x = torch.as_tensor(x, dtype=_dtype, device=device)
    except Exception:
        print('tensor: ', x)
        raise ValueError()
    if device is None and _cuda and not x.is_cuda:
        x = x.cuda()
    return x


def to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if type(x).__module__ == np.__name__:
        return x
    if torch.is_tensor(x):
        return (x.cpu() if x.is_cuda else x).detach().numpy()
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


def byte2float(img) -> torch.FloatTensor:
    img = to_tensor(img).float()
    if len(img.shape) == 2:
        img.unsqueeze_(dim=0)
    else:
        img = img.transpose(1, 2).transpose(0, 1).contiguous()
    img.div_(255.0)
    return img


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
    I = Image.open(path)
    return byte2float(to_tensor(to_numpy(I)))


def bytes2size(_bytes):
    if _bytes < 2*1024:
        return '%d bytes' % _bytes
    elif _bytes < 2*1024*1024:
        return '%.3f KB' % (float(_bytes)/1024)
    elif _bytes < 2*1024*1024*1024:
        return '%.3f MB' % (float(_bytes)/1024/1024)
    else:
        return '%.3f GB' % (float(_bytes)/1024/1024/1024)


def empty_cache(threshold=4096):
    if torch.cuda.memory_cached() > threshold*(2**20):
        torch.cuda.empty_cache()


class Module(object):

    def __init__(self, *args, **kwargs):
        self.add(*args, **kwargs)

    def add(self, *args, inplace=True, **kwargs):
        for module in args:
            for key in module.keys():
                if inplace or key not in self.keys():
                    self.__setattr__(key, module[key])
        for key in kwargs.keys():
            if inplace or key not in self.keys():
                self.__setattr__(key, kwargs[key])

    def keys(self, *args, **kwargs):
        return self.__dict__.keys(*args, **kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class Param(Module):
    def __init__(self, *args, **kwargs):
        self.default = Module()
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self.default

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            super().__setattr__(name, Module(**value))
        elif isinstance(value, Module):
            super().__setattr__(name, value)
        else:
            raise TypeError()

    def __getitem__(self, key):
        if key not in self.keys():
            key = 'default'
        return super().__getitem__(key)

# class StrToBytes:
#     def __init__(self, fileobj):
#         self.fileobj = fileobj
#     def read(self, size):
#         return self.fileobj.read(size).encode()
#     def readline(self, size=-1):
#         return self.fileobj.readline(size).encode()
