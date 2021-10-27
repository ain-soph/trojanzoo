#!/usr/bin/env python3

from .environ import env

from .output import prints
import torch
import torch.cuda
from typing import Union    # TODO: python 3.10


def empty_cache(threshold: float = None):
    r"""Call :any:`torch.cuda.empty_cache() <torch.cuda.empty_cache>`
    to empty GPU cache when
    :any:`torch.cuda.memory_cached() <torch.cuda.memory_cached>`
    ``>`` :attr:`threshold`\ ``MB``.

    Args:
        threshold (float): The cached memory threshold (MB).
            If ``None``, use ``env['cache_threshold']``.
            Defaults to ``None``.
    """
    threshold = threshold if threshold is not None else env['cache_threshold']
    if threshold is not None and env['num_gpus']:
        if torch.cuda.memory_cached() > threshold * (1 << 20):
            torch.cuda.empty_cache()


def output_memory(device: Union[str, torch.device] = None,
                  full: bool = False, indent: int = 0, **kwargs):
    if full:
        prints(torch.cuda.memory_summary(device=device, **kwargs))
    else:
        prints('memory allocated: '.ljust(20),
               bytes2size(torch.cuda.memory_allocated(device=device)),
               indent=indent)
        prints('memory reserved: '.ljust(20),
               bytes2size(torch.cuda.memory_reserved(device=device)),
               indent=indent)


def bytes2size(_bytes: int) -> str:
    if _bytes < 2 << 10:
        return '%d bytes' % _bytes
    elif _bytes < 2 << 20:
        return '%.3f KB' % (float(_bytes) / (1 << 10))
    elif _bytes < 2 << 30:
        return '%.3f MB' % (float(_bytes) / (1 << 20))
    else:
        return '%.3f GB' % (float(_bytes) / (1 << 30))
