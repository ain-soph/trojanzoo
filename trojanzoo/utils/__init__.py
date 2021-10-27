#!/usr/bin/env python3

import torch.cuda

from .tensor import *
from .others import *
from .environ import env
from .param import *


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
