# -*- coding: utf-8 -*-

from .tensor import *
from trojanzoo.config import Config


def empty_cache(threshold=None):
    if threshold is None:
        threshold = Config.env['cache_threshold']
    if threshold is not None:
        if torch.cuda.memory_cached() > threshold*(2**20):
            torch.cuda.empty_cache()
