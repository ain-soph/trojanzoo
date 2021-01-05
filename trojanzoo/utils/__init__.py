#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.cuda

from .tensor import *
from .miscellaneous import *
from .environ import env
from .param import *
# from .output import *
# from .process import *


def empty_cache(threshold: float = None):
    threshold = threshold if threshold is not None else env['cache_threshold']
    if threshold is not None and env['num_gpus']:
        if torch.cuda.memory_cached() > threshold * (2**20):
            torch.cuda.empty_cache()
