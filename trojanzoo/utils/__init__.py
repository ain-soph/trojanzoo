# -*- coding: utf-8 -*-

from .tensor import *
from .config import Config

import torch


def empty_cache(threshold: float = None):
    if threshold is None:
        threshold = Config.env['cache_threshold']
    if threshold is not None:
        if torch.cuda.memory_cached() > threshold * (2**20):
            torch.cuda.empty_cache()


def jaccard_idx(mask: torch.FloatTensor, real_mask: torch.FloatTensor, select_num: int = 9) -> float:
    detect_mask = mask > mask.flatten().topk(select_num)[0][-1]
    sum_temp = detect_mask.int() + real_mask.int()
    overlap = (sum_temp == 2).sum().float() / (sum_temp >= 1).sum().float()
    return float(overlap)
