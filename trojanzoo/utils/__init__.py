# -*- coding: utf-8 -*-

from .tensor import *
from .config import config
from trojanzoo.environ import Env, env
from .output import ansi, prints

import torch
from torch.utils.data import Dataset
from collections import OrderedDict


def summary(indent: int = 0, prefix={'env': Env}, **kwargs):
    od = OrderedDict(prefix)
    od.update(kwargs)
    for key, value in od.items():
        prints('{yellow}{0:<10s}{reset}'.format(key, **ansi), indent=indent)
        try:
            value.summary()
        except Exception:
            prints(value, indent=10)
        prints('-' * 30, indent=indent)
        print()


def empty_cache(threshold: float = None):
    if threshold is None:
        threshold = env['cache_threshold']
    if threshold is not None:
        if torch.cuda.memory_cached() > threshold * (2**20):
            torch.cuda.empty_cache()


class MyDataset(Dataset):
    def __init__(self, data: torch.FloatTensor, targets: torch.LongTensor):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures


def jaccard_idx(mask: torch.FloatTensor, real_mask: torch.FloatTensor, select_num: int = 9) -> float:
    mask = mask.to(dtype=torch.float)
    real_mask = real_mask.to(dtype=torch.float)
    detect_mask = mask > mask.flatten().topk(select_num)[0][-1]
    sum_temp = detect_mask.int() + real_mask.int()
    overlap = (sum_temp == 2).sum().float() / (sum_temp >= 1).sum().float()
    return float(overlap)
