#!/usr/bin/env python3

import torch
import random
from typing import Union  # TODO: python 3.10


__all__ = ['Cutout']


class Cutout:
    def __init__(self, length: int, fill_values: Union[float, torch.Tensor] = 0.0):
        self.length = length
        self.fill_values = fill_values

    def __call__(self, img: torch.Tensor):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=torch.bool, device=img.device)
        y = random.randint(0, h)
        x = random.randint(0, w)
        y1 = max(y - self.length // 2, 0)
        y2 = min(y + self.length // 2, h)
        x1 = max(x - self.length // 2, 0)
        x2 = min(x + self.length // 2, w)
        mask[y1: y2, x1: x2] = False
        return (mask * img + ~mask * self.fill_values).detach()
