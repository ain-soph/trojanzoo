#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

class Std(nn.Module):
    def forward(self, X: torch.Tensor):
        v, m = torch.var_mean(X, dim=[1, 2, 3], keepdim=True, unbiased=False)
        return (X - m) / torch.sqrt(v + 1e-10)


class StdConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parametrize.register_parametrization(self, "weight", Std())


# class StdConv2d(nn.Conv2d):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         w = self.weight
#         v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
#         w = (w - m) / torch.sqrt(v + 1e-10)
#         return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
#     # TODO: Linting for __call__
