#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class Std(nn.Module):
    def forward(self, X: torch.Tensor):
        v, m = torch.var_mean(X, dim=[1, 2, 3], keepdim=True, unbiased=False)
        return (X - m) / torch.sqrt(v + 1e-10)


class StdConv2d(nn.Conv2d):
    def __init__(self, *args, parametrize: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parametrize = parametrize
        if parametrize:
            P.register_parametrization(self, 'weight', Std())

    def parametrize_(self, parametrize: bool = True):
        if parametrize:
            if not self.parametrize:
                P.register_parametrization(self, 'weight', Std())
        elif self.parametrize:
            P.remove_parametrizations(self, 'weight')
        self.parametrize = parametrize
        return self
