#!/usr/bin/env python3

from collections import OrderedDict
from .operations import get_op
from .genotypes import Genotype

import torch
import torch.nn as nn


class Cell(nn.Module):
    def __init__(self, genotype: Genotype,
                 C_prev_prev: int, C_prev: int, C: int,
                 reduction: bool, reduction_prev: bool,
                 dropout_p: float = None, std_conv: bool = False):
        super().__init__()
        op_name = 'factorized_reduce' if reduction_prev else 'conv'
        self.preprocess0 = get_op(op_name, C_prev_prev, C_out=C, std_conv=std_conv)
        self.preprocess1 = get_op('conv', C_prev, C_out=C, std_conv=std_conv)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        op_names: list[str] = list(op_names)
        indices: list[int] = list(indices)
        concat: list[int]
        self._steps = len(op_names) // 2
        self._indices = indices
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = get_op(name, C, stride, dropout_p=dropout_p, std_conv=std_conv)
            self._ops.append(op)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            states.append(op1(h1) + op2(h2))
        return torch.cat([states[i] for i in self._concat], dim=1)


# stride = 3 if CIFAR10
# stride = 2 if ImageNet
def AuxiliaryHead(C: int, num_classes: int = 10, stride: int = 3) -> nn.Sequential:
    """assuming input size 8x8"""
    return nn.Sequential(OrderedDict([
        ('features', nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True))),
        ('flatten', nn.Flatten()),
        ('classifier', nn.Linear(768, num_classes))
    ]))


class FeatureExtractor(nn.Module):
    # CIFAR10:
    #   C: 36
    #   layer: 20
    #   dropout_p: 0.2
    # ImageNet:
    #   C: 48
    #   layer: 14
    #   dropout_p: None
    def __init__(self, genotype: Genotype, C: int = 36, layers: int = 20,
                 dropout_p: float = 0.2, std_conv: bool = False,
                 stem_multiplier: int = 3, **kwargs):
        super().__init__()
        self.genotype = genotype
        self.aux_C: int = 0
        self.aux_layer: int = 0

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        for i in range(layers):
            reduction = False
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev, dropout_p=dropout_p,
                        std_conv=std_conv)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                self.aux_C = C_prev
                self.aux_layer = i
                self.aux_dim = C_prev
        self.feats_dim = C_prev

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return s1

    def forward_with_aux(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux_feats: torch.Tensor = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_layer:
                aux_feats = s1
        return s1, aux_feats
