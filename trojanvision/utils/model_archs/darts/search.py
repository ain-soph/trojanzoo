#!/usr/bin/env python3

from .genotypes import Genotype
from .operations import get_op, PRIMITIVES

import torch
import torch.nn as nn
import numpy as np

from typing import Sequence
from collections.abc import Callable


class MixedOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C: int, stride: int,
                 primitives: list[str] = PRIMITIVES,
                 **kwargs):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = get_op(primitive, C, stride, affine=False, **kwargs)
            self._ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps: int, multiplier: int,
                 C_prev_prev: int, C_prev: int, C: int,
                 reduction: bool, reduction_prev: bool,
                 **kwargs):
        super().__init__()
        self.reduction = reduction

        op_name = 'factorized_reduce' if reduction_prev else 'conv'
        self.preprocess0 = get_op(op_name, C_prev_prev, affine=False, C_out=C)
        self.preprocess1 = get_op('conv', C_prev, affine=False, C_out=C)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, **kwargs)
                self._ops.append(op)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class FeatureExtractor(nn.Module):
    def __init__(self, C: int, layers: int,
                 steps: int = 4, multiplier: int = 4,
                 stem_multiplier: int = 3,
                 primitives: list[str] = PRIMITIVES,
                 **kwargs):
        super().__init__()
        self._C = C
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.primitives = primitives

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        self.cells: Sequence[Cell] = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        for i in range(layers):
            reduction = False
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev, primitives=primitives)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.feats_dim = C_prev

        # k = sum(1 for i in range(self._steps) for n in range(2 + i))
        k = 2 * self._steps + ((self._steps - 1) * self._steps) // 2
        num_ops = len(self.primitives)
        if self._layers != 1:
            self.register_buffer('alphas_normal', 1e-3 * torch.randn(k, num_ops))
        self.register_buffer('alphas_reduce', 1e-3 * torch.randn(k, num_ops))
        self.alphas_normal: torch.Tensor  # = 1e-3 * torch.randn(k, num_ops)
        self.alphas_reduce: torch.Tensor  # = 1e-3 * torch.randn(k, num_ops)
        self.softmax: Callable[[torch.Tensor], torch.Tensor] = nn.Softmax(dim=-1)

    def arch_parameters(self) -> list[torch.Tensor]:
        if self._layers == 1:
            return [self.alphas_reduce]
        return [self.alphas_normal, self.alphas_reduce]

    def named_arch_parameters(self) -> list[tuple[str, torch.Tensor]]:
        if self._layers == 1:
            return [('alphas_reduce', self.alphas_reduce)]
        return [('alphas_normal', self.alphas_normal), ('alphas_reduce', self.alphas_reduce)]

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = self.softmax(self.alphas_reduce if cell.reduction
                                   else self.alphas_normal)
            s0, s1 = s1, cell(s0, s1, weights)
        return s1

    def genotype(self) -> Genotype:
        gene_normal = self._parse(self.softmax(self.alphas_normal).detach().cpu().numpy())
        gene_reduce = self._parse(self.softmax(self.alphas_reduce).detach().cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat,
                            reduce=gene_reduce, reduce_concat=concat)
        return genotype

    def _parse(self, weights: np.ndarray):
        gene: list[tuple[str, int]] = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k]
                           for k in range(len(W[x]))
                           if k != self.primitives.index('none'))
                           )[:2]
            for j in edges:
                k_best = -1
                for k in range(len(W[j])):
                    if (k != self.primitives.index('none') and
                            (k_best == -1 or
                             W[j][k] > W[j][k_best])):
                        k_best = k
                gene.append((self.primitives[k_best], j))
            start = end
            n += 1
        return gene
