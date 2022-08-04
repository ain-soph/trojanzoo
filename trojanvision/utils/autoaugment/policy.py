#!/usr/bin/env python3

from .operations import PRIMITIVES, get_op

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchvision import transforms

from typing import TYPE_CHECKING
from typing import Sequence
from .operations import Operation
if TYPE_CHECKING:
    pass


class Normalize(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.div(self.temperature).softmax(0)


class MixedOp(nn.Module):
    """Mixed operation"""

    def __init__(self, primitives: list[str] = PRIMITIVES,
                 temperature: float = 0.05, **kwargs):
        super().__init__()
        self._ops: Sequence[Operation] = nn.ModuleList()
        for primitive in primitives:
            op = get_op(primitive, **kwargs)
            self._ops.append(op)
        self.temperature = temperature
        self.weights = nn.Parameter(torch.randn(len(primitives)))
        parametrize.register_parametrization(self, 'weights',
                                             Normalize(temperature))

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        result = torch.stack([op(_input) for op in self._ops])
        weights = self.weights.view([-1] + [1] * (result.dim() - 1))
        return (result * weights).sum(dim=0)

    def create_transform(self):
        return transforms.RandomChoice([op.create_transform()
                                        for op in self._ops],
                                       p=self.weights.tolist())


class SubPolicy(nn.Sequential):
    def __init__(self, operation_count: int = 4, **kwargs):
        super().__init__(*[MixedOp(**kwargs) for _ in range(operation_count)])

    def create_transform(self):
        return transforms.Compose([mixed_op.create_transform() for mixed_op in self])


class Policy(nn.Module):
    def __init__(self, num_sub_policies: int = 100, num_chunks: int = 8,
                 primitives: list[str] = PRIMITIVES, **kwargs):
        super().__init__()
        self.num_sub_policies = num_sub_policies
        self.num_chunks = num_chunks
        self.primitives = primitives

        self.sub_policies: Sequence[SubPolicy] = nn.ModuleList(
            [SubPolicy(primitives=primitives, **kwargs) for _ in range(num_sub_policies)])

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        input_list = _input.chunk(self.num_chunks if self.num_chunks != 0 else len(_input))
        idx_list = torch.randint(self.num_sub_policies, [len(input_list)])
        result = []
        for input_chunk, idx in zip(input_list, idx_list):
            result.append(self.sub_policies[idx](input_chunk))
        return torch.cat(result)

    def create_transform(self):
        return transforms.RandomChoice([sp.create_transform() for sp in self.sub_policies])
