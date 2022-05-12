#!/usr/bin/env python3

# https://pytorch.org/functorch/nightly/notebooks/neural_tangent_kernels.html

import torch
import torch.nn as nn
from torch.nn.utils import _stateless

import functools

from typing import Iterable


def empirical_ntk(module: nn.Module, input1: torch.Tensor, input2: torch.Tensor,
                  parameters: dict[str, nn.Parameter] | Iterable[nn.Parameter] = None,
                  compute='full') -> list[torch.Tensor]:
    einsum_expr: str = ''
    match compute:
        case 'full':
            einsum_expr = 'Naf,Mbf->NMab'   # (N, M, C, C)
        case 'trace':
            einsum_expr = 'Naf,Maf->NM'     # (N, M)
        case 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'    # (N, M, C)
        case _:
            raise ValueError(compute)

    if not isinstance(parameters, dict):
        id_map: dict[torch.Tensor, str] = {
            param.data: name for name, param in module.named_parameters()}
        parameters = {id_map[param.data if isinstance(
            param, torch.nn.Parameter) else param]: param
            for param in parameters}
    if parameters is None:
        parameters = dict(module.named_parameters())
    names, values = zip(*parameters.items())

    def func(*params: torch.Tensor, _input: torch.Tensor = None):
        _output: torch.Tensor = _stateless.functional_call(
            module, {n: p for n, p in zip(names, params)}, _input)
        return _output  # (N, C)

    jac1: tuple[torch.Tensor] = torch.autograd.functional.jacobian(
        functools.partial(func, _input=input1), values)
    jac2: tuple[torch.Tensor] = torch.autograd.functional.jacobian(
        functools.partial(func, _input=input2), values)
    jac1 = (j.flatten(2) for j in jac1)  # (N, C, D)
    jac2 = (j.flatten(2) for j in jac2)  # (M, C, D)
    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)]).sum(0)
    return result
