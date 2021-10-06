#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.nn.utils import _stateless

from typing import Iterable, Union


def fim_diag(module: nn.Module, _input: torch.Tensor,
             parameters: Iterable[nn.Parameter] = None
             ) -> list[torch.Tensor]:
    if parameters is None:
        parameters = tuple(module.parameters())
    _output = module(_input)  # (N, C)
    with torch.no_grad():
        prob = F.softmax(_output, dim=1).unsqueeze(-1)  # (N, C, 1)
    log_prob = F.log_softmax(_output, dim=1)  # (N, C)
    fim_dict: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(parameters))}
    N, C = log_prob.shape
    for n in range(N):
        for c in range(C):
            grad_list = torch.autograd.grad(log_prob[n][c], parameters, retain_graph=True)
            for i, grad in enumerate(grad_list):    # different layers
                fim = grad.flatten().square()    # (D)
                fim_dict[i].append(fim.detach().clone())
    fim_list: list[torch.Tensor] = []
    for i, value in fim_dict.items():    # different layers
        D = value[0].shape[0]
        fim = torch.stack(value).view(N, C, D) * prob    # (N, C, D)
        fim = fim.sum(1).mean(0)   # (D)
        fim_list.append(fim)
    return fim_list


def fim(module: nn.Module, _input: torch.Tensor,
        parameters: Iterable[nn.Parameter] = None
        ) -> list[torch.Tensor]:
    if parameters is None:
        parameters = tuple(module.parameters())
    _output = module(_input)  # (N, C)
    with torch.no_grad():
        prob = F.softmax(_output, dim=1).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
    log_prob = F.log_softmax(_output, dim=1)  # (N, C)
    fim_dict: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(parameters))}
    N, C = log_prob.shape
    for n in range(N):
        for c in range(C):
            grad_list = torch.autograd.grad(log_prob[n][c], parameters, retain_graph=True)
            for i, grad in enumerate(grad_list):    # different layers
                flatten_grad = grad.flatten()    # (D)
                fim = flatten_grad.unsqueeze(1) * flatten_grad.unsqueeze(0)   # (D, D)
                fim_dict[i].append(fim.detach().clone())
    fim_list: list[torch.Tensor] = []
    for i, value in fim_dict.items():    # different layers
        D = value[0].shape[0]
        fim = torch.stack(value).view(N, C, D, D) * prob    # (N, C, D, D)
        fim_list.append(fim.sum(1).mean(0))   # (D, D)
    return fim_list


# def new_fim(module: nn.Module, _input: torch.Tensor,
#             parameters: Union[dict[str, nn.Parameter], Iterable[nn.Parameter]] = None
#             ) -> list[torch.Tensor]:
#     if not isinstance(parameters, dict):
#         id_map: dict[torch.Tensor, str] = {param.data: name
#                                            for name, param in module.named_parameters()}
#         parameters = {id_map[param.data if isinstance(param, torch.nn.Parameter) else param]: param
#                       for param in parameters}
#     if parameters is None:
#         parameters = dict(module.named_parameters())
#     with torch.no_grad():
#         _output = module(_input)  # (N, C)
#         prob = F.softmax(_output, dim=1).unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
#     keys, values = zip(*parameters.items())

#     def func(*params: torch.Tensor):
#         _output = _stateless.functional_call(module, {n: p for n, p in zip(keys, params)}, _input)
#         return F.log_softmax(_output, dim=1)  # (N, C)
#     jacobian_list: tuple[torch.Tensor] = torch.autograd.functional.jacobian(func, values)

#     fim_list: list[torch.Tensor] = []
#     for jacobian in jacobian_list:  # TODO: parallel
#         jacobian = jacobian.flatten(start_dim=2)   # (N, C, D)
#         fim = prob * jacobian.unsqueeze(-1) * jacobian.unsqueeze(-2)  # (N, C, D, D)
#         fim_list.append(fim.sum(1).mean(0))   # (D, D)
#     return fim_list
