#!/usr/bin/env python3

from trojanzoo.models import Model
import torch
import torch.nn as nn
import numpy as np
from torch.utils.hooks import RemovableHandle
from collections.abc import Callable

__all__ = ['register_hook', 'remove_hook']


def backward_hook(gamma: float) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
    # implement SGM through grad through ReLU
    def _backward_hook(module: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


# def backward_hook_norm(module: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor):
#     # normalize the gradient to avoid gradient explosion or vanish
#     std = torch.std(grad_in[0])
#     return (grad_in[0] / std,)


def register_hook_for_resnet(model: Model, gamma: float = 1.0) -> list[RemovableHandle]:
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if model.layer in [50, 101, 152]:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    _list: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if 'relu' in name and '0.relu' not in name:
            _list.append(module.register_backward_hook(backward_hook_sgm))
    return _list
    # e.g., 1.layer1.1, 1.layer4.2, ...
    # if len(name.split('.')) == 3:
    #     module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model: Model, gamma: float = 1.0):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    _list: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if 'relu' in name and 'transition' not in name:
            _list.append(module.register_backward_hook(backward_hook_sgm))
    return _list


def register_hook(model: Model, gamma: float = 1.0):
    if 'sgm_remove' in model.__dict__.keys():
        print('SGM is already activated when calling register_hook')
        return
    if 'resnet' in model.name:
        model.sgm_remove = register_hook_for_resnet(model, gamma)
    elif 'densenet' in model.name:
        model.sgm_remove = register_hook_for_densenet(model, gamma)
    else:
        raise ValueError(model.name)


def remove_hook(model: Model):
    if 'sgm_remove' not in model.__dict__.keys():
        print('SGM is not activated when calling remove_hook')
        return
    for handle in model.sgm_remove:
        handle: RemovableHandle
        handle.remove()
    del model.sgm_remove
