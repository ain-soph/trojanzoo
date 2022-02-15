#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

import numpy as np
from collections.abc import Callable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trojanvision.models import ResNet, DenseNet, ImageModel


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


def register_hook_for_resnet(model: 'ResNet', gamma: float = 1.0) -> list[RemovableHandle]:
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    layer = int(model.name.split('_')[0][6:])
    if layer in [50, 101, 152]:
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


def register_hook_for_densenet(model: 'DenseNet', gamma: float = 1.0):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    _list: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if 'relu' in name and 'transition' not in name:
            _list.append(module.register_backward_hook(backward_hook_sgm))
    return _list


def register_hook(model: 'ImageModel', gamma: float = 1.0):
    if 'sgm_remove' in model.__dict__.keys():
        print('SGM is already activated when calling register_hook')
        return
    if 'resnet' in model.name:
        register_hook_for_resnet(model, gamma)
    elif 'densenet' in model.name:
        register_hook_for_densenet(model, gamma)
    else:
        raise ValueError(model.name)


def remove_hook(model: 'ImageModel'):
    if 'sgm_remove' not in model.__dict__.keys():
        print('SGM is not activated when calling remove_hook')
        return
    for handle in model.sgm_remove:
        handle: RemovableHandle
        handle.remove()
    del model.sgm_remove
