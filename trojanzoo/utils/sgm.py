# -*- coding: utf-8 -*-

from trojanzoo.model import ImageModel

import numpy as np
import torch
import torch.nn as nn


def backward_hook(gamma: float):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, layer: int, gamma: float = 1.0):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if layer in [50, 101, 152]:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and '0.relu' not in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        if len(name.split('.')) == 3:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model: ImageModel, gamma: float = 1.0):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and 'transition' not in name:
            module.register_backward_hook(backward_hook_sgm)


def register_hook(model: ImageModel, gamma: float = 1.0):
    if 'resnet' in model.name:
        register_hook_for_resnet(model, gamma)
    elif 'densenet' in model.name:
        register_hook_for_densenet(model, gamma)
    else:
        raise ValueError(model.name)
