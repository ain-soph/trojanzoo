#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanvision.utils.model import Conv2d_SAME
from trojanzoo.utils import to_tensor
from trojanzoo.models import Model
from trojanvision.datasets import ImageSet

import torch
import torch.nn as nn
from typing import Tuple

# Note that MagNet requires "eval" mode to train.


class _MagNet(nn.Module):
    """docstring for Model"""

    def __init__(self, structure: list[Tuple[int, str]] = [3, 'average', 3],
                 activation: str = 'sigmoid', channel: int = 3, **kwargs):
        super(_MagNet, self).__init__()

        activation_fn = nn.ReLU()
        if activation == 'sigmoid':
            activation_fn = nn.Sigmoid()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        in_channels = channel
        for i, layer in enumerate(structure):
            if isinstance(layer, int):
                conv = Conv2d_SAME(in_channels=in_channels, out_channels=structure[i], kernel_size=(3, 3))
                in_channels = structure[i]
                bn = nn.BatchNorm2d(structure[i])
                self.encoder.add_module(f'conv{i+1:d}', conv)
                self.encoder.add_module(f'bn{i+1:d}', bn)
                self.encoder.add_module(f'{activation}{i+1:d}', activation_fn)
            else:
                assert isinstance(layer, str)
                module = nn.MaxPool2d(kernel_size=(2, 2)) if layer == 'max' else nn.AvgPool2d(kernel_size=(2, 2))
                self.encoder.add_module('pool', module)

        for i, layer in enumerate(reversed(structure)):
            if isinstance(layer, int):
                conv = Conv2d_SAME(in_channels=in_channels, out_channels=structure[i], kernel_size=(3, 3))
                in_channels = structure[i]
                bn = nn.BatchNorm2d(structure[i])
                self.decoder.add_module(f'conv{i+1:d}', conv)
                self.decoder.add_module(f'bn{i+1:d}', bn)
                self.decoder.add_module(f'{activation}{i+1:d}', activation_fn)
            else:
                assert isinstance(layer, str)
                self.decoder.add_module('pool', nn.Upsample(scale_factor=(2, 2)))
        conv = Conv2d_SAME(structure[0], channel, kernel_size=(3, 3))
        bn = nn.BatchNorm2d(channel)
        self.decoder.add_module('conv', conv)
        self.decoder.add_module('bn', bn)
        self.decoder.add_module('activation', activation_fn)

    def forward(self, x, **kwargs):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MagNet(Model):
    def __init__(self, name: str = 'magnet', dataset: ImageSet = None, model_class: type = _MagNet,
                 structure: list = None, activation: str = None, v_noise: float = 0.1, **kwargs):
        self.v_noise: float = v_noise
        if structure is None:
            if dataset.n_dim == 1:
                structure = [3, "average", 3]
            else:
                structure = [32]
        if activation is None:
            if dataset.n_dim == 1:
                activation = 'sigmoid'
            else:
                activation = 'relu'
        super().__init__(name=name, dataset=dataset, model_class=model_class,
                         structure=structure, activation=activation, channel=dataset.n_channel, **kwargs)

    def get_data(self, data: tuple[torch.Tensor], v_noise: float = None, mode='train'):
        if v_noise is None:
            v_noise = self.v_noise
        _input = data[0]
        if mode == 'train':
            # future warning: to_tensor, to_valid_img
            noise: torch.Tensor = torch.normal(mean=0.0, std=v_noise, size=_input.shape)
            data[0] = (_input + noise).clamp(0.0, 1.0)
            data[1] = _input.detach()
        else:
            data[0] = _input.detach()
            data[1] = _input.clone().detach()
        return to_tensor(data[0]), to_tensor(data[1])

    # Define the optimizer
    # lr: (default: )
    # return: optimizer
    def define_optimizer(self, lr: float = 0.1, parameters: str = 'full',
                         OptimType='Adam', weight_decay=1e-9,
                         lr_scheduler=True, lr_step_size=30, **kwargs):
        return super().define_optimizer(lr=lr, parameters=parameters,
                                        OptimType=OptimType, weight_decay=weight_decay,
                                        lr_scheduler=lr_scheduler, lr_step_size=lr_step_size, **kwargs)

    # define MSE loss function
    def define_criterion(self, **kwargs):
        entropy_fn = nn.MSELoss()

        def loss_fn(_output: torch.Tensor, _label: torch.Tensor):
            _output = _output.to(device=_label.device, dtype=_label.dtype)
            return entropy_fn(_output, _label)
        return loss_fn

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor, topk=(1, 5)):
        res = []
        for k in topk:
            res.append(-self.criterion(_output, _label))
        return res
