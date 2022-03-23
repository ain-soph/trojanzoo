#!/usr/bin/env python3

from trojanzoo.environ import env
from trojanzoo.models import Model
from trojanvision.datasets import ImageSet
from trojanvision.utils.model import Conv2d_SAME

import torch
import torch.nn as nn

from typing import Iterator
from collections.abc import Callable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class _MagNet(nn.Module):
    """docstring for Model"""

    def __init__(self, structure: list[tuple[int, str]] = [3, 'average', 3],
                 activation: str = 'sigmoid', channel: int = 3, **kwargs):
        super().__init__()

        activation_fn = nn.ReLU()
        if activation == 'sigmoid':
            activation_fn = nn.Sigmoid()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        in_channels = channel
        for i, layer in enumerate(structure):
            match layer:
                case int():
                    conv = Conv2d_SAME(in_channels=in_channels,
                                       out_channels=structure[i],
                                       kernel_size=(3, 3))
                    in_channels = structure[i]
                    bn = nn.BatchNorm2d(structure[i])
                    self.encoder.add_module(f'conv{i+1:d}', conv)
                    self.encoder.add_module(f'bn{i+1:d}', bn)
                    self.encoder.add_module(f'{activation}{i+1:d}', activation_fn)
                case str():
                    module = nn.MaxPool2d(kernel_size=(2, 2)) if layer == 'max' \
                        else nn.AvgPool2d(kernel_size=(2, 2))
                    self.encoder.add_module('pool', module)
                case _:
                    raise TypeError(type(layer))

        for i, layer in enumerate(reversed(structure)):
            match layer:
                case int():
                    conv = Conv2d_SAME(in_channels=in_channels,
                                       out_channels=structure[i],
                                       kernel_size=(3, 3))
                    in_channels = structure[i]
                    bn = nn.BatchNorm2d(structure[i])
                    self.decoder.add_module(f'conv{i+1:d}', conv)
                    self.decoder.add_module(f'bn{i+1:d}', bn)
                    self.decoder.add_module(f'{activation}{i+1:d}', activation_fn)
                case str():
                    self.decoder.add_module('pool',
                                            nn.Upsample(scale_factor=(2, 2)))
                case _:
                    raise TypeError(type(layer))
        conv = Conv2d_SAME(structure[0], channel, kernel_size=(3, 3))
        bn = nn.BatchNorm2d(channel)
        self.decoder.add_module('conv', conv)
        self.decoder.add_module('bn', bn)
        self.decoder.add_module('activation', activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MagNet(Model):
    r"""MagNet proposed by Dongyu Meng from Shanghai Tech University in CCS 2017.
    It is an autoencoder for input images to defend against adversarial attacks.

    :Available model names:

        .. code-block:: python3

            ['magnet']

    See Also:
        * paper: `MagNet\: a Two-Pronged Defense against Adversarial Examples`_

    Args:
        structure (list[int | str]): The MagNet model structure.
            Defaults to

            * 1-channel images: ``[3, 'average', 3]`` (e.g, MNIST)
            * 3-channel images: ``[32]``
        activation (str): The activation layer in MagNet model.
            Choose from ``['sigmoid', 'relu']``.
            Defaults to ``'sigmoid'`` for 1-channel images (e.g, MNIST)
            and ``'relu'`` for 3-channel images.
        v_noise (float): The std of random Gaussian noise added to training data.
            Defaults to ``0.1``.

    .. _MagNet\: a Two-Pronged Defense against Adversarial Examples:
        https://arxiv.org/abs/1705.09064
    """
    available_models = ['magnet']

    def __init__(self, name: str = 'magnet',
                 dataset: ImageSet = None, model: type = _MagNet,
                 structure: list = None, activation: str = None,
                 v_noise: float = 0.1, **kwargs):
        self.v_noise: float = v_noise
        if structure is None:
            if dataset.data_shape[0] == 1:
                structure = [3, 'average', 3]
            else:
                structure = [32]
        if activation is None:
            if dataset.data_shape[0] == 1:
                activation = 'sigmoid'
            else:
                activation = 'relu'
        super().__init__(name=name, dataset=dataset, model=model,
                         structure=structure, activation=activation,
                         channel=dataset.data_shape[0], **kwargs)

    def get_data(self, data: tuple[torch.Tensor], v_noise: float = None,
                 mode='train') -> tuple[torch.Tensor, torch.Tensor]:
        _input = data[0]
        if mode == 'train':
            v_noise = v_noise if v_noise is not None else self.v_noise
            noise: torch.Tensor = v_noise * torch.rand_like(_input)
            data[0] = (_input + noise).clamp(0.0, 1.0)
            data[1] = _input.detach()
        else:
            data[0] = _input.detach()
            data[1] = _input.clone().detach()
        return data[0].to(device=env['device']), data[1].to(device=env['device'])

    def define_optimizer(
            self, parameters: str | Iterator[nn.Parameter] = 'full',
            OptimType: str | type[Optimizer] = 'Adam',
            lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 1e-9,
            lr_scheduler: bool = True,
            lr_scheduler_type: str = 'CosineAnnealingLR',
            lr_step_size: int = 30, lr_gamma: float = 0.1,
            epochs: int = None, lr_min: float = 0.0,
            lr_warmup_epochs: int = 0, lr_warmup_method: str = 'constant',
            lr_warmup_decay: float = 0.01,
            **kwargs) -> tuple[Optimizer, _LRScheduler]:
        return super().define_optimizer(
            parameters=parameters, OptimType=OptimType,
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            lr_scheduler=lr_scheduler, lr_scheduler_type=lr_scheduler_type,
            lr_step_size=lr_step_size, lr_gamma=lr_gamma,
            epochs=epochs, lr_min=lr_min,
            lr_warmup_epochs=lr_warmup_epochs,
            lr_warmup_method=lr_warmup_method,
            lr_warmup_decay=lr_warmup_decay,
            **kwargs)

    # define MSE loss function
    def define_criterion(
            self,
            **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        entropy_fn = nn.MSELoss()

        def loss_fn(_output: torch.Tensor, _label: torch.Tensor):
            _output = _output.to(device=_label.device, dtype=_label.dtype)
            return entropy_fn(_output, _label)
        return loss_fn

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor,
                 num_classes: int = None, topk=(1, 5)):
        res = []
        for k in topk:
            res.append(-self.criterion(_output, _label))
        return res
