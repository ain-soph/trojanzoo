#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --lr 0.01 --mark_height 32 --mark_width 32 --attack invisible_poison --train_generator_epochs 10 --epochs 10 --mark_path tag_white.png
"""  # noqa: E501

from ...abstract import CleanLabelBackdoor
import trojanvision.models
from trojanzoo.environ import env
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from collections import OrderedDict
import os

import argparse


# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            # Alternative:
            # torchvision.ops.Conv2dNormActivation(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.conv_block(x)
        return out


class InvisiblePoison(CleanLabelBackdoor):
    r"""Invisible Poison Backdoor Attack proposed by Rui Ning
    from Old Dominion University in INFOCOM 2021.

    Based on :class:`trojanvision.attacks.CleanLabelBackdoor`,
    InvisiblePoison preprocesses the trigger by a generator (auto-encoder)
    to amplify its feature activation and make it invisible.

    See Also:
        * paper: `Invisible Poison\: A Blackbox Clean Label Backdoor Attack to Deep Neural Networks`_
        * code: https://github.com/rigley007/Invi_Poison

    Args:
        generator_mode (str): Choose from ``['default', 'resnet_comp', 'resnet']``
            Defaults to ``'default'``.
        noise_coeff (float): Minify rate of adversarial features.
            Defaults to ``0.35``.
        train_generator_epochs (int): Epochs of training generator (auto-encoder).
            Defaults to ``10``.

    .. _Invisible Poison\: A Blackbox Clean Label Backdoor Attack to Deep Neural Networks:
        https://ieeexplore.ieee.org/document/9488902
    """  # noqa: E501
    # noise_alpha (float): Weight of noise in :meth:`add_mark()`.
    #     Defaults to ``0.9``.
    name: str = 'invisible_poison'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--generator_mode',
                           help='(default: "default")')
        group.add_argument('--noise_coeff', type=float,
                           help='(default: 0.35)')
        group.add_argument('--noise_alpha', type=float,
                           help='weight of noise in add_mark '
                           '(default: 0.9)')
        group.add_argument('--train_generator_epochs', type=int,
                           help='epochs of training generator (auto-encoder) '
                           '(default: 10)')
        return group

    def __init__(self, generator_mode: str = 'default', noise_coeff: float = 0.35,
                 train_generator_epochs: int = 800, **kwargs):
        # noise_alpha: float = 0.9,
        super().__init__(**kwargs)
        assert self.mark.mark_height == self.dataset.data_shape[-2]
        assert self.mark.mark_width == self.dataset.data_shape[-1]
        self.mark.mark[-1] = 1.0

        self.param_list['invisible_poison'] = ['generator_mode', 'noise_coeff', 'train_generator_epochs']
        self.noise_coeff = noise_coeff
        self.train_generator_epochs = train_generator_epochs
        self.generator_mode = generator_mode
        # self.noise_alpha = noise_alpha

        self.generator = self.define_generator(generator_mode).to(device=env['device'])
        self.generator.requires_grad_(False)
        self.generator.eval()

    # def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
    #     return (x + self.noise_alpha * self.mark.mark[:-1].to(device=x.device)).clamp(0, 1)

    def define_generator(self, mode: str = 'resnet') -> nn.Sequential:
        match mode:
            case 'resnet':
                # torchvision.models.feature_extraction.create_feature_extractor(
                #     torchvision.models.resnet18(pretrained=True),
                #     ['layer1'])
                resnet_model = torchvision.models.resnet18(pretrained=True)
                encoder = nn.Sequential(*list(resnet_model.children())[:5])
                bottleneck = nn.Sequential(
                    ResNetBlock(64),
                    ResNetBlock(64),
                    ResNetBlock(64))
                decoder = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
                    nn.Tanh())
            case 'resnet_comp':
                resnet_model = trojanvision.models.create('resnet18_comp',
                                                          dataset=self.dataset,
                                                          pretrained=True)
                encoder = nn.Sequential(*list(resnet_model._model.features.cpu().children())[:4])
                bottleneck = nn.Sequential(
                    ResNetBlock(64),
                    ResNetBlock(64),
                    ResNetBlock(64))
                decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Tanh())
            case 'default':
                encoder = nn.Sequential(
                    # MNIST:1*28*28
                    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True),
                    nn.InstanceNorm2d(8), nn.ReLU(),
                    # 8*26*26
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
                    nn.InstanceNorm2d(16), nn.ReLU(),
                    # 16*12*12
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
                    nn.InstanceNorm2d(32), nn.ReLU(),
                    # 32*5*5
                )
                bottleneck = nn.Sequential(
                    ResNetBlock(32),
                    ResNetBlock(32),
                    ResNetBlock(32),
                    ResNetBlock(32))
                decoder = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(),
                    # state size. 16 x 11 x 11
                    nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                    nn.InstanceNorm2d(8),
                    nn.ReLU(),
                    # state size. 8 x 23 x 23
                    nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=0, bias=False),
                    nn.Tanh()
                    # state size. image_nc x 28 x 28
                )
            case _:
                raise NotImplementedError(f'{self.generator_mode=}')
        return nn.Sequential(OrderedDict([
            ('encoder', encoder),
            ('bottleneck', bottleneck),
            ('decoder', decoder),
        ]))

    def attack(self, epochs: int, save: bool = False, **kwargs):
        # train generator
        resnet_model = trojanvision.models.create('resnet18_comp',
                                                  dataset=self.dataset,
                                                  pretrained=True)
        model_extractor = nn.Sequential(*list(resnet_model._model.features.children())[:4])
        match self.generator_mode:
            case 'resnet':
                resnet_model = torchvision.models.resnet18(pretrained=True).to(device=env['device'])
                model_extractor = nn.Sequential(*list(resnet_model.children())[:5])
            case _:
                resnet_model = trojanvision.models.create('resnet18_comp',
                                                          dataset=self.dataset,
                                                          pretrained=True)
                model_extractor = nn.Sequential(*list(resnet_model._model.features.children())[:4])
        model_extractor.requires_grad_(False)
        model_extractor.train()

        self.generator.train()
        if self.generator_mode == 'default':
            self.generator.requires_grad_()
            parameters = self.generator.parameters()
        else:
            self.generator.bottleneck.requires_grad_()
            self.generator.decoder.requires_grad_()
            parameters = self.generator[1:].parameters()
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_generator_epochs,
            eta_min=1e-5)

        logger = MetricLogger()
        logger.create_meters(loss=None, acc=None)
        for _epoch in range(self.train_generator_epochs):
            _epoch += 1
            logger.reset()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, self.train_generator_epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader = logger.log_every(self.dataset.loader['train'], header=header,
                                      tqdm_header='Batch')
            for data in loader:
                optimizer.zero_grad()
                _input, _label = self.model.get_data(data)
                adv_input = (self.generator(_input) + 1) / 2

                _feats = model_extractor(_input)
                adv_feats = model_extractor(adv_input)

                loss = F.l1_loss(_feats, self.noise_coeff * adv_feats)
                loss.backward()
                optimizer.step()
                batch_size = len(_label)
                with torch.no_grad():
                    org_class = self.model.get_class(_input)
                    adv_class = self.model.get_class(adv_input)
                    acc = (org_class == adv_class).float().sum().item() * 100.0 / batch_size
                logger.update(n=batch_size, loss=loss.item(), acc=acc)
            lr_scheduler.step()
        self.save_generator()
        optimizer.zero_grad()
        self.generator.eval()
        self.generator.requires_grad_(False)

        self.mark.mark[:-1] = (self.generator(self.mark.mark[:-1].unsqueeze(0))[0] + 1) / 2
        self.poison_set = self.get_poison_dataset(load_mark=False)
        return super().attack(epochs, save=save, **kwargs)

    def get_filename(self, **kwargs) -> str:
        return super().get_filename(**kwargs) + f'_{self.generator_mode}'

    def save_generator(self, filename: str = None, **kwargs):
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self.generator.state_dict(), file_path + '_generator.pth')
        print('generator results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.generator.load_state_dict(torch.load(file_path + '_generator.pth'))
        return super().load(filename, **kwargs)
