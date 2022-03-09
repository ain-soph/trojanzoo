#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 600 --lr 0.01 --attack input_aware_dynamic
"""  # noqa: E501

from .badnet import BadNet

from trojanzoo.environ import env
from trojanzoo.utils.data import sample_batch
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints

import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

import math
import random

import argparse
from collections.abc import Callable


class InputAwareDynamic(BadNet):
    r"""
    | Input-Aware Dynamic Backdoor Attack proposed by Anh Nguyen and Anh Tran
      from VinAI Research in NIPS 2020.
    |
    | Based on :class:`trojanvision.attacks.BadNet`,
      InputAwareDynamic trains mark generator and mask generator
      to synthesize unique watermark for each input.

    See Also:
        * paper: `Input-Aware Dynamic Backdoor Attack`_
        * code: https://github.com/VinAIResearch/input-aware-backdoor-attack-release

    Args:
        attack_remask_epochs (int): Inner epoch to optimize watermark during each training epoch.
            Defaults to ``20``.
        attack_remask_lr (float): Learning rate of Adam optimizer to optimize watermark.
            Defaults to ``0.1``.

    .. _Input-Aware Dynamic Backdoor Attack:
        https://arxiv.org/abs/2010.08138
    """  # noqa: E501

    name: str = 'input_aware_dynamic'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--train_mask_epochs', type=int)
        return group

    def __init__(self, train_mask_epochs: int = 25,
                 lambda_div: float = 1.0, lambda_norm: float = 100.0,
                 mask_density: float = 0.032,
                 cross_percent: float = 0.1,
                 poison_percent: float = 0.1, **kwargs):
        super().__init__(poison_percent=poison_percent, **kwargs)
        self.param_list['input_aware_dynamic'] = ['train_mask_epochs',
                                                  'lambda_div', 'lambda_norm',
                                                  'mask_density', 'cross_percent']

        self.train_mask_epochs = train_mask_epochs
        self.lambda_div = lambda_div
        self.lambda_norm = lambda_norm
        self.mask_density = mask_density

        self.poison_ratio = self.poison_percent
        self.poison_num = 0  # monkey patch: to avoid batch size change in badnet
        self.cross_percent = cross_percent

        data_channel = self.dataset.data_shape[0]
        num_channels = [16, 32] if data_channel == 1 else [32, 64, 128]
        self.mark_generator = self.define_generator(
            num_channels, in_channel=data_channel
        ).to(device=env['device']).eval()
        self.mask_generator = self.define_generator(
            num_channels, in_channel=data_channel,
            out_channel=1).to(device=env['device']).eval()

        self.train_set = self.dataset.loader['train'].dataset
        self.idx = torch.randperm(len(self.train_set))
        self.pos = 0

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mark = self.get_mark(x)
        mask = self.get_mask(x)
        return x + mask * (mark - x)

    def get_mark(self, _input: torch.Tensor) -> torch.Tensor:
        raw_output: torch.Tensor = self.mark_generator(_input)
        return raw_output.tanh() / 2 + 0.5

    def get_mask(self, _input: torch.Tensor) -> torch.Tensor:
        raw_output: torch.Tensor = self.mask_generator(_input)
        return raw_output.tanh().mul(10).tanh() / 2 + 0.5

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        if not org:
            if keep_org:
                decimal, integer = math.modf(len(_label) * self.poison_percent)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                trigger_input = self.add_mark(_input[:integer])
                _input = torch.cat([trigger_input, _input[integer:]])
                if poison_label:
                    trigger_label = self.target_class * torch.ones_like(_label[:integer])
                    _label = torch.cat([trigger_label, _label[integer:]])
        return _input, _label

    def get_cross_data(self, data: tuple[torch.Tensor, torch.Tensor],
                       **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        batch_size = len(_input)
        data2 = sample_batch(self.train_set, idx=self.idx[self.pos:self.pos + batch_size])
        _input2, _label2 = self.model.get_data(data2)
        self.pos += batch_size
        if self.pos >= len(self.idx):
            self.pos = 0
            self.idx = torch.randperm(len(self.idx))
        mark, mask = self.get_mark(_input2), self.get_mask(_input2)
        _input = _input + mask * (mark - _input)
        return _input, _label

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0,
                    threshold: float = 5.0,
                    **kwargs) -> tuple[float, float]:
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger', main_tag='valid trigger',
                                             get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                             indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate Cross', main_tag='valid cross',
                             get_data_fn=self.get_cross_data, indent=indent, **kwargs)
        prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        prints(f'Neuron Jaccard Idx: {self.get_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > threshold:
            target_acc = 0.0
        return clean_acc, target_acc

    def attack(self, epochs: int, optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               validate_interval: int = 1,
               **kwargs):
        print('train mask generator')
        self.mark_generator.requires_grad_(False)
        self.mask_generator.requires_grad_()
        self.model.requires_grad_(False)
        self.train_mask()
        print()
        print('train mark generator and model')

        self.mark_generator.requires_grad_()
        self.mask_generator.requires_grad_(False)
        params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        self.model.activate_params(params)

        mark_optimizer = torch.optim.Adam(self.mark_generator.parameters(), lr=1e-2, betas=(0.5, 0.9))
        mark_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            mark_optimizer, T_max=epochs)
        loader = self.dataset.loader['train']
        dataset = loader.dataset
        logger = MetricLogger()
        logger.create_meters(loss=None, div=None, ce=None)

        if validate_interval != 0:
            self.validate_fn()
        for _epoch in range(epochs):
            _epoch += 1
            idx = torch.randperm(len(dataset))
            pos = 0
            logger.reset()
            self.model.train()
            self.mark_generator.train()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            for data in logger.log_every(loader, header=header):
                optimizer.zero_grad()
                mark_optimizer.zero_grad()
                _input, _label = self.model.get_data(data)
                batch_size = len(_input)
                data2 = sample_batch(dataset, idx=idx[pos:pos + batch_size])
                _input2, _label2 = self.model.get_data(data2)
                pos += batch_size
                final_input, final_label = _input.clone(), _label.clone()

                # generate trigger input
                trigger_dec, trigger_int = math.modf(len(_label) * self.poison_percent)
                trigger_int = int(trigger_int)
                if random.uniform(0, 1) < trigger_dec:
                    trigger_int += 1
                x = _input[:trigger_int]
                trigger_mark, trigger_mask = self.get_mark(x), self.get_mask(x)
                trigger_input = x + trigger_mask * (trigger_mark - x)
                final_input[:trigger_int] = trigger_input
                final_label[:trigger_int] = self.target_class

                # generate cross input
                cross_dec, cross_int = math.modf(len(_label) * self.cross_percent)
                cross_int = int(cross_int)
                if random.uniform(0, 1) < cross_dec:
                    cross_int += 1
                x = _input[trigger_int:trigger_int + cross_int]
                x2 = _input2[trigger_int:trigger_int + cross_int]
                cross_mark, cross_mask = self.get_mark(x2), self.get_mask(x2)
                cross_input = x + cross_mask * (cross_mark - x)
                final_input[trigger_int:trigger_int + cross_int] = cross_input

                # div loss
                if len(trigger_input) <= len(cross_input):
                    length = len(trigger_input)
                    cross_input = cross_input[:length]
                    cross_mark = cross_mark[:length]
                    cross_mask = cross_mask[:length]
                else:
                    length = len(cross_input)
                    trigger_input = trigger_input[:length]
                    trigger_mark = trigger_mark[:length]
                    trigger_mask = trigger_mask[:length]
                input_dist: torch.Tensor = (trigger_input - cross_input).flatten(1).norm(p=2, dim=1)
                mark_dist: torch.Tensor = (trigger_mark - cross_mark).flatten(1).norm(p=2, dim=1) + 1e-5

                loss_ce = self.model.loss(final_input, final_label)
                loss_div = input_dist.div(mark_dist).mean()

                loss = loss_ce + self.lambda_div * loss_div
                loss.backward()
                optimizer.step()
                mark_optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), div=loss_div.item(), ce=loss_ce.item())
            if lr_scheduler:
                lr_scheduler.step()
            mark_scheduler.step()
            self.model.eval()
            self.mark_generator.eval()
            if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
                self.validate_fn()
        optimizer.zero_grad()
        mark_optimizer.zero_grad()
        self.mark_generator.requires_grad_(False)
        self.mask_generator.requires_grad_(False)
        self.model.requires_grad_(False)

    def train_mask(self):
        optimizer = torch.optim.Adam(self.mask_generator.parameters(), lr=1e-2, betas=(0.5, 0.9))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_mask_epochs)
        loader = self.dataset.loader['train']
        dataset = loader.dataset
        logger = MetricLogger()
        logger.create_meters(loss=None, div=None, norm=None)
        print_prefix = 'Mask Epoch'
        for _epoch in range(self.train_mask_epochs):
            _epoch += 1
            idx = torch.randperm(len(dataset))
            pos = 0
            logger.reset()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                print_prefix, output_iter(_epoch, self.train_mask_epochs), **ansi)
            header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
            self.mask_generator.train()
            for data in logger.log_every(loader, header=header):
                optimizer.zero_grad()
                _input, _label = self.model.get_data(data)
                batch_size = len(_input)
                data2 = sample_batch(dataset, idx=idx[pos:pos + batch_size])
                _input2, _label2 = self.model.get_data(data2)
                pos += batch_size

                _mask = self.get_mask(_input)
                _mask2 = self.get_mask(_input2)

                input_dist: torch.Tensor = (_input - _input2).flatten(1).norm(p=2, dim=1)
                mask_dist: torch.Tensor = (_mask - _mask2).flatten(1).norm(p=2, dim=1) + 1e-5

                loss_div = input_dist.div(mask_dist).mean()
                loss_norm = _mask.sub(self.mask_density).relu().mean()

                loss = self.lambda_norm * loss_norm + self.lambda_div * loss_div
                loss.backward()
                optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), div=loss_div.item(), norm=loss_norm.item())
            lr_scheduler.step()
            self.mask_generator.eval()
            if _epoch % (max(self.train_mask_epochs // 5, 1)) == 0 or _epoch == self.train_mask_epochs:
                self.validate_mask()
        optimizer.zero_grad()

    @torch.no_grad()
    def validate_mask(self):
        loader = self.dataset.loader['valid']
        dataset = loader.dataset
        logger = MetricLogger()
        logger.create_meters(loss=None, div=None, norm=None)
        idx = torch.randperm(len(dataset))
        pos = 0

        print_prefix = 'Validate'
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        for data in logger.log_every(loader, header=header):
            _input, _label = self.model.get_data(data)
            batch_size = len(_input)
            data2 = sample_batch(dataset, idx=idx[pos:pos + batch_size])
            _input2, _label2 = self.model.get_data(data2)
            pos += batch_size

            _mask = self.get_mask(_input)
            _mask2 = self.get_mask(_input2)

            input_dist: torch.Tensor = (_input - _input2).flatten(1).norm(p=2, dim=1)
            mask_dist: torch.Tensor = (_mask - _mask2).flatten(1).norm(p=2, dim=1) + 1e-5

            loss_div = input_dist.div(mask_dist).mean()
            loss_norm = _mask.sub(self.mask_density).relu().mean()

            loss = self.lambda_norm * loss_norm + self.lambda_div * loss_div
            logger.update(n=batch_size, loss=loss.item(), div=loss_div.item(), norm=loss_norm.item())

    @staticmethod
    def define_generator(num_channels: list[int] = [32, 64, 128],
                         in_channel: int = 3, out_channel: int = None
                         ) -> nn.Sequential:
        out_channel = out_channel or in_channel
        down_channel_list = num_channels.copy()
        down_channel_list.insert(0, in_channel)
        up_channel_list = num_channels[::-1].copy()
        up_channel_list.append(out_channel)

        seq = nn.Sequential()
        down_seq = nn.Sequential()
        middle_seq = nn.Sequential()
        up_seq = nn.Sequential()
        for i in range(len(num_channels)):
            down_seq.add_module(f'conv{3*i+1}', conv3x3(down_channel_list[i], down_channel_list[i + 1]))
            down_seq.add_module(f'bn{3*i+1}', nn.BatchNorm2d(down_channel_list[i + 1], momentum=0.05))
            down_seq.add_module(f'relu{3*i+1}', nn.ReLU(inplace=True))
            down_seq.add_module(f'conv{3*i+2}', conv3x3(down_channel_list[i + 1], down_channel_list[i + 1]))
            down_seq.add_module(f'bn{3*i+2}', nn.BatchNorm2d(down_channel_list[i + 1], momentum=0.05))
            down_seq.add_module(f'relu{3*i+2}', nn.ReLU(inplace=True))
            down_seq.add_module(f'maxpool{3*i+3}', nn.MaxPool2d(kernel_size=2))
        middle_seq.add_module('conv', conv3x3(num_channels[-1], num_channels[-1]))
        middle_seq.add_module('bn', nn.BatchNorm2d(num_channels[-1], momentum=0.05))
        middle_seq.add_module('relu', nn.ReLU(inplace=True))
        for i in range(len(num_channels)):
            up_seq.add_module(f'maxpool{3*i+1}', nn.Upsample(scale_factor=2.0))
            up_seq.add_module(f'conv{3*i+2}', conv3x3(up_channel_list[i], up_channel_list[i]))
            up_seq.add_module(f'bn{3*i+2}', nn.BatchNorm2d(up_channel_list[i], momentum=0.05))
            up_seq.add_module(f'relu{3*i+2}', nn.ReLU(inplace=True))
            up_seq.add_module(f'conv{3*i+3}', conv3x3(up_channel_list[i], up_channel_list[i + 1]))
            up_seq.add_module(f'bn{3*i+3}', nn.BatchNorm2d(up_channel_list[i + 1], momentum=0.05))
            up_seq.add_module(f'relu{3*i+3}', nn.ReLU(inplace=True))
        seq.add_module('down', down_seq)
        seq.add_module('middle', middle_seq)
        seq.add_module('up', up_seq)
        return seq
