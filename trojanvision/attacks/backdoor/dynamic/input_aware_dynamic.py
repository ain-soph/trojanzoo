#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --attack input_aware_dynamic
"""  # noqa: E501

from ...abstract import BackdoorAttack

from trojanzoo.environ import env
from trojanzoo.utils.data import sample_batch
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter

import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

import math
import random

import os

import argparse
from collections.abc import Callable


class InputAwareDynamic(BackdoorAttack):
    r"""Input-Aware Dynamic Backdoor Attack proposed by Anh Nguyen and Anh Tran
    from VinAI Research in NIPS 2020.

    Based on :class:`trojanvision.attacks.BadNet`,
    InputAwareDynamic trains mark generator and mask generator
    to synthesize unique watermark for each input.

    In classification loss, besides attacking poison inputs and classifying clean inputs,
    InputAwareDynamic also requires inputs attached
    with triggers generated from other inputs
    are still classified correctly (cross-trigger mode).

    See Also:
        * paper: `Input-Aware Dynamic Backdoor Attack`_
        * code: https://github.com/VinAIResearch/input-aware-backdoor-attack-release

    .. math::
       \begin{aligned}
            &\textbf{\# train mask generator}                                                                                                      \\
            &{opt}_{mask} = \text{Adam}(G_{mask}.parameters(), \text{lr}=0.01, \text{betas}=(0.5, 0.9))                                            \\
            &\textbf{for} \: e=1 \: \textbf{to} \: \text{train\_mask\_epochs}                                                                      \\
            &\hspace{5mm}\textbf{for} \: x_1 \: \textbf{in} \: \text{train\_set}                                                                   \\
            &\hspace{10mm}x_2 = \text{sample\_another\_batch}(\text{train\_set})                                                                   \\
            &\hspace{10mm}\mathcal{L}_{div}  = \frac{\lVert x_1 - x_2 \rVert}{\lVert G_{mask}(x_1) - G_{mask}(x_2) \rVert}                         \\
            &\hspace{10mm}\mathcal{L}_{norm} = ReLU(G_{mask}(x_1) - \text{mask\_density}).mean()                                                   \\
            &\hspace{10mm}\mathcal{L}_{mask} = \lambda_{div} \mathcal{L}_{div} + \lambda_{norm} \mathcal{L}_{norm}                                 \\
            &\hspace{10mm}{opt}_{mask}.step()                                                                                                      \\
            &\rule{110mm}{0.4pt}                                                                                                                   \\
            &\textbf{\# train mark generator and model}                                                                                            \\
            &{opt}_{mark} = \text{Adam}(G_{mark}.parameters(), \text{lr}=0.01, \text{betas}=(0.5, 0.9))                                            \\
            &\textbf{for} \: e=1 \: \textbf{to} \: \text{epochs}                                                                                   \\
            &\hspace{5mm}\textbf{for} \: (x_1, y_1) \: \textbf{in} \: \text{train\_set}                                                            \\
            &\hspace{10mm}x_2 = \text{sample\_another\_batch}(\text{train\_set})                                                                   \\
            &\hspace{10mm}{mark}_{poison}, {mask}_{poison} = G_{mark}, G_{mask} (x_1[:n_{poison}])                                                 \\
            &\hspace{10mm}{mark}_{cross}, {mask}_{cross}   = G_{mark}, G_{mask} (x_2[n_{poison}: n_{poison} + n_{cross}])                          \\
            &\hspace{10mm}x_{poison} = {mask}_{poison} \cdot {mark}_{poison} + (1 - {mask}_{poison}) \cdot x_1[:n_{poison}]                        \\
            &\hspace{10mm}x_{cross}  = {mask}_{cross}  \cdot {mark}_{cross}  + (1 - {mask}_{cross})  \cdot x_1[n_{poison}: n_{poison} + n_{cross}] \\
            &\hspace{10mm}x = cat([x_{poison}, x_{cross}, x_1[n_{poison}+n_{cross}:]])                                                             \\
            &\hspace{10mm}y = cat([y_{poison}, y_1[n_{poison}:]])                                                                                  \\
            &\hspace{10mm}\mathcal{L}_{div} = \frac{\lVert x_{poison} - x_{cross} \rVert}{\lVert {mark}_{poison} - {mark}_{cross} \rVert}          \\
            &\hspace{10mm}\mathcal{L}_{ce}  = cross\_entropy(x, y)                                                                                 \\
            &\hspace{10mm}\mathcal{L}       = \mathcal{L}_{ce} + \lambda_{div}\mathcal{L}_{div}                                                    \\
            &\hspace{10mm}{opt}_{mark}.step()                                                                                                      \\
            &\hspace{10mm}{opt}_{model}.step()                                                                                                     \\
       \end{aligned}

    Args:
        train_mask_epochs (int): Epoch to optimize mask generator.
            Defaults to ``25``.
        lambda_div (float): Weight of diversity loss
            during both optimization processes.
            Defaults to ``1.0``.
        lambda_norm (float): Weight of norm loss
            when optimizing mask generator.
            Defaults to ``100.0``.
        mask_density (float): Threshold of mask values
            when optimizing norm loss.
            Defaults to ``0.032``.
        cross_percent (float): Percentage of cross inputs
            in the whole training set.
            Defaults to ``0.1``.
        poison_percent (float): Percentage of poison inputs
            in the whole training set.
            Defaults to ``0.1``.
        natural (bool): Whether to use natural backdoors.
            If ``True``, model parameters will be frozen.
            Defaults to ``False``.

    Attributes:
        mark_generator (torch.nn.Sequential): Mark generator instance
            constructed by :meth:`define_generator()`.
            Output shape ``(N, C, H, W)``.
        mask_generator (torch.nn.Sequential): Mark generator instance
            constructed by :meth:`define_generator()`.
            Output shape ``(N, 1, H, W)``.

    Note:
        Do **NOT** directly call :attr:`self.mark_generator`
        or :attr:`self.mask_generator`.
        Their raw outputs are not normalized into range ``[0, 1]``.
        Please call :meth:`get_mark()` and :meth:`get_mask()` instead.


    .. _Input-Aware Dynamic Backdoor Attack:
        https://arxiv.org/abs/2010.08138
    """  # noqa: E501
    name: str = 'input_aware_dynamic'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--train_mask_epochs', type=int,
                           help='Epoch to optimize mask generator '
                           'before optimizing mark generator and model '
                           '(default: 25)')
        group.add_argument('--lambda_div', type=float,
                           help='weight of diversity loss '
                           'during both optimization processes '
                           '(default: 1.0)')
        group.add_argument('--lambda_norm', type=float,
                           help='weight of norm loss '
                           'when optimizing mask generator '
                           '(default: 100.0)')
        group.add_argument('--mask_density', type=float,
                           help='threshold of mask values '
                           'when optimizing norm loss '
                           '(default: 0.032)')
        group.add_argument('--cross_percent', type=float,
                           help='percentage of cross inputs '
                           'in the whole training set '
                           '(default: 0.032)')
        group.add_argument('--natural', action='store_true',
                           help='whether to use natural backdoors. '
                           'if true, model parameters will be frozen')
        return group

    def __init__(self, train_mask_epochs: int = 25,
                 lambda_div: float = 1.0, lambda_norm: float = 100.0,
                 mask_density: float = 0.032,
                 cross_percent: float = 0.1,
                 natural: bool = False,
                 poison_percent: float = 0.1, **kwargs):
        super().__init__(poison_percent=poison_percent, **kwargs)
        self.param_list['input_aware_dynamic'] = ['train_mask_epochs', 'natural',
                                                  'lambda_div', 'lambda_norm',
                                                  'mask_density', 'cross_percent']

        self.train_mask_epochs = train_mask_epochs
        self.lambda_div = lambda_div
        self.lambda_norm = lambda_norm
        self.mask_density = mask_density
        self.natural = natural

        self.poison_ratio = self.poison_percent
        self.poison_num = 0  # monkey patch: to avoid batch size change in badnet
        self.cross_percent = cross_percent

        data_channel = self.dataset.data_shape[0]
        num_channels = [16, 32] if data_channel == 1 else [32, 64, 128]
        self.mark_generator = self.define_generator(
            num_channels, in_channels=data_channel
        ).to(device=env['device']).eval()
        self.mask_generator = self.define_generator(
            num_channels, in_channels=data_channel,
            out_channels=1).to(device=env['device']).eval()

        self.train_set = self.dataset.loader['train'].dataset
        self.idx = torch.randperm(len(self.train_set))
        self.pos = 0

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor by calling
        :meth:`get_mark()` and :meth:`get_mask()`."""
        mark = self.get_mark(x)
        mask = self.get_mask(x)
        return x + mask * (mark - x)

    def get_mark(self, _input: torch.Tensor) -> torch.Tensor:
        r"""Get mark with shape ``(N, C, H, W)``.

        .. math::
            \begin{aligned}
                &raw = \text{self.mark\_generator(input)} \\
                &\textbf{return} \frac{\tanh{(raw)} + 1}{2}
            \end{aligned}
        """
        raw_output: torch.Tensor = self.mark_generator(_input)
        return raw_output.tanh() / 2 + 0.5

    def get_mask(self, _input: torch.Tensor) -> torch.Tensor:
        r"""Get mask with shape ``(N, 1, H, W)``.

        .. math::
            \begin{aligned}
                &raw = \text{self.mask\_generator(input)} \\
                &\textbf{return} \frac{\tanh{[10 \cdot \tanh{(raw)}]} + 1}{2}
            \end{aligned}
        """
        raw_output: torch.Tensor = self.mask_generator(_input)
        return raw_output.tanh().mul(10).tanh() / 2 + 0.5

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get data.

        Note:
            The difference between this and
            :meth:`trojanvision.attacks.BadNet.get_data()` is:

            This method replaces some clean data with poison version,
            while BadNet's keeps the clean data and append poison version.
        """
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

    def _get_cross_data(self, data: tuple[torch.Tensor, torch.Tensor],
                        **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get cross-trigger mode data.
        Sample another batch from train set
        and apply their marks and masks to current batch.
        """
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
        clean_acc, _ = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        asr, _ = self.model._validate(print_prefix='Validate ASR', main_tag='valid asr',
                                      get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                      indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate Cross', main_tag='valid cross',
                             get_data_fn=self._get_cross_data, indent=indent, **kwargs)
        # prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        # prints(f'Neuron Jaccard Idx: {self.get_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > threshold:
            asr = 0.0
        return asr, clean_acc

    def attack(self, epochs: int, optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               validate_interval: int = 1, save: bool = False,
               verbose: bool = True, **kwargs):
        if verbose:
            print('train mask generator')
        self.mark_generator.requires_grad_(False)
        self.mask_generator.requires_grad_()
        self.model.requires_grad_(False)
        self.train_mask_generator(verbose=verbose)
        if verbose:
            print()
            print('train mark generator and model')

        self.mark_generator.requires_grad_()
        self.mask_generator.requires_grad_(False)
        if not self.natural:
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

        best_validate_result = (0.0, float('inf'))
        if validate_interval != 0:
            best_validate_result = self.validate_fn(verbose=verbose)
            best_asr = best_validate_result[0]
        for _epoch in range(epochs):
            _epoch += 1
            idx = torch.randperm(len(dataset))
            pos = 0
            logger.reset()
            if not self.natural:
                self.model.train()
            self.mark_generator.train()
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            for data in logger.log_every(loader, header=header) if verbose else loader:
                if not self.natural:
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

                loss_ce = self.model.loss(final_input, final_label)
                loss = loss_ce
                # div loss
                loss_div = torch.zeros_like(loss_ce)
                if len(trigger_input) > 0 and len(cross_input) > 0:
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
                    loss_div = input_dist.div(mark_dist).mean().nan_to_num(0.0)
                    loss = loss_ce + self.lambda_div * loss_div

                loss.backward()
                if not self.natural:
                    optimizer.step()
                mark_optimizer.step()
                logger.update(n=batch_size, loss=loss.item(), div=loss_div.item(), ce=loss_ce.item())
            if not self.natural and lr_scheduler:
                lr_scheduler.step()
            mark_scheduler.step()
            if not self.natural:
                self.model.eval()
            self.mark_generator.eval()
            if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
                validate_result = self.validate_fn(verbose=verbose)
                cur_asr = validate_result[0]
                if cur_asr >= best_asr:
                    best_validate_result = validate_result
                    best_asr = cur_asr
                    if save:
                        self.save()
        if not self.natural:
            optimizer.zero_grad()
        mark_optimizer.zero_grad()
        self.mark_generator.requires_grad_(False)
        self.mask_generator.requires_grad_(False)
        self.model.requires_grad_(False)
        return best_validate_result

    def get_filename(self, target_class: int = None, **kwargs) -> str:
        r"""Get filenames for current attack settings."""
        if target_class is None:
            target_class = self.target_class
        _file = 'tar{target:d}'.format(target=target_class)
        _file = 'tar{target:d} poison{poison:.2f} cross{cross:.2f}'.format(
            target=target_class, poison=self.poison_percent, cross=self.cross_percent)
        return _file

    def save(self, filename: str = None, **kwargs):
        r"""Save attack results to files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        torch.save(self.mask_generator.state_dict(), file_path + '_mask.pth')
        torch.save(self.mark_generator.state_dict(), file_path + '_mark.pth')
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, filename: str = None, **kwargs):
        r"""Load attack results from previously saved files."""
        filename = filename or self.get_filename(**kwargs)
        file_path = os.path.join(self.folder_path, filename)
        self.mask_generator.load_state_dict(torch.load(file_path + '_mask.pth'))
        self.mark_generator.load_state_dict(torch.load(file_path + '_mark.pth'))
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)

    def train_mask_generator(self, verbose: bool = True):
        r"""Train :attr:`self.mask_generator`."""
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
            for data in logger.log_every(loader, header=header) if verbose else loader:
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
            if verbose and (_epoch % (max(self.train_mask_epochs // 5, 1)) == 0 or _epoch == self.train_mask_epochs):
                self.validate_mask_generator()
        optimizer.zero_grad()

    @torch.no_grad()
    def validate_mask_generator(self):
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
                         in_channels: int = 3, out_channels: int = None
                         ) -> nn.Sequential:
        r"""Define a generator used in :attr:`self.mark_generator` and :attr:`self.mask_generator`.

        Similar to auto-encoders, the generator is composed of ``['down', 'middle', 'up']``.

        * **down**: :math:`[\text{conv-bn-relu}(c_{i}, c_{i+1}), \text{conv-bn-relu}(c_{i+1}, c_{i+1}), \text{maxpool}(2)]`
        * **middle**: :math:`[\text{conv-bn-relu}(c_{-1}, c_{-1})]`
        * **up**: :math:`[\text{upsample}(2), \text{conv-bn-relu}(c_{i+1}, c_{i+1}), \text{conv-bn-relu}(c_{i+1}, c_{i})]`

        Args:
            num_channels (list[int]): List of intermediate feature numbers.
                Each element serves as the :attr:`in_channels` of current layer
                and :attr:`out_features` of preceding layer.
                Defaults to ``[32, 64, 128]``.

                * MNIST: ``[16, 32]``
                * CIFAR: ``[32, 64, 128]``

            in_channels (int): :attr:`in_channels` of first conv layer in ``down``.
                It should be image channels.
                Defaults to ``3``.
            out_channels (int): :attr:`out_channels` of last conv layer in ``up``.
                Defaults to ``None`` (:attr:`in_channels`).

        Returns:
            torch.nn.Sequential: Generator instance with input shape ``(N, in_channels, H, W)``
                and output shape ``(N, out_channels, H, W)``.
        """  # noqa: E501
        out_channels = out_channels or in_channels
        down_channel_list = num_channels.copy()
        down_channel_list.insert(0, in_channels)
        up_channel_list = num_channels[::-1].copy()
        up_channel_list.append(out_channels)

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
            up_seq.add_module(f'upsample{3*i+1}', nn.Upsample(scale_factor=2.0, mode='bilinear'))
            up_seq.add_module(f'conv{3*i+2}', conv3x3(up_channel_list[i], up_channel_list[i]))
            up_seq.add_module(f'bn{3*i+2}', nn.BatchNorm2d(up_channel_list[i], momentum=0.05))
            up_seq.add_module(f'relu{3*i+2}', nn.ReLU(inplace=True))
            up_seq.add_module(f'conv{3*i+3}', conv3x3(up_channel_list[i], up_channel_list[i + 1]))
            up_seq.add_module(f'bn{3*i+3}', nn.BatchNorm2d(up_channel_list[i + 1], momentum=0.05))
            if i != len(num_channels) - 1:
                up_seq.add_module(f'relu{3*i+3}', nn.ReLU(inplace=True))
        seq.add_module('down', down_seq)
        seq.add_module('middle', middle_seq)
        seq.add_module('up', up_seq)
        return seq
