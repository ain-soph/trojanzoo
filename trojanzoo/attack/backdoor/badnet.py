# -*- coding: utf-8 -*-

from trojanzoo.attack import Attack
from trojanzoo.utils.attack import Watermark

import random
from typing import Union, List

import os
import torch


class BadNet(Attack):

    name = 'badnet'

    def __init__(self, mark: Watermark = None, target_class: int = 0, percent: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mark: Watermark = mark
        self.target_class: int = target_class
        self.percent: float = percent
        self.filename: str = self.get_filename()

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, epoch: int = None, **kwargs):
        if epoch is None:
            epoch = self.iteration
        self.model._train(epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler,
                          get_data=self.get_data, validate_func=self.validate_func, **kwargs)

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, **kwargs)

    def get_filename(self, mark_alpha: float = None, target_class: int = None, iteration: int = None):
        if mark_alpha is None:
            mark_alpha = self.mark.mark_alpha
        if target_class is None:
            target_class = self.target_class
        if iteration is None:
            iteration = self.iteration
        _file = '{mark}_tar{target:d}_alpha{mark_alpha:.2f}_mark({height:d},{width:d})_iter{iteration:d}_percent{percent:.2f}'.format(
            mark=os.path.split(self.path)[1][:-4], target=target_class,
            mark_alpha=mark_alpha, iteration=iteration, percent=self.percent,
            height=self.mark.height, width=self.mark.width)
        return _file

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        if not keep_org or random.uniform(0, 1) < self.percent:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input)
            _label = self.target_class*torch.ones_like(org_label)
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_func(self, **kwargs) -> (float, float, float):
        self.model._validate(print_prefix='Validate Clean',
                             get_data=None, **kwargs)
        self.model._validate(print_prefix='Validate Watermark',
                             keep_org=False, **kwargs)
        return 0.0, 0.0, 0.0
