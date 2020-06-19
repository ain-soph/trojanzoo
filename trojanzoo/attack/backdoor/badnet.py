# -*- coding: utf-8 -*-

from trojanzoo.attack import Attack
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils import save_tensor_as_img

import random
from typing import Union, List

import os
import torch


class BadNet(Attack):

    name = 'badnet'

    def __init__(self, mark: Watermark = None, target_class: int = 0, percent: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['badnet'] = ['target_class', 'percent']
        self.mark: Watermark = mark
        self.target_class: int = target_class
        self.percent: float = percent

    def attack(self, epoch: int, save=False, **kwargs):
        self.model._train(epoch, get_data=self.get_data, validate_func=self.validate_func, **kwargs)
        if save:
            self.save(epoch=epoch)

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, **kwargs)

    def get_filename(self, epoch: int, mark_alpha: float = None, target_class: int = None):
        if mark_alpha is None:
            mark_alpha = self.mark.mark_alpha
        if target_class is None:
            target_class = self.target_class
        _file = '{mark}_tar{target:d}_alpha{mark_alpha:.2f}_mark({height:d},{width:d})_percent{percent:.2f}_epoch{epoch:d}'.format(
            mark=os.path.split(self.mark.mark_path)[1][:-4],
            target=target_class, mark_alpha=mark_alpha, epoch=epoch, percent=self.percent,
            height=self.mark.height, width=self.mark.width)
        return _file

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True, poison_label=True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        if not keep_org or random.uniform(0, 1) < self.percent:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input)
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label)
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_func(self, get_data=None, **kwargs) -> (float, float, float):
        self.model._validate(print_prefix='Validate Clean',
                             get_data=None, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Tgt',
                             get_data=get_data, keep_org=False, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org',
                             get_data=get_data, keep_org=False, poison_label=False, **kwargs)
        # todo: Return value
        return 0.0, 0.0, 0.0

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mark.save_npz(file_path + '.npz')
        save_tensor_as_img(file_path + '.png', self.mark.mark)
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mark.load_npz(file_path + '.npz')
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)
