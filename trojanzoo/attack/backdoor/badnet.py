# -*- coding: utf-8 -*-

from trojanzoo.attack import Attack
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils import save_tensor_as_img

import random
from typing import Union, List

import os
import torch


class BadNet(Attack):
    r"""
    BadNet Backdoor Attack is described in detail in the paper `BadNet`_ by Tianyu Gu. 

    It attaches a fixed watermark to benign images and inject them into training set with target label.
    After retraining, the model will classify all images with watermark attached into target class.

    The authors have posted `original source code`_.

    Args:
        mark (Watermark): the attached watermark image.
        target_class (int): the target class. Default: ``0``.
        percent (int): The proportion of malicious images in the training set (Max 0.5). Default: 0.1.

    .. _BadNet:
        https://arxiv.org/abs/1708.06733

    .. _original source code:
        https://github.com/Kooscii/BadNets
    """

    name: str = 'badnet'

    def __init__(self, mark: Watermark = None, target_class: int = 0, percent: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['badnet'] = ['target_class', 'percent']
        self.mark: Watermark = mark
        self.target_class: int = target_class
        self.percent: float = percent
        _, clean_acc, _ = self.model._validate(print_prefix='Baseline Clean',
                                               get_data=None, **kwargs)
        self.clean_acc = clean_acc

    def attack(self, epoch: int, save=False, get_data=None, loss_fn='self', **kwargs):
        if isinstance(get_data, str) and get_data == 'self':
            get_data = self.get_data
        if isinstance(loss_fn, str) and loss_fn == 'self':
            loss_fn = self.loss_fn
        self.model._train(epoch, save=save,
                          validate_func=self.validate_func, get_data=get_data, loss_fn=loss_fn,
                          save_fn=self.save, **kwargs)

    def get_filename(self, mark_alpha: float = None, target_class: int = None, **kwargs):
        if mark_alpha is None:
            mark_alpha = self.mark.mark_alpha
        if target_class is None:
            target_class = self.target_class
        _file = '{mark}_tar{target:d}_alpha{mark_alpha:.2f}_mark({height:d},{width:d})_percent{percent:.2f}'.format(
            mark=os.path.split(self.mark.mark_path)[1][:-4],
            target=target_class, mark_alpha=mark_alpha, percent=self.percent,
            height=self.mark.height, width=self.mark.width)
        # _epoch{epoch:d} epoch=epoch,
        if self.mark.random_pos:
            _file = 'random_pos_' + _file
        return _file

    # ---------------------- I/O ----------------------------- #

    def save(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mark.save_npz(file_path + '.npz')
        self.mark.save_img(file_path + '.png')
        self.model.save(file_path + '.pth')
        print('attack results saved at: ', file_path)

    def load(self, **kwargs):
        filename = self.get_filename(**kwargs)
        file_path = self.folder_path + filename
        self.mark.load_npz(file_path + '.npz')
        self.model.load(file_path + '.pth')
        print('attack results loaded from: ', file_path)

    # ---------------------- Utils ---------------------------- #

    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, **kwargs)

    def loss_fn(self, _input: torch.Tensor, _label: torch.LongTensor, **kwargs) -> torch.Tensor:
        loss_clean = self.model.loss(_input, _label, **kwargs)
        poison_input = self.mark.add_mark(_input)
        poison_label = self.target_class * torch.ones_like(_label)
        loss_poison = self.model.loss(poison_input, poison_label, **kwargs)
        return (1 - self.percent) * loss_clean + self.percent * loss_poison

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True, poison_label=True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        percent = self.percent / (1 - self.percent)
        if not keep_org or random.uniform(0, 1) < percent:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input)
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label)
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_func(self, get_data=None, loss_fn=None, **kwargs) -> (float, float, float):
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        target_loss, target_acc, _ = self.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_data, keep_org=False, **kwargs)
        _, orginal_acc, _ = self.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_data, keep_org=False, poison_label=False, **kwargs)
        # todo: Return value
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc
