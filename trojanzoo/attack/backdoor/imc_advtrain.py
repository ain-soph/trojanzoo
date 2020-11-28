# -*- coding: utf-8 -*-

from .imc import IMC

import torch

import math
import random
from typing import Dict, Tuple


class IMC_AdvTrain(IMC):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attack.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc_advtrain'

    def get_data(self, data: Tuple[torch.Tensor, torch.LongTensor], **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        _input, _label = self.model.get_data(data, **kwargs)

        def loss_fn(X: torch.FloatTensor):
            return -self.model.loss(X, _label)
        adv_x, _ = self.pgd.optimize(_input=_input, loss_fn=loss_fn)
        return adv_x, _label

    def get_poison_data(self, data: Tuple[torch.Tensor, torch.LongTensor], poison_label: bool = True, strip: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        _input, _label = self.model.get_data(data)
        integer = len(_label)
        if poison_label:
            _label = self.target_class * torch.ones_like(_label[:integer])
        else:
            _label = _label[:integer]
        return _input, _label

    def validate_func(self, get_data=None, loss_fn=None, **kwargs) -> Tuple[float, float, float]:
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        target_loss, target_acc, _ = self.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_poison_data, **kwargs)
        _, orginal_acc, _ = self.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_poison_data, poison_label=False, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Tgt',
                             get_data=self.get_poison_data, strip=True, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Org',
                             get_data=self.get_poison_data, strip=True, poison_label=False, **kwargs)
        print(f'Validate Confidence : {self.validate_confidence():.3f}')
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc
