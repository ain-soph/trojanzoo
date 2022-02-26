#!/usr/bin/env python3

from .trojannn import TrojanNN
from trojanvision.utils.sgm import register_hook, remove_hook
from trojanzoo.utils.logger import AverageMeter
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.optim as optim
import argparse

from collections.abc import Callable


class IMC(TrojanNN):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attacks.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--attack_remask_epoch', type=int)
        group.add_argument('--attack_remask_lr', type=float)
        return group

    def __init__(self, attack_remask_epoch: int = 20, attack_remask_lr: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if self.mark.mark_random_pos:
            raise Exception('IMC requires \'random pos\' to be False to train mark.')
        self.param_list['imc'] = ['attack_remask_epoch', 'attack_remask_lr']
        self.attack_remask_epoch = attack_remask_epoch
        self.attack_remask_lr = attack_remask_lr

    def attack(self, epochs: int, **kwargs):
        super().attack(epochs, epoch_fn=self.epoch_fn, **kwargs)

    def epoch_fn(self, **kwargs):
        if self.model.sgm and 'sgm_remove' not in self.model.__dict__.keys():
            register_hook(self.model, self.model.sgm_gamma)
        self.optimize_mark()
        if self.model.sgm:
            remove_hook(self.model)

    def optimize_mark(self, loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        loss_fn = loss_fn or self.model.loss

        atanh_mark = torch.randn_like(self.mark.mark[:-1], requires_grad=True)
        self.mark.mark[:-1] = tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.attack_remask_lr)
        optimizer.zero_grad()

        losses = AverageMeter('Loss', ':.4e')
        for _ in range(self.attack_remask_epoch):
            for i, data in enumerate(self.dataset.loader['train']):
                if i > 20:  # TODO: remove this?
                    break
                _input, _label = self.model.get_data(data)
                poison_x = self.mark.add_mark(_input)
                loss = loss_fn(poison_x, self.target_class * torch.ones_like(_label))
                loss.backward(inputs=[atanh_mark])
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark[:-1] = tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(_label))
        atanh_mark.requires_grad_(False)
        self.mark.mark.detach_()
