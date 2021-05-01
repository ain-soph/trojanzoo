#!/usr/bin/env python3

from .trojannn import TrojanNN
from trojanvision.utils.sgm import register_hook, remove_hook
from trojanzoo.utils import AverageMeter, tanh_func

import torch
import torch.optim as optim
import argparse


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
        group.add_argument('--inner_iter', type=int)
        group.add_argument('--inner_lr', type=float)
        return group

    def __init__(self, inner_iter: int = 20, inner_lr: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')
        self.param_list['imc'] = ['inner_iter', 'inner_lr']
        self.inner_iter: int = inner_iter
        self.inner_lr: float = inner_lr

    def attack(self, epoch: int, **kwargs):
        super().attack(epoch, epoch_fn=self.epoch_fn, **kwargs)

    def epoch_fn(self, **kwargs):
        if self.model.sgm and 'sgm_remove' not in self.model.__dict__.keys():
            register_hook(self.model, self.model.sgm_gamma)
        self.optimize_mark()
        if self.model.sgm:
            remove_hook(self.model)

    def optimize_mark(self, loss_fn=None, **kwargs):
        atanh_mark = torch.randn_like(self.mark.mark) * self.mark.mask
        atanh_mark.requires_grad_()
        self.mark.mark = tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.inner_lr)
        optimizer.zero_grad()

        if loss_fn is None:
            loss_fn = self.model.loss

        losses = AverageMeter('Loss', ':.4e')
        for _epoch in range(self.inner_iter):
            for i, data in enumerate(self.dataset.loader['train']):
                if i > 20:
                    break
                _input, _label = self.model.get_data(data)
                poison_x = self.mark.add_mark(_input)
                loss = loss_fn(poison_x, self.target_class * torch.ones_like(_label))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark = tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(_label))
        atanh_mark.requires_grad = False
        self.mark.mark.detach_()
