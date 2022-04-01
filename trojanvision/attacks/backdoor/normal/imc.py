#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack imc
"""  # noqa: E501

from .trojannn import TrojanNN
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.optim as optim
import argparse

from collections.abc import Callable


class IMC(TrojanNN):
    r"""Input Model Co-optimization (IMC) proposed by Ren Pang
    from Pennsylvania State University in CCS 2020.

    It inherits :class:`trojanvision.attacks.BackdoorAttack`.

    Based on :class:`trojanvision.attacks.TrojanNN`,
    IMC optimizes the watermark using Adam optimizer during model retraining.

    See Also:
        * paper: `A Tale of Evil Twins\: Adversarial Inputs versus Poisoned Models`_
        * code: TrojanZoo is the official implementation of IMC ^_^

    Args:
        attack_remask_epochs (int): Inner epoch to optimize watermark during each training epoch.
            Defaults to ``20``.
        attack_remask_lr (float): Learning rate of Adam optimizer to optimize watermark.
            Defaults to ``0.1``.

    .. _A Tale of Evil Twins\: Adversarial Inputs versus Poisoned Models:
        https://arxiv.org/abs/1911.01559
    """  # noqa: E501

    name: str = 'imc'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--attack_remask_epochs', type=int,
                           help='inner epoch to optimize watermark during each training epoch '
                           '(default: 1)')
        group.add_argument('--attack_remask_lr', type=float,
                           help='learning rate of Adam optimizer to optimize watermark'
                           '(default: 0.1)')
        return group

    def __init__(self, attack_remask_epochs: int = 1, attack_remask_lr: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.param_list['imc'] = ['attack_remask_epochs', 'attack_remask_lr']
        self.attack_remask_epochs = attack_remask_epochs
        self.attack_remask_lr = attack_remask_lr

    def attack(self, epochs: int, **kwargs):
        return super().attack(epochs, epoch_fn=self.epoch_fn, **kwargs)

    def epoch_fn(self, **kwargs):
        self.optimize_mark()

    def optimize_mark(self, loss_fn: Callable[..., torch.Tensor] = None, **kwargs):
        r"""Optimize watermark at the beginning of each training epoch."""
        loss_fn = loss_fn or self.model.loss

        atanh_mark = torch.randn_like(self.mark.mark[:-1], requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.attack_remask_lr)
        optimizer.zero_grad()

        for _ in range(self.attack_remask_epochs):
            for data in self.dataset.loader['train']:
                self.mark.mark[:-1] = tanh_func(atanh_mark)
                _input, _label = self.model.get_data(data)
                trigger_input = self.add_mark(_input)
                trigger_label = self.target_class * torch.ones_like(_label)
                loss = loss_fn(trigger_input, trigger_label)
                loss.backward(inputs=[atanh_mark])
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark.detach_()
        atanh_mark.requires_grad_(False)
        self.mark.mark[:-1] = tanh_func(atanh_mark)
        self.mark.mark.detach_()
