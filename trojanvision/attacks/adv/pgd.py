#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trojanvision.optim import PGD as PGD_Optimizer
from trojanzoo.attacks import Attack
from trojanzoo.utils import to_list
from trojanzoo.utils.output import prints

import torch
import argparse
from collections.abc import Callable
from typing import Union


class PGD(Attack, PGD_Optimizer):
    r"""PGD Adversarial Attack.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.
    """

    name: str = 'pgd'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--alpha', dest='alpha', type=float,
                           help='PGD learning rate per step, defaults to 3.0/255')
        group.add_argument('--epsilon', dest='epsilon', type=float,
                           help='Projection norm constraint, defaults to 8.0/255')
        group.add_argument('--iteration', dest='iteration', type=int,
                           help='Attack Iteration, defaults to 20')
        group.add_argument('--stop_threshold', dest='stop_threshold', type=float,
                           help='early stop confidence, defaults to None')
        group.add_argument('--target_idx', dest='target_idx', type=int,
                           help='Target label order in original classification, defaults to 1 '
                           '(0 for untargeted attack, 1 for most possible class, -1 for most unpossible class)')

        group.add_argument('--grad_method', dest='grad_method',
                           help='gradient estimation method, defaults to \'white\'')
        group.add_argument('--query_num', dest='query_num', type=int,
                           help='query numbers for black box gradient estimation, defaults to 100.')
        group.add_argument('--sigma', dest='sigma', type=float,
                           help='gaussian sampling std for black box gradient estimation, defaults to 1e-3')

    def __init__(self, target_idx: int = 1, **kwargs):
        self.target_idx: int = target_idx
        super().__init__(**kwargs)

    def attack(self):
        # model._validate()
        correct = 0
        total = 0
        total_iter = 0
        for data in self.dataset.loader['test']:
            if total >= 100:
                break
            _input, _label = self.model.remove_misclassify(data)
            if len(_label) == 0:
                continue
            adv_input, _iter = self.craft_example(_input)

            total += 1
            if _iter:
                correct += 1
                total_iter += _iter
            print(f'{correct} / {total}')
            print('current iter: ', _iter)
            print('succ rate: ', float(correct) / total)
            if correct > 0:
                print('avg  iter: ', float(total_iter) / correct)
            print('-------------------------------------------------')
            print()

    def craft_example(self, _input: torch.Tensor, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                      target: Union[torch.Tensor, int] = None, target_idx: int = None, **kwargs):
        if len(_input) == 0:
            return _input, None
        if target_idx is None:
            target_idx = self.target_idx
        if loss_fn is None and self.loss_fn is None:
            if target is None:
                target = self.generate_target(_input, idx=target_idx)
            elif isinstance(target, int):
                target = target * torch.ones(len(_input), dtype=torch.long, device=_input.device)

            def _loss_fn(_X: torch.Tensor, **kwargs):
                t = target
                if len(_X) != len(target) and len(target) == 1:
                    t = target * torch.ones(len(_X), dtype=torch.long, device=_X.device)
                loss = self.model.loss(_X, t, **kwargs)
                return loss if target_idx else -loss
            loss_fn = _loss_fn
        return self.optimize(_input, loss_fn=loss_fn, target=target, **kwargs)

    def early_stop_check(self, X, target=None, loss_fn=None, **kwargs):
        if not self.stop_threshold:
            return False
        with torch.no_grad():
            _confidence = self.model.get_target_prob(X, target)
        if self.target_idx and _confidence.min() > self.stop_threshold:
            return True
        if not self.target_idx and _confidence.max() < self.stop_threshold:
            return True
        return False

    def output_info(self, _input: torch.Tensor, noise: torch.Tensor, target: torch.Tensor, **kwargs):
        super().output_info(_input, noise, **kwargs)
        # prints('Original class     : ', to_list(_label), indent=self.indent)
        # prints('Original confidence: ', to_list(_confidence), indent=self.indent)
        with torch.no_grad():
            _confidence = self.model.get_target_prob(_input + noise, target)
        prints('Target   class     : ', to_list(target), indent=self.indent)
        prints('Target   confidence: ', to_list(_confidence), indent=self.indent)
