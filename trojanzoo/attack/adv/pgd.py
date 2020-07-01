# -*- coding: utf-8 -*-

from ..attack import Attack

from trojanzoo.utils import add_noise, to_list
from trojanzoo.utils.output import prints
from trojanzoo.optim import PGD as PGD_Optimizer

import torch
import torch.nn.functional as F
from typing import Union, List
from collections.abc import Callable


class PGD(Attack, PGD_Optimizer):
    r"""PGD Adversarial Attack.
    Args:
        alpha (float): learning rate :math:`\alpha`. Default: :math:`\frac{3}{255}`.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{8}{255}`.
    """

    name = 'pgd'

    def __init__(self, target_idx: int = 1, **kwargs):
        self.target_idx: int = target_idx
        super().__init__(**kwargs)

    def attack(self):
        # model._validate()
        correct = 0
        total = 0
        total_iter = 0
        for i, data in enumerate(self.dataset.loader['test']):
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
            print('{} / {}'.format(correct, total))
            print('current iter: ', _iter)
            print('succ rate: ', float(correct) / total)
            if correct > 0:
                print('avg  iter: ', float(total_iter) / correct)
            print('-------------------------------------------------')
            print()

    def craft_example(self, _input: torch.Tensor, loss_fn: Callable = None,
                      target: Union[torch.LongTensor, int] = None, target_idx: int = None, **kwargs):
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
                loss = F.cross_entropy(self.model(_X), t, **kwargs)
                return loss if target_idx else -loss
            loss_fn = _loss_fn
        return self.optimize(_input, loss_fn=loss_fn, target=target, **kwargs)

    def early_stop_check(self, X, target, loss_fn=None, **kwargs):
        if not self.stop_threshold:
            return False
        _confidence = self.model.get_target_prob(X, target)
        if self.target_idx and _confidence.min() > self.stop_threshold:
            return True
        if not self.target_idx and _confidence.max() < self.stop_threshold:
            return True
        return False

    def output_info(self, _input: torch.Tensor, noise: torch.Tensor, target: torch.LongTensor, indent: int = 0, **kwargs):
        super().output_info(_input, noise, indent=indent, **kwargs)
        # prints('Original class     : ', to_list(_label), indent=indent)
        # prints('Original confidence: ', to_list(_confidence), indent=indent)
        _confidence = self.model.get_target_prob(_input + noise, target)
        prints('Target   class     : ', to_list(target), indent=indent)
        prints('Target   confidence: ', to_list(_confidence), indent=indent)
