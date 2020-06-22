# -*- coding: utf-8 -*-

from ..attack import Attack

from trojanzoo.utils import add_noise, to_list
from trojanzoo.utils.output import prints
from trojanzoo.optim import PGD as PGD_Optimizer

import torch
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

    def attack(self, _input: torch.Tensor, loss_fn: Callable = None,
               target: Union[torch.LongTensor, int] = None, target_idx: int = None, **kwargs):
        if len(_input) == 0:
            return _input, None
        if target_idx is None:
            target_idx = self.target_idx
        if loss_fn is None and self.loss_fn is None:
            if target is None:
                target = self.generate_target(_input, idx=target_idx)
            elif isinstance(target, int):
                target = torch.ones_like(_input) * target

            def _loss_fn(_X):
                loss = self.model.loss(_X, target)
                return loss if target_idx else -loss
            loss_fn = _loss_fn
        return self.optimize(_input, loss_fn=loss_fn, target=target, **kwargs)

    def early_stop_check(self, X, target, loss_fn=None, **kwargs):
        if self.stop_threshold is None:
            return False
        _confidence = self.model.get_target_prob(X, target)
        if self.targeted and _confidence.min() > self.stop_threshold:
            return True
        if not self.targeted and _confidence.max() < self.stop_threshold:
            return True
        return False

    def output_info(self, _input, noise, target, indent=None, **kwargs):
        super().output_info(_input, noise, indent=indent, **kwargs)
        _confidence = self.model.get_target_prob(_input + noise, target)
        prints(to_list(_confidence), indent=indent)
