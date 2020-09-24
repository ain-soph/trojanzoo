# -*- coding: utf-8 -*-

from .badnet import BadNet

from trojanzoo.optim import PGD
from trojanzoo.utils.data import MyDataset

import torch
import numpy as np
import math
import random
from typing import Tuple, Callable


class Hidden_Trigger(BadNet):
    r"""
    Hidden Trigger Backdoor Attack is described in detail in the paper `Hidden Trigger`_ by Aniruddha Saha. 

    Different from :class:`trojanzoo.attack.backdoor.TrojanNN`, The mark and mask is designated and stable.

    The authors have posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen feature layer patched by trigger where distance to poisoned images is minimized. Default: 'features'.
        pgd_alpha (float, optional): the learning rate to generate poison images. Default: 0.01.
        pgd_epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{16}{255}`.
        pgd_iteration (int): the iteration number to generate one poison image. Default: 5000.

    .. _Hidden Trigger:
        https://arxiv.org/abs/1910.00033

    .. _original source code:
        https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
    """

    name: str = 'hidden_trigger'

    def __init__(self, preprocess_layer: str = 'features', pgd_epsilon: int = 16.0 / 255,
                 pgd_iteration: int = 40, pgd_alpha: float = 4.0 / 255, **kwargs):
        super().__init__(**kwargs)

        self.param_list['hidden_trigger'] = ['preprocess_layer', 'pgd_alpha', 'pgd_epsilon', 'pgd_iteration']

        self.preprocess_layer: str = preprocess_layer
        self.pgd_alpha: float = pgd_alpha
        self.pgd_epsilon: float = pgd_epsilon
        self.pgd_iteration: int = pgd_iteration

        self.target_loader = self.dataset.get_dataloader('train', full=True, classes=self.target_class,
                                                         drop_last=True, num_workers=0)
        self.pgd: PGD = PGD(alpha=self.pgd_alpha, epsilon=pgd_epsilon, iteration=pgd_iteration, output=self.output)

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True, poison_label=True, training=True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org:
            integer = len(_label)
        if not keep_org or integer:
            org_input, org_label = _input, _label
            if training:
                _input = org_input[org_label != self.target_class][:integer]
                _input = self.generate_poisoned_data(_input)
            else:
                _input = self.add_mark(org_input[:integer])
            _label = _label[:integer]
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label[:integer][:len(_input)])
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label

    def validate_func(self, get_data=None, loss_fn=None, **kwargs) -> (float, float, float):
        clean_loss, clean_acc, _ = self.model._validate(print_prefix='Validate Clean',
                                                        get_data=None, **kwargs)
        target_loss, target_acc, _ = self.model._validate(print_prefix='Validate Trigger Tgt',
                                                          get_data=self.get_data, keep_org=False, training=False, **kwargs)
        _, orginal_acc, _ = self.model._validate(print_prefix='Validate Trigger Org',
                                                 get_data=self.get_data, keep_org=False, poison_label=False, training=False, **kwargs)
        # todo: Return value
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:
            target_acc = 0.0
        return clean_loss + target_loss, target_acc, clean_acc

    def loss(self, poison_imgs: torch.Tensor, source_feats: torch.Tensor) -> torch.Tensor:
        poison_feats = self.model.get_layer(poison_imgs, layer_output=self.preprocess_layer)
        return (poison_feats - source_feats).flatten(start_dim=1).norm(p=2, dim=1).mean()

    def generate_poisoned_data(self, source_imgs: torch.FloatTensor) -> torch.Tensor:
        r"""
        **Algorithm1**

        Sample K images of target class (Group I)

        Initialize poisoned images (Group III) to be Group I.

        **while** loss is large:

            Sample K images of other classes (trigger attached at random location) (Group II).

            conduct gradient descent on group III to minimize the distance to Group II in feature space.

            Clip Group III to ensure the distance to Group I in input space to be smaller than :math:`\epsilon`.

        **Return** Group III

        .. note::
            In the original code, Group II is sampled with Group I together rather than resampled in every loop. We are following this style.
        """

        # -----------------------------Prepare Data--------------------------------- #
        source_imgs = self.add_mark(source_imgs)
        source_feats = self.model.get_layer(source_imgs, layer_output=self.preprocess_layer).detach()

        target_imgs, _ = self.model.get_data(next(iter(self.target_loader)))
        target_imgs = target_imgs[:len(source_imgs)]
        # -----------------------------Poison Frog--------------------------------- #

        def loss_func(poison_imgs):
            return self.loss(poison_imgs, source_feats=source_feats)
        poison_imgs, _ = self.pgd.optimize(_input=target_imgs, loss_fn=loss_func)
        return poison_imgs
