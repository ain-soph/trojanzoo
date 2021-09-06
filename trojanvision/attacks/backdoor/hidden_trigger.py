#!/usr/bin/env python3

from .badnet import BadNet
from trojanvision.optim import PGDoptimizer
from trojanzoo.utils.output import prints

import torch
import math
import random
import argparse
from typing import Callable


class HiddenTrigger(BadNet):
    r"""
    Hidden Trigger Backdoor Attack is described in detail in the paper `Hidden Trigger`_ by Aniruddha Saha. 

    Different from :class:`trojanzoo.attacks.backdoor.TrojanNN`, The mark and mask is designated and stable.

    The authors have posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen feature layer patched by trigger where distance to poisoned images is minimized. Default: 'features'.
        pgd_alpha (float, optional): the learning rate to generate poison images. Default: 0.01.
        pgd_eps (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{16}{255}`.
        pgd_iter (int): the iteration number to generate one poison image. Default: 5000.

    .. _Hidden Trigger:
        https://arxiv.org/abs/1910.00033

    .. _original source code:
        https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
    """

    name: str = 'hidden_trigger'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--preprocess_layer',
                           help='the chosen feature layer patched by trigger where distance to poisoned images is minimized, defaults to ``flatten``')
        group.add_argument('--pgd_alpha', type=float,
                           help='the learning rate to generate poison images, defaults to 0.01')
        group.add_argument('--pgd_eps', type=int, help='the perturbation threshold in input space, defaults to 16')
        group.add_argument('--pgd_iter', type=int,
                           help='the iteration number to generate one poison image, defaults to 5000')
        return group

    def __init__(self, preprocess_layer: str = 'features', pgd_eps: int = 16.0 / 255,
                 pgd_iter: int = 40, pgd_alpha: float = 4.0 / 255, **kwargs):
        super().__init__(**kwargs)

        self.param_list['hidden_trigger'] = ['preprocess_layer', 'pgd_alpha', 'pgd_eps', 'pgd_iter']

        self.preprocess_layer: str = preprocess_layer
        self.pgd_alpha: float = pgd_alpha
        self.pgd_eps: float = pgd_eps
        self.pgd_iter: int = pgd_iter

        self.target_loader = self.dataset.get_dataloader('train', full=True, class_list=[self.target_class],
                                                         drop_last=True, num_workers=0)
        self.pgd = PGDoptimizer(pgd_alpha=self.pgd_alpha, pgd_eps=pgd_eps, iteration=pgd_iter, output=self.output)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True, poison_label=True, training=True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
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

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs) -> tuple[float, float]:
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                             get_data_fn=self.get_data, keep_org=False, training=False, indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org', main_tag='',
                             get_data_fn=self.get_data, keep_org=False,
                             poison_label=False, training=False, indent=indent, **kwargs)
        prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        prints(f'Neuron Jaccard Idx: {self.check_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
        return clean_acc, target_acc

    def loss(self, poison_imgs: torch.Tensor, source_feats: torch.Tensor,
             reduction: str = 'mean', **kwargs) -> torch.Tensor:
        poison_feats = self.model.get_layer(poison_imgs, layer_output=self.preprocess_layer)
        result: torch.Tensor = (poison_feats - source_feats).flatten(start_dim=1).norm(p=2, dim=1)
        return result if reduction == 'none' else result.mean()

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

        def loss_func(poison_imgs: torch.Tensor, source_feats: torch.Tensor, **kwargs):
            return self.loss(poison_imgs, source_feats=source_feats, **kwargs)
        poison_imgs, _ = self.pgd.optimize(_input=target_imgs, loss_fn=loss_func,
                                           loss_kwargs={'source_feats': source_feats})
        return poison_imgs
