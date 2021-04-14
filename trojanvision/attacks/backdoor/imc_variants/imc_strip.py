#!/usr/bin/env python3

from trojanvision.attacks.backdoor.imc import IMC
from trojanzoo.utils.output import prints

import torch
import math
import random
from typing import Callable


class IMC_STRIP(IMC):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attacks.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc_strip'

    def add_strip_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mark.add_mark(x, alpha=1 - (1 - self.mark.mark_alpha) / 2, **kwargs)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer])
            _label = self.target_class * torch.ones_like(org_label[:integer])
            strip_input = self.add_strip_mark(org_input[:integer])
            strip_label = org_label[:integer]
            _input = torch.cat((_input, org_input, strip_input))
            _label = torch.cat((_label, org_label, strip_label))
        return _input, _label

    def get_poison_data(self, data: tuple[torch.Tensor, torch.Tensor], poison_label: bool = True, strip: bool = False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        integer = len(_label)
        if strip:
            _input = self.add_strip_mark(_input[:integer])
        else:
            _input = self.add_mark(_input[:integer])
        if poison_label:
            _label = self.target_class * torch.ones_like(_label[:integer])
        else:
            _label = _label[:integer]
        return _input, _label

    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, **kwargs) -> tuple[float, float]:
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)
        _, target_acc = self.model._validate(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                             get_data_fn=self.get_poison_data, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org', main_tag='',
                             get_data_fn=self.get_poison_data, poison_label=False,
                             indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Tgt', main_tag='',
                             get_data_fn=self.get_poison_data, strip=True,
                             indent=indent, **kwargs)
        self.model._validate(print_prefix='Validate STRIP Org', main_tag='',
                             get_data_fn=self.get_poison_data, strip=True, poison_label=False,
                             indent=indent, **kwargs)
        prints(f'Validate Confidence: {self.validate_confidence():.3f}', indent=indent)
        prints(f'Neuron Jaccard Idx: {self.check_neuron_jaccard():.3f}', indent=indent)
        if self.clean_acc - clean_acc > 3 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
        return clean_acc, target_acc
