#!/usr/bin/env python3

from trojanvision.attacks.backdoor.imc import IMC
from trojanvision.models import MagNet

import torch
import math
import random


class IMC_MagNet(IMC):
    name: str = 'imc_magnet'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.magnet = MagNet(dataset=self.dataset, pretrain=True)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True, poison_label=True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        _input, _label = self.model.get_data(data)
        decimal, integer = math.modf(self.poison_num)
        integer = int(integer)
        if random.uniform(0, 1) < decimal:
            integer += 1
        if not keep_org:
            integer = len(_label)
        if not keep_org or integer:
            org_input, org_label = _input, _label
            _input = self.add_mark(org_input[:integer])
            _input = self.magnet(_input)
            _label = _label[:integer]
            if poison_label:
                _label = self.target_class * torch.ones_like(org_label[:integer])
            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label
