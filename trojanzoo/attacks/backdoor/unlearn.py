# -*- coding: utf-8 -*-

from .badnet import BadNet

import torch
from typing import Tuple


class Unlearn(BadNet):

    name: str = 'unlearn'

    def get_data(self, data: Tuple[torch.Tensor, torch.LongTensor], keep_org: bool = True, poison_label=False, **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
        return super().get_data(data, keep_org=keep_org, poison_label=poison_label, **kwargs)
