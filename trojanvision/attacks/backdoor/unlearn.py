#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .badnet import BadNet
import torch


class Unlearn(BadNet):

    name: str = 'unlearn'

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 keep_org: bool = True, poison_label=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super().get_data(data, keep_org=keep_org, poison_label=poison_label, **kwargs)
