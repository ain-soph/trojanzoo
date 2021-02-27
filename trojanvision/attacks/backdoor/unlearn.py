#!/usr/bin/env python3

from .badnet import BadNet
import torch
import argparse


class Unlearn(BadNet):

    name: str = 'unlearn'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--mark_source', dest='mark_source', type=str,
                           help='mark source, defaults to ``attack``')

    def __init__(self, mark_source: str = 'attack', **kwargs):
        super().__init__(**kwargs)
        self.mark_source = mark_source

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 keep_org: bool = True, poison_label=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super().get_data(data, keep_org=keep_org, poison_label=poison_label, **kwargs)

    def get_filename(self, **kwargs):
        return f'{self.mark_source}_' + super().get_filename(**kwargs)
