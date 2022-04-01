#!/usr/bin/env python3

from ...abstract import BackdoorAttack

import torch
import argparse

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class Unlearn(BackdoorAttack):

    name: str = 'unlearn'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--attack_source', help='attack source (default: "badnet")')
        group.add_argument('--mark_source', help='mark source (default: "attack")')
        return group

    def __init__(self, mark_source: str = 'attack', attack_source: str = 'badnet', **kwargs):
        self.attack_source = attack_source
        mark_source = mark_source if mark_source != 'attack' else attack_source
        self.mark_source = mark_source
        super().__init__(**kwargs)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 keep_org: bool = True, poison_label=False, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return super().get_data(data, keep_org=keep_org, poison_label=poison_label, **kwargs)

    def get_filename(self, **kwargs):
        return f'{self.attack_source}_{self.mark_source}_{self.train_mode}_' + super().get_filename(**kwargs)

    def get_poison_dataset(self, poison_label: bool = False, poison_num: int = None) -> torch.utils.data.Dataset:
        return super().get_poison_dataset(poison_label, poison_num)
