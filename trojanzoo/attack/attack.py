# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Model_Process

import torch
import argparse


class Attack(Model_Process):
    name: str = 'attack'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--output', dest='output', type=int,
                           help='output level, defaults to 0.')

    def attack(self, **kwargs):
        pass
    # ----------------------Utility----------------------------------- #

    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.LongTensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)
