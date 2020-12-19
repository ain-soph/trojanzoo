# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Model_Process

import torch
import argparse


class Attack(Model_Process):
    name: str = None

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('--attack', dest='attack_name')
        group.add_argument('--attack_dir', dest='attack_dir',
                           help='directory to contain attack results')
        group.add_argument('--output', dest='output', type=int,
                           help='output level, defaults to 0.')

    def attack(self, **kwargs):
        pass
    # ----------------------Utility----------------------------------- #

    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.Tensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)
