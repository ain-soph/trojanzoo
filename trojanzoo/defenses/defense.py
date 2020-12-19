# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Model_Process
from trojanzoo.attacks import Attack

import argparse


class Defense(Model_Process):

    name: str = None

    @staticmethod
    def add_argument(group: argparse._ArgumentGroup):
        group.add_argument('--defense', dest='defense_name')
        group.add_argument('--defense_dir', dest='defense_dir',
                           help='directory to contain defense results')

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack: Attack = attack

    def detect(self, **kwargs):
        raise NotImplementedError()
