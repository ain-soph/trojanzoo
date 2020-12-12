# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Model_Process
from trojanzoo.attack import Attack

import argparse


class Defense(Model_Process):

    name: str = 'defense'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        group.add_argument('--defense', dest='defense_name')

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack: Attack = attack

    def detect(self, **kwargs):
        raise NotImplementedError()
