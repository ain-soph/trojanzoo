# -*- coding: utf-8 -*-

from trojanzoo.defenses import Defense
from trojanvision.attacks.backdoor import BadNet

import argparse


class BackdoorDefense(Defense):

    name: str = None

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--original', dest='original', action='store_true',
                           help='load original clean model, defaults to False.')

    def __init__(self, original: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.original: bool = original
        self.attack: BadNet  # for linting purpose
        self.target_class = self.attack.target_class

    def detect(self, **kwargs):
        if not self.original:
            self.attack.load(**kwargs)
        if self.attack.name == 'trojannet':
            self.model = self.attack.combined_model
        self.attack.validate_func()

    def get_filename(self, **kwargs):
        return self.attack.name + '_' + self.attack.get_filename(**kwargs)
