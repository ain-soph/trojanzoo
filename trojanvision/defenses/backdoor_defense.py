#!/usr/bin/env python3

from trojanzoo.defenses import Defense
from trojanvision.datasets import ImageSet
from trojanvision.models import ImageModel
from trojanvision.attacks.backdoor import BadNet

import argparse


class BackdoorDefense(Defense):

    name: str = 'backdoor_defense'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--original', action='store_true',
                           help='load original clean model, defaults to False.')
        return group

    def __init__(self, original: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dataset: ImageSet
        self.model: ImageModel
        self.attack: BadNet  # for linting purpose
        self.original: bool = original
        self.target_class = self.attack.target_class

    def detect(self, **kwargs):
        if not self.original:
            self.attack.load(**kwargs)
        if self.attack.name == 'trojannet':
            self.model = self.attack.combined_model
        self.attack.validate_fn()

    def get_filename(self, **kwargs):
        return self.attack.name + '_' + self.attack.get_filename(**kwargs)
