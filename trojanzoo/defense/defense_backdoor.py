# -*- coding: utf-8 -*-

from .defense import Defense
from trojanzoo.attack.backdoor.badnet import BadNet


class Defense_Backdoor(Defense):

    name: str = 'defense_backdoor'

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
        return self.attack.name + self.attack.get_filename(**kwargs)
