# -*- coding: utf-8 -*-

from .defense import Defense
from trojanzoo.attack import BadNet


class Defense_Backdoor(Defense):

    name = 'defense_backdoor'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attack: BadNet

    def detect(self, **kwargs):
        self.model.activate_params([])
        self.attack.load(**kwargs)
        self.attack.validate_func()
