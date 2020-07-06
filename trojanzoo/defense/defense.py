# -*- coding: utf-8 -*-

from trojanzoo.attack import Attack


class Defense(Attack):

    name: str = 'defense'

    def __init__(self, attack: Attack = None, **kwargs):
        super().__init__(**kwargs)
        self.attack: Attack = attack

    def detect(self, **kwargs):
        raise NotImplementedError()
