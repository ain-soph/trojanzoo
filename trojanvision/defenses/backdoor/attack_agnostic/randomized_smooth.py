#!/usr/bin/env python3

from ...abstract import BackdoorDefense


class RandomizedSmooth(BackdoorDefense):
    name: str = 'randomized_smooth'

    def detect(self, **kwargs):
        super().detect(**kwargs)
        self.model.randomized_smooth = True
        self.attack.validate_fn()
