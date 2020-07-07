# -*- coding: utf-8 -*-

from trojanzoo.utils.process import Model_Process


class Attack(Model_Process):
    name: str = 'attack'

    def attack(self, **kwargs):
        pass
    # ----------------------Utility----------------------------------- #

    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.LongTensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)
