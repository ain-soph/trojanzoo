# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor

import torch
from trojanzoo.utils.model import AverageMeter


class STRIP(Defense_Backdoor):
    def __init__(self, alpha: float = 0.5, N: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.alpha: float = alpha
        self.N: int = N

    def detect(self, **kwargs):
        super().detect(**kwargs)
        entropy = AverageMeter('entropy', fmt='.4e')
        for i, data in enumerate(self.dataset.loader['test']):
            _input, _label = self.model.get_data(data)
            entropy.update(self.defense.check(_input), n=_label.size(0))
            print(f'{i:<10d}{entropy.avg:<20.4f}')

    def check(self, _input) -> float:
        h = 0.0
        for i, data in enumerate(self.dataset.loader['train']):
            if i >= self.N:
                break
            X, Y = self.model.get_data(data)
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test)
            h += entropy
        h /= self.N
        return h

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        _input2 = _input2[:_input1.shape[0]]

        result = alpha * (_input1 - _input2) + _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        p = self.model.get_prob(_input)
        return (-p * p.log()).sum(1).mean()
