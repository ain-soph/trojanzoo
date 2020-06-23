# -*- coding: utf-8 -*-

from trojanzoo.dataset import ImageSet
from trojanzoo.model import ImageModel

import torch


class STRIP():
    def __init__(self, dataset: ImageSet, model: ImageModel,
                 alpha: float = 0.5, N: int = 8, detach: bool = True):
        self.dataset: ImageSet = dataset
        self.model: ImageModel = model

        self.alpha: float = alpha
        self.N: int = N

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        _input2 = _input2[:_input1.shape[0]]

        result = alpha * (_input1 - _input2) + _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        p = self.model.get_prob(_input)
        return (-p * p.log()).sum(1).mean()

    def detect(self, _input) -> float:
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
