#!/usr/bin/env python3

from ..abstract import InputFiltering

import torch
import argparse


class Strip(InputFiltering):
    name: str = 'strip'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--defense_fpr', type=float,
                           help='FPR value for input filtering defense (default: 0.05)')
        return group

    def __init__(self, defense_fpr: float = 0.05,
                 alpha: float = 0.5, N: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.param_list['strip'] = ['defense_fpr', 'alpha', 'N']
        self.defense_fpr = defense_fpr
        self.alpha: float = alpha
        self.N: int = N
        self.loader = self.dataset.get_dataloader(mode='train', drop_last=True)

    def check(self, _input: torch.Tensor, **kwargs) -> torch.Tensor:
        _list = []
        for i, data in enumerate(self.loader):
            if i >= self.N:
                break
            X, Y = self.model.get_data(data)
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test).cpu()
            _list.append(entropy)
        return torch.stack(_list).mean(0)

    def score2label(self, clean_scores: torch.Tensor, poison_scores: torch.Tensor) -> torch.Tensor:
        threshold_low = float(clean_scores[int(self.defense_fpr * len(poison_scores))])
        threshold_high = float(clean_scores[int((1 - self.defense_fpr) * len(poison_scores))])
        entropy = torch.cat((clean_scores, poison_scores))
        print('Entropy Clean  Median: ', float(clean_scores.median()))
        print('Entropy Poison Median: ', float(poison_scores.median()))
        print(f'Threshold: ({threshold_low:5.3f}, {threshold_high:5.3f})')
        return torch.where(((entropy < threshold_low).int() + (entropy > threshold_high).int()).bool(),
                           torch.ones_like(entropy).bool(), torch.zeros_like(entropy).bool())

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        _input2 = _input2[:_input1.shape[0]]

        result = alpha * (_input1 - _input2) + _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        _output = self.model(_input)
        return self.model.criterion(_output, _output.softmax(1))
