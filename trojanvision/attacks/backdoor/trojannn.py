#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .badnet import BadNet
from trojanvision.attacks.adv import PGD    # TODO: Need to check whether this will cause ImportError
from trojanvision.environ import env
from trojanzoo.utils.output import ansi

import torch
import argparse
from tqdm import tqdm


class TrojanNN(BadNet):

    r"""
    TrojanNN Backdoor Attack is described in detail in the paper `TrojanNN`_ by Yingqi Liu. 

    Based on :class:`trojanzoo.attacks.backdoor.BadNet`,
    TrojanNN preprocesses the watermark pixel values to maximize the neuron activation on rarely used neurons
    to avoid the negative impact of model performance on clean iamges.

    The authors have posted `original source code`_.

    Args:
        preprocess_layer (str): The preprocess layer.
        threshold (float): the target class. Default: ``5``.
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _TrojanNN:
        https://github.com/PurduePAML/TrojanNN/blob/master/trojan_nn.pdf

    .. _original source code:
        https://github.com/PurduePAML/TrojanNN
    """

    name: str = 'trojannn'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                           help='the chosen feature layer patched by trigger where rare neuron activation is maxmized, defaults to ``flatten``')
        group.add_argument('--threshold', dest='threshold', type=float,
                           help='Trojan Net Threshold, defaults to 5')
        group.add_argument('--target_value', dest='target_value', type=float,
                           help='Trojan Net Target_Value, defaults to 10')
        group.add_argument('--neuron_lr', dest='neuron_lr', type=float,
                           help='Trojan Net learning rate in neuron preprocessing, defaults to 0.015')
        group.add_argument('--neuron_epoch', dest='neuron_epoch', type=int,
                           help='Trojan Net epoch in neuron preprocessing, defaults to 20')
        group.add_argument('--neuron_num', dest='neuron_num', type=int,
                           help='Trojan Net neuron numbers in neuron preprocessing, defaults to 2')

    def __init__(self, preprocess_layer: str = 'flatten', threshold: float = 5, target_value: float = 10,
                 neuron_lr: float = 0.015, neuron_epoch: int = 20, neuron_num: int = 2, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('TrojanNN requires "random pos" to be False to max activate neurons.')

        self.param_list['trojannn'] = ['preprocess_layer', 'threshold', 'target_value',
                                       'neuron_lr', 'neuron_epoch', 'neuron_num']
        self.preprocess_layer: str = preprocess_layer
        self.threshold: float = threshold
        self.target_value: float = target_value

        self.neuron_lr: float = neuron_lr
        self.neuron_epoch: int = neuron_epoch
        self.neuron_num: int = neuron_num
        self.neuron_idx = None

        self.pgd = PGD(alpha=self.neuron_lr, epsilon=1.0, iteration=self.neuron_epoch, output=0)

    def attack(self, *args, **kwargs):
        self.neuron_idx = self.get_neuron_idx()
        self.mark.mark = self.preprocess_mark(mark=self.mark.mark * self.mark.mask, neuron_idx=self.neuron_idx)
        super().attack(*args, **kwargs)

    # get the neuron idx for preprocess.
    def get_neuron_idx(self) -> torch.Tensor:
        with torch.no_grad():
            result = []
            loader = self.dataset.loader['train']
            if env['tqdm']:
                loader = tqdm(loader)
            for i, data in enumerate(loader):
                _input, _label = self.model.get_data(data)
                fm = self.model.get_layer(_input, layer_output=self.preprocess_layer)
                if len(fm.shape) > 2:
                    fm = fm.flatten(start_dim=2).mean(dim=2)
                fm = fm.mean(dim=0)
                result.append(fm.detach())
            if env['tqdm']:
                print('{upline}{clear_line}'.format(**ansi))
            return torch.stack(result).sum(dim=0).argsort(descending=False)[:self.neuron_num]

    def get_neuron_value(self, x: torch.Tensor, neuron_idx: torch.Tensor) -> torch.Tensor:
        return self.model.get_layer(x, layer_output=self.preprocess_layer)[:, neuron_idx].mean()

    # train the mark to activate the least-used neurons.
    def preprocess_mark(self, mark: torch.Tensor, neuron_idx: torch.Tensor, **kwargs):
        with torch.no_grad():
            print("Neuron Value Before Preprocessing: ",
                  self.get_neuron_value(mark, neuron_idx))

        def loss_fn(X: torch.Tensor):
            fm = self.model.get_layer(X, layer_output=self.preprocess_layer)
            loss = fm[:, neuron_idx].mean(dim=0) - self.target_value
            return loss.norm(p=2)
        noise = torch.zeros_like(mark)
        x = mark
        for _iter in range(self.neuron_epoch):
            cost = loss_fn(x)
            if cost < self.threshold:
                break
            x, _ = self.pgd.craft_example(mark, noise=noise, iteration=1, loss_fn=loss_fn)
            noise = noise * self.mark.mask
            x = x * self.mark.mask
        x = x.detach()
        with torch.no_grad():
            print("Neuron Value After Preprocessing: ",
                  self.get_neuron_value(x, neuron_idx))
        return x

    def validate_func(self, get_data_fn=None, loss_fn=None, **kwargs) -> tuple[float, float, float]:
        if self.neuron_idx is not None:
            with torch.no_grad():
                print("Neuron Value After Preprocessing: ",
                      self.get_neuron_value(self.mark.mark * self.mark.mask, self.neuron_idx))
        return super().validate_func(get_data_fn=get_data_fn, loss_fn=loss_fn, **kwargs)
