#!/usr/bin/env python3

from .badnet import BadNet
from trojanvision.optim import PGDoptimizer    # TODO: Need to check whether this will cause ImportError
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
        threshold (float): the target threshold. Default: ``5``.
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
        group.add_argument('--preprocess_layer',
                           help='the chosen feature layer patched by trigger where rare neuron activation is maxmized, defaults to ``flatten``')
        group.add_argument('--threshold', type=float, help='Trojan Net Threshold, defaults to 5')
        group.add_argument('--target_value', type=float,
                           help='Trojan Net Target_Value, defaults to 10')
        group.add_argument('--neuron_lr', type=float,
                           help='Trojan Net learning rate in neuron preprocessing, defaults to 0.015')
        group.add_argument('--neuron_epoch', type=int, help='Trojan Net epoch in neuron preprocessing, defaults to 20')
        group.add_argument('--neuron_num', type=int,
                           help='Trojan Net neuron numbers in neuron preprocessing, defaults to 2')
        return group

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

        self.pgd = PGDoptimizer(pgd_alpha=self.neuron_lr, pgd_eps=1.0,
                                iteration=self.neuron_epoch, output=0, **kwargs)

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
                if fm.dim() > 2:
                    fm = fm.flatten(start_dim=2).mean(dim=2)
                fm = fm.mean(dim=0)
                result.append(fm.detach())
            if env['tqdm']:
                print('{upline}{clear_line}'.format(**ansi))
            return torch.stack(result).sum(dim=0).argsort(descending=False)[:self.neuron_num]

    def get_neuron_value(self, x: torch.Tensor, neuron_idx: torch.Tensor) -> torch.Tensor:
        fm = self.model.get_layer(x, layer_output=self.preprocess_layer)
        loss: torch.Tensor = fm[:, neuron_idx].flatten(1).norm(p=2, dim=1)
        return loss.mean()

    # train the mark to activate the least-used neurons.
    def preprocess_mark(self, mark: torch.Tensor, neuron_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            print("Neuron Value Before Preprocessing: ",
                  float(self.get_neuron_value(mark.unsqueeze(0), neuron_idx)))

        def loss_fn(_input: torch.Tensor, reduction: str = 'mean', **kwargs) -> torch.Tensor:
            fm = self.model.get_layer(_input, layer_output=self.preprocess_layer)
            loss: torch.Tensor = (fm[:, neuron_idx] - self.target_value).flatten(1).norm(p=2, dim=1)
            return loss if reduction == 'none' else loss.mean()
        noise = torch.zeros_like(mark.unsqueeze(0))
        x = mark.unsqueeze(0)
        for _iter in range(self.neuron_epoch):
            cost = loss_fn(x)
            if cost < self.threshold:
                break
            x, _ = self.pgd.optimize(x, noise=noise, iteration=1, loss_fn=loss_fn)
            noise = noise * self.mark.mask
            x = x * self.mark.mask
        x = x.detach()
        # with torch.no_grad():
        #     print("Neuron Value After Preprocessing: ",
        #           float(self.get_neuron_value(x, neuron_idx)))
        return x[0]

    def validate_fn(self, get_data_fn=None, **kwargs) -> tuple[float, float]:
        if self.neuron_idx is not None:
            with torch.no_grad():
                mark = (self.mark.mark * self.mark.mask).unsqueeze(0)
                print("Neuron Value After Preprocessing: ",
                      float(self.get_neuron_value(mark, self.neuron_idx)))
        return super().validate_fn(get_data_fn=get_data_fn, **kwargs)
