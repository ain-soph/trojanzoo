# -*- coding: utf-8 -*-

from .badnet import BadNet

from trojanzoo.attack.adv import PGD

import torch
from tqdm import tqdm

from trojanzoo.utils import Config
env = Config.env


class IMC(BadNet):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang. 

    Based on :class:`trojanzoo.attack.backdoor.BadNet`,
    IMC preprocesses the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc'

    def __init__(self, preprocess_layer: str = 'features', threshold: float = 5, target_value: float = 10,
                 neuron_lr: float = 0.015, neuron_epoch: int = 20, neuron_num: int = 2, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')

        self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        self.param_list['imc'] = ['']

        self.pgd = PGD(alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
                       target_idx=self.target_class, universal=True, output=0)

    def attack(self, epoch: int, save=False, **kwargs):
        super().attack(epoch, save=save, epoch_func=self.epoch_func, **kwargs)

    def epoch_func(self, **kwargs):
        for data in self.dataset.loader['test']:
            _input, _label = self.model.get_data(data)
            adv_input, _iter = self.craft_example(_input)
