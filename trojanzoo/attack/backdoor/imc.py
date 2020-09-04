# -*- coding: utf-8 -*-

from .trojannn import TrojanNN

from trojanzoo.optim import PGD as PGD_Optim
from trojanzoo.utils.sgm import register_hook, remove_hook

import torch
from tqdm import tqdm

from trojanzoo.utils import Config
env = Config.env


class IMC(TrojanNN):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang. 

    Based on :class:`trojanzoo.attack.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc'

    def __init__(self, pgd_alpha: float = 20 / 255, pgd_epsilon: float = 1.0, pgd_iteration: int = 20, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')

        self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        # self.param_list['imc'] = ['']

        self.pgd_alpha: float = pgd_alpha
        self.pgd_epsilon: float = pgd_epsilon
        self.pgd_iteration: int = pgd_iteration
        self.pgd_optim = PGD_Optim(alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
                                   loss_fn=self.loss_pgd, universal=True, output=0)

    def attack(self, epoch: int, **kwargs):
        super().attack(epoch, epoch_func=self.epoch_func, **kwargs)

    def epoch_func(self, **kwargs):
        if self.model.sgm and 'sgm_remove' not in self.model.__dict__.keys():
            register_hook(self.model, self.model.sgm_gamma)
        for data in tqdm(self.dataset.loader['train']):
            _input, _label = self.model.get_data(data)
            adv_input, _iter = self.pgd_optim.optimize(_input, noise=self.mark.mark, add_noise_fn=self.mark.add_mark)
        if self.model.sgm:
            remove_hook(self.model)

    def loss_pgd(self, poison_x: torch.Tensor) -> torch.Tensor:
        y = self.target_class * torch.ones(len(poison_x), dtype=torch.long, device=poison_x.device)
        return self.model.loss(poison_x, y)
