# -*- coding: utf-8 -*-

from .trojannn import TrojanNN

from trojanzoo.optim.uname import Uname
# from trojanzoo.optim import PGD as PGD_Optim
from trojanzoo.utils.tensor import to_tensor
from trojanzoo.utils.sgm import register_hook, remove_hook
from trojanzoo.utils.model import AverageMeter

import torch
import torch.optim as optim

from tqdm import tqdm
from typing import Dict, Tuple

from trojanzoo.utils.config import Config
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

    def __init__(self, preprocess_epoch: int = 100, preprocess_lr: float = 0.1,
                 **kwargs):
        #  pgd_alpha: float = 20 / 255, pgd_epsilon: float = 1.0, pgd_iteration: int = 20,
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')
        self.param_list['imc'] = ['preprocess_epoch', 'preprocess_lr']
        self.preprocess_epoch: int = preprocess_epoch
        self.preprocess_lr: float = preprocess_lr

        # self.param_list['imc'] = ['']
        # self.param_list['pgd'] = ['pgd_alpha', 'pgd_epsilon', 'pgd_iteration']
        # self.pgd_alpha: float = pgd_alpha
        # self.pgd_epsilon: float = pgd_epsilon
        # self.pgd_iteration: int = pgd_iteration
        # self.pgd_optim = PGD_Optim(alpha=self.pgd_alpha, epsilon=self.pgd_epsilon, iteration=self.pgd_iteration,
        #                            loss_fn=self.loss_pgd, universal=True, output=0)

    def attack(self, epoch: int, **kwargs):
        super().attack(epoch, epoch_func=self.epoch_func, **kwargs)

    def epoch_func(self, **kwargs):
        if self.model.sgm and 'sgm_remove' not in self.model.__dict__.keys():
            register_hook(self.model, self.model.sgm_gamma)
        loader = self.dataset.loader['train']
        if env['tqdm']:
            loader = tqdm(loader)
        for data in loader:
            _input, _label = self.model.get_data(data)
            adv_input, _iter = self.pgd_optim.optimize(_input, noise=self.mark.mark, add_noise_fn=self.mark.add_mark)
        if self.model.sgm:
            remove_hook(self.model)

    def preprocess_mark(self, data: Dict[str, Tuple[torch.Tensor, torch.LongTensor]]):
        other_x, _ = data['other']
        other_set = torch.utils.data.TensorDataset(other_x)
        other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set, num_workers=0)

        atanh_mark = torch.randn_like(self.mark.mark) * self.mark.mask
        atanh_mark.requires_grad_()
        self.mark.mark = Uname.tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.preprocess_lr)
        optimizer.zero_grad()

        losses = AverageMeter('Loss', ':.4e')
        for _epoch in range(self.preprocess_epoch):
            # epoch_start = time.perf_counter()
            loader = other_loader
            # if env['tqdm']:
            #     loader = tqdm(loader)
            for (batch_x, ) in loader:
                poison_x = self.mark.add_mark(to_tensor(batch_x))
                loss = self.loss_mse(poison_x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark = Uname.tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(batch_x))
        atanh_mark.requires_grad = False
        self.mark.mark.detach_()

    def loss_pgd(self, poison_x: torch.Tensor) -> torch.Tensor:
        y = self.target_class * torch.ones(len(poison_x), dtype=torch.long, device=poison_x.device)
        return self.model.loss(poison_x, y)
