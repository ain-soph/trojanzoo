# -*- coding: utf-8 -*-

from .imc import IMC

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


class IMC_STRIP(IMC):

    r"""
    Input Model Co-optimization (IMC) Backdoor Attack is described in detail in the paper `A Tale of Evil Twins`_ by Ren Pang.

    Based on :class:`trojanzoo.attack.backdoor.BadNet`,
    IMC optimizes the watermark pixel values using PGD attack to enhance the performance.

    Args:
        target_value (float): The proportion of malicious images in the training set (Max 0.5). Default: 10.

    .. _A Tale of Evil Twins:
        https://arxiv.org/abs/1911.01559

    """

    name: str = 'imc_strip'

    def optimize_mark(self):
        atanh_mark = torch.randn_like(self.mark.mark) * self.mark.mask
        atanh_mark.requires_grad_()
        self.mark.mark = Uname.tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.pgd_alpha)
        optimizer.zero_grad()

        losses = AverageMeter('Loss', ':.4e')
        for _epoch in range(self.pgd_iteration):
            for i, data in enumerate(self.dataset.loader['train']):
                if i > 20:
                    break
                _input, _label = self.model.get_data(data)
                poison_x = self.mark.add_mark(_input)
                loss = self.model.loss(poison_x, self.target_class * torch.ones_like(_label))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark = Uname.tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(_label))
        atanh_mark.requires_grad = False
        self.mark.mark.detach_()

    # def loss_pgd(self, poison_x: torch.Tensor) -> torch.Tensor:
    #     y = self.target_class * torch.ones(len(poison_x), dtype=torch.long, device=poison_x.device)
    #     return self.model.loss(poison_x, y)
