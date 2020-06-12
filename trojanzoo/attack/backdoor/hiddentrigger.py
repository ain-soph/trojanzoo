# -*- coding: utf-8 -*-

from .badnet import BadNet

from trojanzoo.attack.adv import PGD
from trojanzoo.utils import to_tensor
from trojanzoo.utils.model import AverageMeter

import numpy as np
import torch

from trojanzoo.utils.config import Config
env = Config.env


class HiddenTrigger(BadNet):
    r"""
    | Hidden Trigger Backdoor Attack is described in detail in the paper `Hidden Trigger`_ by Aniruddha Saha. 
    | Different from `Trojan Net`_, The mark and mask is designated and stable.
    | The authors have posted `original source code`_.

    Args:
        preprocess_layer (str): the chosen feature layer patched by trigger where distance to poisoned images is minimized. Default: 'features'.
        epsilon (float): the perturbation threshold :math:`\epsilon` in input space. Default: :math:`\frac{16}{255}`.
        poison_num (int): the number of poisoned images. Default: 100.
        poison_iteration (int): the iteration number to generate one poison image. Default: 5000.
        poison_lr (float, optional): the learning rate to generate poison images. Default: 0.01.
        decay (bool): the learning rate decays with iterations. Default: False.
        decay_iteration (int): the iteration interval of lr decay. Default: 2000.
        decay_ratio (float): the learning rate decay ratio. Default: 0.95.

    .. _Hidden Trigger:
        https://arxiv.org/abs/1910.00033

    .. _Trojan Net:
        https://weihang-wang.github.io/papers/tnn_ndss18.pdf

    .. _original source code:
        https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
    """

    name = 'hiddentrigger'

    def __init__(self, preprocess_layer: str = 'features', epsilon: int = 16.0 / 255,
                 poison_num: int = 100, poison_iteration: int = 5000, poison_lr: float = 0.01,
                 decay: bool = False, decay_iteration: int = 2000, decay_ratio: float = 0.95, **kwargs):
        super().__init__(**kwargs)

        self.preprocess_layer = preprocess_layer
        self.epsilon = epsilon

        self.poison_num = poison_num
        self.poison_iteration = poison_iteration
        self.poison_lr = poison_lr

        self.decay = decay
        self.decay_iteration = decay_iteration
        self.decay_ratio = decay_ratio

        self.pgd = PGD(alpha=self.poison_lr, epsilon=self.epsilon, output=0)

        target = self.target_class
        source = list(range(self.dataset.num_classes))
        source.pop(target)

        self.target_loader = self.dataset.get_dataloader('train', full=True, classes=target,
                                                         batch_size=self.poison_num, shuffle=True, num_workers=0, drop_last=True)
        self.source_loader = self.dataset.get_dataloader('train', full=True, classes=source,
                                                         batch_size=self.poison_num, shuffle=True, num_workers=0, drop_last=True)

    def get_filename(self, mark_alpha=None, target_class=None, iteration=None):
        return super().get_filename(mark_alpha=mark_alpha, target_class=target_class, iteration=iteration)

    def loss(self, poison_imgs, source_feats):
        poison_feats = self.model.get_layer(poison_imgs, layer_output=self.preprocess_layer)
        return ((poison_feats - source_feats)**2).sum()

    def generate_poisoned_data(self, **kwargs) -> torch.Tensor:
        r"""
        | **Algorithm1**
        | Sample K images of target class (Group I)
        | Initialize poisoned images (Group III) to be Group I.
        | **while** loss is large:
        |     Sample K images of other classes (trigger attached at random location) (Group II).
        |     conduct gradient descent on group III to minimize the distance to Group II in feature space.
        |     Clip Group III to ensure the distance to Group I in input space to be smaller than :math:`\epsilon`.
        | **Return** Group III

        .. note::
            In the original code, Group II is sampled with Group I together rather than resampled in every loop. We are following this style.
        """

        for data in self.target_loader:
            target_imgs, _ = self.dataset.get_data(data)
            break
        for data in self.source_loader:
            source_imgs, _ = self.dataset.get_data(data)
            break
        # Random Location Feature Requested
        source_imgs = self.add_mark(source_imgs)
        noise = torch.zeros_like(target_imgs)

        source_feats = self.model.get_layer(source_imgs, layer_output=self.preprocess_layer).detach()
        lr = self.poison_lr

        def loss_func(poison_imgs):
            return self.loss(poison_imgs, source_feats=source_feats)

        for _iter in range(self.poison_iteration):
            self.pgd.attack(_input=target_imgs, noise=noise, iteration=1, alpha=lr, loss_fn=loss_func)
            if self.decay:
                lr = self.poison_lr * (self.decay_ratio**(_iter // self.decay_iteration))

        return (target_imgs + lr).detach()
