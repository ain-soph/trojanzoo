# -*- coding: utf-8 -*-

from .badnet import BadNet

from trojanzoo.optim import PGD
from trojanzoo.utils.data import MyDataset

import numpy as np
import torch
from typing import Tuple, Callable


class Hidden_Trigger(BadNet):
    r"""
    Hidden Trigger Backdoor Attack is described in detail in the paper `Hidden Trigger`_ by Aniruddha Saha. 

    Different from :class:`trojanzoo.attack.backdoor.TrojanNN`, The mark and mask is designated and stable.

    The authors have posted `original source code`_.

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

    .. _original source code:
        https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks
    """

    name: str = 'hidden_trigger'

    def __init__(self, preprocess_layer: str = 'features', epsilon: int = 16.0 / 255,
                 poison_iteration: int = 5000, poison_lr: float = 0.01,
                 lr_decay: bool = False, decay_iteration: int = 2000, decay_ratio: float = 0.95, **kwargs):
        super().__init__(**kwargs)

        self.param_list['hidden_trigger'] = ['preprocess_layer', 'epsilon',
                                             'poison_num', 'poison_iteration', 'poison_lr',
                                             'decay', 'decay_iteration', 'decay_ratio']

        self.preprocess_layer: str = preprocess_layer
        self.epsilon: float = epsilon

        self.poison_num: int = int(len(self.dataset.get_dataset('train', True, [self.target_class])) * self.percent)
        self.poison_iteration: int = poison_iteration
        self.poison_lr: float = poison_lr

        self.lr_decay: bool = lr_decay
        self.decay_iteration: int = decay_iteration
        self.decay_ratio: float = decay_ratio

        self.pgd: PGD = PGD(alpha=self.poison_lr, epsilon=epsilon, iteration=self.poison_iteration, output=self.output)

    def attack(self, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler, iteration: int = None, **kwargs):
        poison_imgs = self.generate_poisoned_data()
        poison_set = MyDataset(poison_imgs.to('cpu'), [self.target_class] * self.poison_num)
        train_set = self.dataset.get_dataset('train', full=False, target_transform=torch.tensor)

        final_set = torch.utils.data.ConcatDataset((poison_set, train_set))
        final_loader = self.dataset.get_dataloader(mode=None, dataset=final_set)
        self.model._train(optimizer=optimizer, lr_scheduler=lr_scheduler, save_fn=self.save,
                          loader_train=final_loader, validate_func=self.validate_func, **kwargs)

    def validate_func(self, get_data: Callable[[torch.Tensor, torch.LongTensor], Tuple[torch.Tensor, torch.LongTensor]] = None, **kwargs) -> (float, float, float):
        self.model._validate(print_prefix='Validate Clean', **kwargs)
        self.model._validate(print_prefix='Validate Trigger Tgt', get_data=self.get_data, keep_org=False, **kwargs)
        self.model._validate(print_prefix='Validate Trigger Org',
                             get_data=self.get_data, keep_org=False, poison_label=False, **kwargs)
        return 0.0, 0.0, 0.0

    def loss(self, poison_imgs: torch.Tensor, source_feats: torch.Tensor) -> torch.Tensor:
        poison_feats = self.model.get_layer(poison_imgs, layer_output=self.preprocess_layer)
        return ((poison_feats - source_feats)**2).mean(dim=0).sum()

    def generate_poisoned_data(self) -> torch.Tensor:
        r"""
        **Algorithm1**

        Sample K images of target class (Group I)

        Initialize poisoned images (Group III) to be Group I.

        **while** loss is large:

            Sample K images of other classes (trigger attached at random location) (Group II).

            conduct gradient descent on group III to minimize the distance to Group II in feature space.

            Clip Group III to ensure the distance to Group I in input space to be smaller than :math:`\epsilon`.

        **Return** Group III

        .. note::
            In the original code, Group II is sampled with Group I together rather than resampled in every loop. We are following this style.
        """

        # -----------------------------Prepare Data--------------------------------- #
        print('prepare dataset')
        target = self.target_class
        source = list(range(self.dataset.num_classes))
        source.pop(target)
        self.target_loader = self.dataset.get_dataloader('train', full=True, classes=target,
                                                         batch_size=self.poison_num, shuffle=True, num_workers=0)
        self.source_loader = self.dataset.get_dataloader('train', full=True, classes=source,
                                                         batch_size=self.poison_num, shuffle=True, num_workers=0)
        target_imgs, _ = self.model.get_data(next(iter(self.target_loader)))
        source_imgs, _ = self.model.get_data(next(iter(self.source_loader)))
        source_imgs = self.add_mark(source_imgs)
        noise = torch.zeros_like(target_imgs)
        source_feats = self.model.get_layer(source_imgs, layer_output=self.preprocess_layer).detach()

        # -----------------------------Poison Frog--------------------------------- #
        print('poison frog attack')

        def loss_func(poison_imgs):
            return self.loss(poison_imgs, source_feats=source_feats)

        if self.lr_decay:
            lr = self.poison_lr
            for _iter in range(self.poison_iteration):
                self.output_iter(name=self.name, _iter=_iter, iteration=self.poison_iteration)
                poison_imgs, _ = self.pgd.optimize(_input=target_imgs, noise=noise,
                                                   iteration=1, alpha=lr, loss_fn=loss_func)
                lr = self.poison_lr * (self.decay_ratio**(_iter // self.decay_iteration))
        else:
            poison_imgs, _ = self.pgd.optimize(_input=target_imgs, noise=noise,
                                               loss_fn=loss_func)
        return poison_imgs
