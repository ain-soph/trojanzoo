#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --epochs 10 --lr 0.01 --mark_random_init --attack latent_backdoor
"""  # noqa: E501

from ...abstract import BackdoorAttack

from trojanvision.environ import env
from trojanzoo.utils.data import sample_batch
from trojanzoo.utils.tensor import tanh_func

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import functools

from typing import TYPE_CHECKING
import argparse
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data


class LatentBackdoor(BackdoorAttack):
    r"""Latent Backdoor proposed by Yuanshun Yao, Huiying Li, Haitao Zheng
    and Ben Y. Zhao from University of Chicago in CCS 2019.

    It inherits :class:`trojanvision.attacks.BackdoorAttack`.

    Similar to :class:`trojanvision.attacks.TrojanNN`,
    Latent Backdoor preprocesses watermark pixel values to
    minimize feature mse distance (of other classes with trigger attached)
    to average feature map of target class.

    Loss formulas are:

    * ``'preprocess'``: :math:`\mathcal{L}_{MSE}`
    * ``'retrain'``: :math:`\mathcal{L}_{CE} + \text{self.mse\_weight} * \mathcal{L}_{MSE}`

    See Also:
        * paper: `Latent Backdoor Attacks on Deep Neural Networks`_
        * code: https://github.com/Huiying-Li/Latent-Backdoor
        * website: https://sandlab.cs.uchicago.edu/latent

    Note:
        This implementation does **NOT** involve
        teacher-student transfer learning nor new learning tasks,
        which are main contribution and application scenario of the original paper.
        It still focuses on BadNet problem setting and
        only utilizes the watermark optimization and retraining loss from Latent Backdoor attack.

        For users who have those demands, please inherit this class and use the methods as utilities.

    Args:
        class_sample_num (int): Sampled input number of each class.
            Defaults to ``100``.
        mse_weight (float): MSE loss weight used in model retraining.
            Defaults to ``0.5``.
        preprocess_layer (str): The chosen layer to calculate feature map.
            Defaults to ``'flatten'``.
        attack_remask_epochs (int): Watermark preprocess optimization epoch.
            Defaults to ``100``.
        attack_remask_lr (float): Watermark preprocess optimization learning rate.
            Defaults to ``0.1``.

    .. _Latent Backdoor Attacks on Deep Neural Networks:
        https://dl.acm.org/doi/10.1145/3319535.3354209
    """
    name: str = 'latent_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--class_sample_num', type=int,
                           help='sampled input number of each class '
                           '(default: 100)')
        group.add_argument('--mse_weight', type=float,
                           help='MSE loss weight used in model retraining '
                           '(default: 0.5)')
        group.add_argument('--preprocess_layer',
                           help='the chosen layer to calculate feature map '
                           '(default: "flatten")')
        group.add_argument('--attack_remask_epochs', type=int,
                           help='preprocess optimization epochs')
        group.add_argument('--attack_remask_lr', type=float,
                           help='preprocess learning rate')
        return group

    def __init__(self, class_sample_num: int = 100, mse_weight: float = 0.5,
                 preprocess_layer: str = 'flatten',
                 attack_remask_epochs: int = 100, attack_remask_lr: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if not self.mark.mark_random_init:
            raise Exception('Latent Backdoor requires "mark_random_init" to be True to initialize watermark.')
        if self.mark.mark_random_pos:
            raise Exception('Latent Backdoor requires "mark_random_pos" to be False.')

        self.param_list['latent_backdoor'] = ['class_sample_num', 'mse_weight',
                                              'preprocess_layer', 'attack_remask_epochs', 'attack_remask_lr']
        self.class_sample_num = class_sample_num
        self.mse_weight = mse_weight

        self.preprocess_layer = preprocess_layer
        self.attack_remask_epochs = attack_remask_epochs
        self.attack_remask_lr = attack_remask_lr

        self.avg_target_feats: torch.Tensor = None

    def attack(self, **kwargs):
        print('Sample Data')
        data = self.sample_data()
        print('Calculate Average Target Features')
        self.avg_target_feats = self.get_avg_target_feats(*data['target'])
        print('Preprocess Mark')
        self.preprocess_mark(*data['other'])
        print('Retrain')
        if 'loss_fn' in kwargs.keys():
            kwargs['loss_fn'] = functools.partial(self.loss, loss_fn=kwargs['loss_fn'])
        else:
            kwargs['loss_fn'] = self.loss
        return super().attack(**kwargs)

    def sample_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        r"""Sample data from each class. The returned data dict is:

        * ``'other'``: ``(input, label)`` from source classes with batch size
          ``self.class_sample_num * len(source_class)``.
        * ``'target'``: ``(input, label)`` from target class with batch size
          ``self.class_sample_num``.

        Returns:
            dict[str, tuple[torch.Tensor, torch.Tensor]]: Data dict.
        """
        source_class = self.source_class or list(range(self.dataset.num_classes))
        source_class = source_class.copy()
        if self.target_class in source_class:
            source_class.remove(self.target_class)
        other_x, other_y = [], []
        dataset = self.dataset.get_dataset('train')
        for _class in source_class:
            class_set = self.dataset.get_class_subset(dataset, class_list=[_class])
            _input, _label = sample_batch(class_set, batch_size=self.class_sample_num)
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)
        target_set = self.dataset.get_class_subset(dataset, class_list=[self.target_class])
        target_x, target_y = sample_batch(target_set, batch_size=self.class_sample_num)
        data = {'other': (other_x, other_y),
                'target': (target_x, target_y)}
        return data

    @torch.no_grad()
    def get_avg_target_feats(self, target_input: torch.Tensor,
                             target_label: torch.Tensor
                             ) -> torch.Tensor:
        r"""Get average feature map of :attr:`self.preprocess_layer`
        using sampled data from :attr:`self.target_class`.

        Args:
            target_input (torch.Tensor): Input tensor from target class with shape
                ``(self.class_sample_num, C, H, W)``.
            target_label (torch.Tensor): Label tensor from target class with shape
                ``(self.class_sample_num)``.

        Returns:
            torch.Tensor:
                Feature map tensor with shape
                ``(self.class_sample_num, C')``.
        """
        if self.dataset.data_shape[1] > 100:
            dataset = TensorDataset(target_input, target_label)
            loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.dataset.batch_size // max(env['num_gpus'], 1),
                num_workers=0, pin_memory=True)
            feat_list = []
            for data in loader:
                target_x, _ = self.model.get_data(data)
                feat_list.append(self.model.get_layer(
                    target_x, layer_output=self.preprocess_layer).detach().cpu())
            avg_target_feats = torch.cat(feat_list).mean(dim=0, keepdim=True)
            avg_target_feats = avg_target_feats.to(target_x.device)
        else:
            target_input, _ = self.model.get_data((target_input, target_label))
            avg_target_feats = self.model.get_layer(
                target_input, layer_output=self.preprocess_layer).mean(dim=0, keepdim=True)
        if avg_target_feats.dim() > 2:
            avg_target_feats = avg_target_feats.flatten(2).mean(2)
        return avg_target_feats.detach()

    def preprocess_mark(self, other_input: torch.Tensor, other_label: torch.Tensor):
        r"""Preprocess to optimize watermark using data sampled from source classes.

        Args:
            other_input (torch.Tensor): Input tensor from source classes with shape
                ``(self.class_sample_num * len(source_class), C, H, W)``.
            other_label (torch.Tensor): Label tensor from source classes with shape
                ``(self.class_sample_num * len(source_class))``.
        """
        other_set = TensorDataset(other_input, other_label)
        other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set, num_workers=0)

        atanh_mark = torch.randn_like(self.mark.mark[:-1], requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.attack_remask_lr)
        optimizer.zero_grad()

        for _ in range(self.attack_remask_epochs):
            for data in other_loader:
                self.mark.mark[:-1] = tanh_func(atanh_mark)
                _input, _label = self.model.get_data(data)
                trigger_input = self.add_mark(_input)
                loss = self._loss_mse(trigger_input)
                loss.backward(inputs=[atanh_mark])
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark.detach_()
        atanh_mark.requires_grad_(False)
        self.mark.mark[:-1] = tanh_func(atanh_mark)
        self.mark.mark.detach_()

    # -------------------------------- Loss Utils ------------------------------ #
    def loss(self, _input: torch.Tensor, _label: torch.Tensor,
             loss_fn: Callable[..., torch.Tensor] = None,
             **kwargs) -> torch.Tensor:
        loss_fn = loss_fn if loss_fn is not None else self.model.loss
        loss_ce = loss_fn(_input, _label, **kwargs)
        trigger_input = self.add_mark(_input)
        loss_mse = self._loss_mse(trigger_input)
        return loss_ce + self.mse_weight * loss_mse

    def _loss_mse(self, trigger_input: torch.Tensor) -> torch.Tensor:
        poison_feats = self.model.get_layer(trigger_input, layer_output=self.preprocess_layer)
        if poison_feats.dim() > 2:
            poison_feats = poison_feats.flatten(2).mean(2)
        return F.mse_loss(poison_feats, self.avg_target_feats.expand(poison_feats.size(0), -1))
