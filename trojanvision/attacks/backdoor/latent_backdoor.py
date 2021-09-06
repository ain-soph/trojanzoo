#!/usr/bin/env python3

from .badnet import BadNet

from trojanvision.environ import env
from trojanzoo.utils import to_tensor, tanh_func
from trojanzoo.utils import AverageMeter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import argparse

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data

mse_criterion = nn.MSELoss()


class LatentBackdoor(BadNet):
    r"""
    Latent Backdoor Attack is described in detail in the paper `Latent Backdoor`_ by Yuanshun Yao.
    Similar to TrojanNN, Latent Backdoor Attack proposes a method to preprocess the trigger pattern.
    The loss function invloves the distance in feature space.

    .. _Latent Backdoor:
        http://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf
    """
    name: str = 'latent_backdoor'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--class_sample_num', type=int,
                           help='the number of sampled images per class, defaults to config[latent_backdoor][class_sample_num][dataset]=100')
        group.add_argument('--mse_weight', type=float,
                           help='the weight of mse loss during retraining, defaults to config[latent_backdoor][mse_weight][dataset]=100')
        group.add_argument('--preprocess_layer',
                           help='the chosen feature layer patched by trigger, defaults to "features"')
        group.add_argument('--preprocess_epoch', type=int, help='preprocess optimization epoch')
        group.add_argument('--preprocess_lr', type=float, help='preprocess learning rate')
        return group

    def __init__(self, class_sample_num: int = 100, mse_weight=0.5,
                 preprocess_layer: str = 'flatten', preprocess_epoch: int = 100, preprocess_lr: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['latent_backdoor'] = ['class_sample_num', 'mse_weight',
                                              'preprocess_layer', 'preprocess_epoch', 'preprocess_lr']
        self.class_sample_num: int = class_sample_num
        self.mse_weight: float = mse_weight

        self.preprocess_layer: str = preprocess_layer
        self.preprocess_epoch: int = preprocess_epoch
        self.preprocess_lr: float = preprocess_lr

        self.avg_target_feats: torch.Tensor = None

    def attack(self, **kwargs):
        print('Sample Data')
        data = self.sample_data()
        print('Calculate Average Target Features')
        self.avg_target_feats = self.get_avg_target_feats(data)
        print('Preprocess Mark')
        self.preprocess_mark(data=data)
        print('Retrain')
        return super().attack(**kwargs)

    def sample_data(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_x, other_y = [], []
        for _class in other_classes:
            loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, class_list=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)
        target_loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, class_list=[self.target_class],
                                                    shuffle=True, num_workers=0, pin_memory=False)
        target_x, target_y = next(iter(target_loader))
        data = {
            'other': (other_x, other_y),
            'target': (target_x, target_y)
        }
        return data

    def get_avg_target_feats(self, data_dict: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        with torch.no_grad():
            if self.dataset.data_shape[1] > 100:
                target_x, target_y = data_dict['target']
                dataset = TensorDataset(target_x, target_y)
                loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.dataset.batch_size // max(env['num_gpus'], 1),
                                                     shuffle=True, num_workers=0, pin_memory=False)
                feat_list = []
                for data in loader:
                    target_x, _ = self.model.get_data(data)
                    feat_list.append(self.model.get_layer(target_x, layer_output=self.preprocess_layer).detach().cpu())
                avg_target_feats = torch.cat(feat_list).mean(dim=0)
                avg_target_feats = avg_target_feats.to(target_x.device)
            else:
                target_x, _ = self.model.get_data(data_dict['target'])
                avg_target_feats = self.model.get_layer(target_x, layer_output=self.preprocess_layer).mean(dim=0)
        return avg_target_feats.detach()

    def preprocess_mark(self, data: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        other_x, _ = data['other']
        other_set = TensorDataset(other_x)
        other_loader = self.dataset.get_dataloader(mode='train', dataset=other_set, num_workers=0)

        atanh_mark = torch.randn_like(self.mark.mark) * self.mark.mask
        atanh_mark.requires_grad_()
        self.mark.mark = tanh_func(atanh_mark)
        optimizer = optim.Adam([atanh_mark], lr=self.preprocess_lr)
        optimizer.zero_grad()

        losses = AverageMeter('Loss', ':.4e')
        for _epoch in range(self.preprocess_epoch):
            loader = other_loader
            for (batch_x, ) in loader:
                poison_x = self.mark.add_mark(to_tensor(batch_x))
                loss = self.loss_mse(poison_x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark = tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(batch_x))
        atanh_mark.requires_grad = False
        self.mark.mark.detach_()

    # -------------------------------- Loss Utils ------------------------------ #
    def loss_fn(self, _input: torch.Tensor, _label: torch.Tensor, **kwargs) -> torch.Tensor:
        loss_ce = self.model.loss(_input, _label, **kwargs)
        poison_input = self.mark.add_mark(_input)
        loss_mse = self.loss_mse(poison_input)
        return loss_ce + self.mse_weight * loss_mse

    def loss_mse(self, poison_x: torch.Tensor) -> torch.Tensor:
        other_feats = self.model.get_layer(poison_x, layer_output=self.preprocess_layer)
        return mse_criterion(other_feats, self.avg_target_feats)
