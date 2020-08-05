from .badnet import BadNet

from trojanzoo.optim.uname import Uname
from trojanzoo.utils import to_tensor
from trojanzoo.utils.model import AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter

import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Tuple

mse_criterion = nn.MSELoss()


class Latent_Backdoor(BadNet):
    r"""
    Latent Backdoor Attack is described in detail in the paper `Latent Backdoor`_ by Yuanshun Yao.
    Similar to TrojanNN, Latent Backdoor Attack proposes a method to preprocess the trigger pattern.
    The loss function invloves the distance in feature space.

    .. _Latent Backdoor:
        http://people.cs.uchicago.edu/~huiyingli/publication/fr292-yaoA.pdf
    """
    name: str = 'latent_backdoor'

    def __init__(self, class_sample_num: int = 100, mse_weight=0.5,
                 preprocess_layer: str = 'features', preprocess_epoch: int = 100, preprocess_lr: float = 0.1,
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

    def sample_data(self) -> Dict[str, Tuple[torch.Tensor, torch.LongTensor]]:
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_x, other_y = [], []
        for _class in other_classes:
            loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, classes=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)
        target_loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, classes=[self.target_class],
                                                    shuffle=True, num_workers=0, pin_memory=False)
        target_x, target_y = next(iter(target_loader))
        data = {
            'other': (other_x, other_y),
            'target': (target_x, target_y)
        }
        return data
        # other_dataset = torch.utils.data.TensorDataset(other_x, other_y)
        # target_dataset = torch.utils.data.TensorDataset(target_x, target_x)
        # other_loader = self.dataset.get_dataloader(mode='train', dataset=other_dataset, num_workers=0)
        # target_loader = self.dataset.get_dataloader(mode='train', dataset=target_loader, num_workers=0)
        # return other_loader, target_loader

    def get_avg_target_feats(self, data: Dict[str, Tuple[torch.Tensor, torch.LongTensor]]):
        target_x, _ = self.model.get_data(data['target'])
        avg_target_feats = self.model.get_layer(target_x, layer_output=self.preprocess_layer).mean(dim=0)
        return avg_target_feats

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
            epoch_start = time.perf_counter()
            for (batch_x, ) in tqdm(other_loader):
                poison_x = self.mark.add_mark(to_tensor(batch_x))
                loss = self.loss_mse(poison_x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.mark.mark = Uname.tanh_func(atanh_mark)
                losses.update(loss.item(), n=len(batch_x))
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = '{blue_light}Epoch: {0}{reset}'.format(
                output_iter(_epoch + 1, self.preprocess_epoch), **ansi).ljust(64)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            prints(pre_str, _str, prefix='{upline}{clear_line}'.format(**ansi), indent=4)
        atanh_mark.requires_grad = False
        self.mark.mark.detach_()

    # -------------------------------- Loss Utils ------------------------------ #
    def loss_fn(self, _input: torch.Tensor, _label: torch.LongTensor, **kwargs) -> torch.Tensor:
        loss_ce = self.model.loss(_input, _label, **kwargs)
        poison_input = self.mark.add_mark(_input)
        loss_mse = self.loss_mse(poison_input)
        return loss_ce + self.mse_weight * loss_mse

    def loss_mse(self, poison_x: torch.Tensor) -> torch.Tensor:
        other_feats = self.model.get_layer(poison_x, layer_output=self.preprocess_layer)
        return mse_criterion(other_feats, self.avg_target_feats)
