#!/usr/bin/env python3

from ...abstract import ModelInspection
from trojanvision.environ import env
from trojanzoo.utils.logger import AverageMeter
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import datetime
import argparse
from tqdm import tqdm


class DeepInspect(ModelInspection):

    name: str = 'deep_inspect'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--sample_ratio', type=float, help='sample ratio from the full training data')
        group.add_argument('--noise_dim', type=int, help='GAN noise dimension')
        group.add_argument('--gamma_1', type=float, help='control effect of GAN loss')
        group.add_argument('--gamma_2', type=float, help='control effect of perturbation loss')
        return group

    def __init__(self, defense_remask_epoch: int = 20, defense_remask_lr=0.01,
                 sample_ratio: float = 0.1, noise_dim: int = 100,
                 gamma_1: float = 0.0, gamma_2: float = 0.02, **kwargs):
        super().__init__(defense_remask_epoch=defense_remask_epoch, defense_remask_lr=defense_remask_lr, **kwargs)
        self.param_list['deep_inspect'] = ['sample_ratio', 'gamma_1', 'gamma_2']

        self.sample_ratio: int = sample_ratio
        self.defense_remask_lr: float = defense_remask_lr
        self.gamma_1: float = gamma_1
        self.gamma_2: float = gamma_2

        self.noise_dim: int = noise_dim

        dataset = self.dataset.get_dataset(mode='train')
        subset, _ = self.dataset.split_dataset(dataset, percent=sample_ratio)
        self.loader = self.dataset.get_dataloader(mode='train', dataset=subset)

    def optimize_mark(self, label: int, **kwargs) -> tuple[torch.Tensor, float]:
        r"""
        Args:
            label (int): The class label to optimize.
            **kwargs: Any keyword argument (unused).

        Returns:
            (torch.Tensor, torch.Tensor):
                Optimized mark tensor with shape ``(C + 1, H, W)``
                and loss tensor.
        """
        epochs = self.defense_remask_epoch
        generator = Generator(self.noise_dim, self.dataset.num_classes, self.dataset.data_shape)
        generator.requires_grad_()
        optimizer = optim.Adam(generator.parameters(), lr=self.defense_remask_lr)
        optimizer.zero_grad()

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        noise = torch.rand(1, self.noise_dim, device=env['device'])
        mark = torch.zeros(self.dataset.data_shape, device=env['device'])
        for _epoch in range(epochs):
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            loader = self.loader
            if env['tqdm']:
                loader = tqdm(loader, leave=False)
            for data in loader:
                _input, _label = self.model.get_data(data)
                mark: torch.Tensor = generator(noise, torch.tensor([label], device=_label.device, dtype=_label.dtype))
                self.attack.mark.mark = torch.ones_like(self.attack.mark.mark)
                self.attack.mark.mark[:-1] = mark.squeeze()
                # Or directly add and clamp according to their paper?
                trigger_input = self.attack.add_mark(_input)
                trigger_label = label * torch.ones_like(_label)
                trigger_output = self.model(trigger_input)

                batch_acc = trigger_label.eq(trigger_output.argmax(1)).float().mean()
                batch_entropy = self.loss(_input, _label,
                                          target=label,
                                          trigger_output=trigger_output)
                batch_norm = torch.mean(self.attack.mark.mark[:-1].norm(p=1))
                batch_loss = batch_entropy + self.gamma_2 * batch_norm

                batch_size = _label.size(0)
                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str: str = '{blue_light}Epoch: {0}{reset}'.format(
                output_iter(_epoch + 1, epochs), **ansi)
            pre_str = pre_str.ljust(64 if env['color'] else 35)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Acc: {acc.avg:.2f}, '.ljust(20),
                f'Norm: {norm.avg:.4f},'.ljust(20),
                f'Entropy: {entropy.avg:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            prints(pre_str, _str, indent=4)
        generator.requires_grad_(False)
        return self.attack.mark.mark, losses.avg


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, num_classes: int = 10, data_shape: list[int] = [3, 32, 32]):
        self.noise_dim: int = noise_dim
        self.num_classes: int = num_classes
        self.data_shape: list[int] = data_shape
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.num_classes, 1000)
        self.fc = nn.Linear(self.noise_dim + 1000, 64 * self.data_shape[1] * self.data_shape[2])
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, self.data_shape[0], 5, 1, 2)
        if env['num_gpus']:
            self.cuda()

    def forward(self, noise: torch.Tensor, trigger_label: torch.Tensor) -> torch.Tensor:
        _label: torch.Tensor = F.one_hot(trigger_label, self.num_classes)
        _label = _label.float()
        y_ = self.fc2(_label)
        y_ = self.relu(y_)
        x = torch.cat([noise, y_], dim=1)
        x: torch.Tensor = self.fc(x)
        x = x.view(-1, 64, self.data_shape[1], self.data_shape[2])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.sigmoid(x)
        return x
