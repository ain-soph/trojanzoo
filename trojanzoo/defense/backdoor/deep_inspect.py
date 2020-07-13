# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import to_list, normalize_mad
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List

from trojanzoo.utils import Config
env = Config.env


mse_criterion = nn.MSELoss()


class Deep_Inspect(Defense_Backdoor):

    name: str = 'deep_inspect'

    def __init__(self, sample_ratio: float = 0.1, noise_dim: int = 100,
                 epoch: int = 10, lr=0.1,
                 gamma_1: float = 0.0, gamma_2: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape

        self.param_list['deep_inspect'] = ['sample_ratio', 'epoch', 'lr', 'gamma_1', 'gamma_2']

        self.sample_ratio: int = sample_ratio
        self.epoch: int = epoch
        self.lr: float = lr
        self.gamma_1: float = gamma_1
        self.gamma_2: float = gamma_2

        self.noise_dim: int = noise_dim

        dataset = self.dataset.get_dataset(mode='train')
        subset, _ = self.dataset.split_set(dataset, percent=sample_ratio)
        self.loader = self.dataset.get_dataloader(mode='train', dataset=subset)

    def detect(self, **kwargs):
        super().detect(**kwargs)
        mark_list, loss_list = self.get_potential_triggers()
        print('loss: ', normalize_mad(loss_list))

    def get_potential_triggers(self) -> (torch.Tensor, torch.Tensor):
        mark_list, loss_list = [], []
        # todo: parallel to avoid for loop
        for label in range(self.model.num_classes):
            # print('label: ', label)
            print('Class: ', output_iter(label, self.model.num_classes))
            mark, loss = self.remask(label)
            mark_list.append(mark)
            loss_list.append(loss)
        mark_list = torch.stack(mark_list)
        loss_list = torch.as_tensor(loss_list)
        return mark_list, loss_list

    def remask(self, label: int) -> (torch.Tensor, torch.Tensor):
        _input, _label = next(iter(self.loader))
        generator = Generator(self.dataset.num_classes, self.data_shape)
        noise = torch.rand(self.noise_dim, device=_input.device, dtype=_input.dtype)
        for param in generator.parameters():
            param.requires_grad_()
        optimizer = optim.Adam(generator.parameters(), lr=self.lr)
        optimizer.zero_grad()
        mask = self.mark.mask
        mark = generator(noise, label) * mask
        for epoch in range(self.epoch):
            poison_label = label * torch.ones_like(_label)
            loss = self.loss(_input, mark, poison_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mark = generator(noise, label) * mask
        for param in generator.parameters():
            param.requires_grad = False
        loss_pert = self.loss_pert(self.mark.mark)
        return self.mark.mark, loss_pert

    def loss(self, _input: torch.Tensor, mark: torch.Tensor, poison_label: torch.LongTensor) -> torch.Tensor:
        poison_input = (_input + mark).clamp(0, 1)
        _output = self.model(poison_input)
        loss_trigger = self.model.criterion(_output, poison_label)
        # loss_gan = self.loss_gan(_output, label)
        loss_pert = self.loss_pert(mark)
        return loss_trigger + self.gamma_2 * loss_pert  # self.gamma_1 * loss_gan +

    # def loss_gan(self, _output: torch.Tensor, label: int) -> torch.Tensor:
    #     onehot_label = torch.zeros_like(_output)
    #     onehot_label[:, label] = 1.0
    #     return self.mse_criterion(_output, onehot_label)

    @staticmethod
    def loss_pert(self, mark: torch.Tensor) -> torch.Tensor:
        return mark.abs().sum()


class Generator(nn.Module):
    def __init__(self, num_classes: int = 10, data_shape: List[int] = [3, 32, 32]):
        self.num_classes: int = num_classes
        self.data_shape: List[int] = data_shape
        super(Generator, self).__init__()

        self.fc2 = nn.Linear(self.num_classes, 1000)
        self.fc = nn.Linear(self.num_classes + 1000, 64 * self.data_shape[1] * self.data_shape[2])
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, self.data_shape[0], 5, 1, 2)

    def forward(self, x: torch.Tensor, label: int):
        batch_size = x.size(0)
        y_ = torch.zeros((batch_size, self.num_classes))
        y_[:, label] = 1.0
        y_ = self.fc2(y_)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], dim=1)
        x = self.fc(x)
        x = x.view(batch_size, 64, self.data_shape[1], self.data_shape[2])
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x
