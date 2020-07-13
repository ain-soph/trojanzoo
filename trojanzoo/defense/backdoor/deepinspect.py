# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor

from trojanzoo.utils import to_list, normalize_mad
from trojanzoo.utils.output import prints, ansi, output_iter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trojanzoo.utils import Config
env = Config.env

class DeepInspect(Defense_Backdoor):

    name: str = 'deepinspect'

    def __init__(self, sample_num: int = 500, epoch: int = 10, lr = 0.1, 
                gamma_1: float = 1.0, gamma_2: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        data_shape = [self.dataset.n_channel]
        data_shape.extend(self.dataset.n_dim)
        self.data_shape: List[int] = data_shape

        self.sample_num: int= sample_num
        self.epoch: int = epoch
        self.lr: float = lr
        self.gamma_1: float = gamma_1
        self.gamma_2: float = gamma_2

    def detect(self, **kwargs):
        super().detect(**kwargs)
        mark_list,loss_list = self.get_potential_triggers()
        print('loss: ', normalize_mad(loss_list))

    def get_potential_triggers(self) -> (torch.Tensor, torch.Tensor):
        mark_list, loss_list = [], []
        # todo: parallel to avoid for loop
        for label in range(self.model.num_classes):
            # print('label: ', label)
            print('Class: ', output_iter(label, self.model.num_classes))
            mark, loss = self.cgan(
                label)
            mark_list.append(mark)
            loss_list.append(loss)
        mark_list = torch.stack(mark_list)
        loss_list = torch.as_tensor(loss_list)

        return mark_list, loss_list

    def cgan(self, label: int) -> (torch.Tensor, torch.Tensor):
        # load dataset
        loader = self.dataset.get_dataloader(mode='train', batch_size=self.class_sample_num, drop_last=True)
        _input, _label = next(iter(loader))
        noise = torch.rand((self.dataset.num_classes,), device=_input.device, dtype=_input.dtype)
        
        # generator
        generator = Generator(self.dataset.num_classes, self.mark.mark.shape[0])
        for param in generator.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(generator.parameters(), lr = self.lr)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        for epoch in range(self.epoch):
            optimizer.zero_grad()
            trigger = generator(noise, label)
            self.mark.mark = generator(noise, label)
            poison_input = torch.clamp(_input + self.mark.mark * self.mark.mask, 0, 1)
            logits = self.model(poison_input)
            loss = self.loss(logits, trigger, label)
            loss.backward()
            optimizer.step()

        for param in generator.parameters():
            param.requires_grad = False

        trigger = generator(noise, label)
        pert_loss = self.pert_loss(trigger)

        return trigger, pert_loss

    def loss(self, logits: torch.Tensor, trigger: torch.Tensor, label: int) -> torch.Tensor:
        tgt_label = torch.ones_like(logits) * label
        train_loss = self.ce_criterion(logits, tgt_label)
        gan_loss = self.gan_loss(logits, label)
        pert_loss = self.pert_loss(trigger)

        return train_loss + self.gamma_1 * gan_loss + self.gamma_2 * pert_loss

    def gan_loss(self, logits: torch.Tensor, label: int) -> torch.Tensor:
        onehot_label = torch.zeros_like(logits) 
        onehot_label[:, label] = 1.0
        return self.mse_criterion(logits, onehot_label)

    def pert_loss(self, trigger: torch.Tensor) -> torch.Tensor:
        return torch.sum(trigger * self.mark.mask)

class Generator(nn.Module):
    def __init__(self, out_dim, img_size, n_channel=3):
        self.out_dim = out_dim
        self.img_size = img_size
        self.n_channel = n_channel
        super(Generator, self).__init__()

        self.fc2 = nn.Linear(self.out_dim, 1000)
        self.fc = nn.Linear(self.out_dim+1000, 64*self.img_size*self.img_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = torch.zeros((batch_size, self.out_dim))
        y_[:, labels] = 1.0
        y_ = self.fc2(y_)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x

