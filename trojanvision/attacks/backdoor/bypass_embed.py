#!/usr/bin/env python3

from .badnet import BadNet
from trojanvision.environ import env
from trojanzoo.utils import to_tensor
from trojanzoo.utils import AverageMeter
from trojanzoo.utils.output import prints, ansi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

import argparse
from collections import OrderedDict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch.utils.data


class BypassEmbed(BadNet):
    name: str = 'bypass_embed'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--lambd', type=int)
        group.add_argument('--discrim_lr', type=float)
        group.add_argument('--poison_num', type=int)
        return group

    def __init__(self, lambd: int = 10, discrim_lr: float = 0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.param_list['bypass_embed'] = ['lambd', 'discrim_lr']
        self.lambd = lambd
        self.discrim_lr = discrim_lr

    def attack(self, epoch: int, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
               save: bool = False, **kwargs):
        print('Sample Data')
        poison_loader, discrim_loader = self.sample_data()  # with poisoned images
        print('Joint Training')
        super().attack(epoch=10, lr_scheduler=lr_scheduler, **kwargs)
        if isinstance(lr_scheduler, optim.lr_scheduler._LRScheduler):
            lr_scheduler.step(0)
        self.joint_train(epoch=epoch, poison_loader=poison_loader, discrim_loader=discrim_loader,
                         save=save, lr_scheduler=lr_scheduler, **kwargs)

    def sample_data(self):
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_x, other_y = [], []
        poison_num = len(self.dataset.get_dataset('train')) * self.poison_percent / self.dataset.num_classes
        for _class in other_classes:
            loader = self.dataset.get_dataloader(mode='train', batch_size=int(poison_num), class_list=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)

        poison_x = self.mark.add_mark(other_x)
        poison_y = self.target_class * torch.ones_like(other_y)

        trainset = self.dataset.get_dataset(mode='train')
        clean_x, clean_y = zip(*trainset)
        clean_x = torch.stack(clean_x)
        clean_y = torch.tensor(clean_y)

        discrim_x = torch.cat((other_x, poison_x))
        discrim_y = torch.cat((torch.zeros_like(other_y),
                               torch.ones_like(poison_y)))
        discrim_set = TensorDataset(discrim_x, discrim_y)
        discrim_loader = self.dataset.get_dataloader(
            mode='train', dataset=discrim_set, batch_size=self.dataset.batch_size)

        all_x = torch.cat((clean_x, poison_x))
        all_y = torch.cat((clean_y, poison_y))
        all_discrim_y = torch.cat((torch.zeros_like(clean_y),
                                   torch.ones_like(poison_y)))
        # used for training
        poison_set = TensorDataset(all_x, all_y, all_discrim_y)
        poison_loader = self.dataset.get_dataloader(mode='train', dataset=poison_set,
                                                    batch_size=self.dataset.batch_size)
        return poison_loader, discrim_loader

    @staticmethod
    def bypass_get_data(data: tuple[torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return to_tensor(data[0]), to_tensor(data[1], dtype='long'), to_tensor(data[2], dtype='long')

    def joint_train(self, epoch: int = 0, optimizer: optim.Optimizer = None, lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                    poison_loader=None, discrim_loader=None, save=False, **kwargs):
        in_dim = self.model._model.classifier[0].in_features
        D = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_dim, 256)),
            ('bn1', nn.BatchNorm1d(256)),
            ('relu1', nn.LeakyReLU()),
            ('fc2', nn.Linear(256, 128)),
            ('bn2', nn.BatchNorm1d(128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, 2))
        ]))
        if env['num_gpus']:
            D.cuda()
        optim_params: list[nn.Parameter] = []
        for param_group in optimizer.param_groups:
            optim_params.extend(param_group['params'])
        optimizer.zero_grad()

        best_acc = 0.0
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        for _epoch in range(epoch):
            self.discrim_train(epoch=100, D=D, discrim_loader=discrim_loader)

            self.model.train()
            self.model.activate_params(optim_params)
            for data in poison_loader:
                optimizer.zero_grad()
                _input, _label_f, _label_d = self.bypass_get_data(data)
                out_f = self.model(_input)
                loss_f = self.model.criterion(out_f, _label_f)
                out_d = D(self.model.get_final_fm(_input))
                loss_d = self.model.criterion(out_d, _label_d)

                loss = loss_f - self.lambd * loss_d
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler:
                lr_scheduler.step()
            self.model.activate_params([])
            self.model.eval()
            _, cur_acc = self.validate_fn(get_data_fn=self.bypass_get_data)
            if cur_acc >= best_acc:
                print('{purple}best result update!{reset}'.format(**ansi))
                print(f'Current Acc: {cur_acc:.3f}    Previous Best Acc: {best_acc:.3f}')
                best_acc = cur_acc
                if save:
                    self.save()
            print('-' * 50)

    def discrim_train(self, epoch: int, D: nn.Sequential, discrim_loader: torch.utils.data.DataLoader):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        d_optimizer = optim.Adam(D.parameters(), lr=self.discrim_lr)
        d_optimizer.zero_grad()
        for _epoch in range(epoch):
            losses.reset()
            top1.reset()
            self.model.activate_params(D.parameters())
            D.train()
            for data in discrim_loader:
                # train D
                _input, _label = self.model.get_data(data)
                out_f = self.model.get_final_fm(_input).detach()
                out_d = D(out_f)
                loss_d = self.model.criterion(out_d, _label)

                acc1 = self.model.accuracy(out_d, _label, topk=(1, ))[0]
                batch_size = int(_label.size(0))
                losses.update(loss_d.item(), batch_size)
                top1.update(acc1, batch_size)

                loss_d.backward()
                d_optimizer.step()
                d_optimizer.zero_grad()
            print(f'Discriminator - epoch {_epoch:4d} / {epoch:4d} | loss {losses.avg:.4f} | acc {top1.avg:.4f}')
            self.model.activate_params([])
            D.eval()
