# -*- coding: utf-8 -*-

from .badnet import BadNet

import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from torch.utils.data import Dataset
from typing import Tuple


class Bypass_Embed(BadNet):
    name: str = 'bypass_embed'

    def __init__(self, poison_num=100, lambd: int = 10, discrim_lr: float = 0.001,
                 **kwargs):
        super().__init__(**kwargs)

        self.param_list['bypass_embed'] = ['poison_num', 'lambd', 'discrim_lr']

        self.poison_num: int = poison_num
        self.lambd = lambd
        self.discrim_lr = discrim_lr

    def attack(self, optimizer=None, **kwargs):
        print('Sample Data')
        trainloader = self.sample_data()  # with poisoned images
        print('Joint Training')
        self.joint_train(optimzier, trainloader)

    def sample_data(self):
        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        other_x, other_y = [], []
        for _class in other_classes:
            loader = self.dataset.get_dataloader(mode='train', batch_size=self.poison_num, classes=[_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
            _input, _label = next(iter(loader))
            other_x.append(_input)
            other_y.append(_label)
        other_x = torch.cat(other_x)
        other_y = torch.cat(other_y)

        poison_x = self.mark.add_mark(other_x)
        poison_y = self.target_class * torch.ones_like(other_y)

        trainset = self.dataset.get_dataset(mode='train')
        clean_x, clean_y = next(iter(self.dataset.get_dataloader(dataset=trainset, batch_size=len(trainset),
                                                                 shuffle=True, num_workers=0, pin_memory=False)))
        all_x = torch.cat((clean_x, poison_x))
        all_y = torch.cat((clean_y, poison_y))
        all_discrim_y = torch.cat((torch.zeros_like(clean_y),
                                   torch.ones_like(poison_y)))
        # used for training
        poison_trainset = TwoLabelsDataset(all_x, all_y, all_discrim_y)
        poison_trainloader = self.dataset.get_dataloader(
            mode='train', dataset=poison_trainset, batch_size=self.dataset.batch_size)

        return poison_trainloader

    @staticmethod
    def get_data(data: Tuple[torch.Tensor], **kwargs) -> (torch.Tensor, torch.LongTensor, torch.LongTensor):
        return to_tensor(data[0]), to_tensor(data[1], dtype='long'), to_tensor(data[2], dtype='long')

    def joint_train(self, epoch: int = 0, optimizer: optim.Optimizer = None, _lr_scheduler: optim.lr_scheduler._LRScheduler = None,
                    poison_trainloader=None, save=False, **kwargs):
        # get in_dim
        batch_x = poison_trainloader[0][0].unsqueeze(0)  # sample one batch
        batch_x = self.model.get_final_fm(batch_x)
        in_dim = batch_x.shape[-1]

        D = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_dim, 256)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.LeakyReLU()),
            ('fc2', nn.Linear(256, 128)),
            ('bn2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, 2)),
            ('softmax', nn.Softmax())
        ]))
        if env['num_gpus']:
            D.cuda()
        d_optimizer = optim.Adam(D.parameters(), lr=self.discrim_lr)

        params = [param_group['params'] for param_group in optimizer.param_groups]
        params.append(D.parameters())
        self.activate_params(params)

        d_optimizer.zero_grad()
        optimizer.zero_grad()

        for _epoch in range(47):
            for data in poison_trainloader:
                # train D
                _input, _label_f, _label_d = self.get_data(data)
                out_d = D(self.model.get_final_fm(_input))
                loss_d = self.model.criterion(out_d, _label_d)
                loss_d.backward()
                d_optimizer.step()
                d_optimizer.zero_grad()

        for _epoch in range(epoch):
            for inner_epoch in range(3):
                for data in poison_trainloader:
                    # train D
                    _input, _label_f, _label_d = self.get_data(data)
                    out_d = D(self.model.get_final_fm(_input))
                    loss_d = self.model.criterion(out_d, _label_d)
                    loss_d.backward()
                    d_optimizer.step()
                    d_optimizer.zero_grad()
            # output gan loss information
            for data in poison_trainloader:
                # train model
                _input, _label_f, _label_d = self.get_data(data)
                out_f = self.model(_input)
                loss_f = self.model.criterion(out_f, _label_f)
                out_d = D(self.model.get_final_fm(_input))
                loss_d = self.model.criterion(out_d, _label_d)

                loss = loss_f - self.lambd * loss_d
                loss.backward()
                optimizer.step()
                _lr_scheduler.step()
                optimizer.zero_grad()
            # validate and save
            self.validate_func()
            self.model.train()
            D.train()
        self.model.eval()
        D.eval()

# ---------------------------------------------------------------------------------- #


class TwoLabelsDataset(Dataset):
    def __init__(self, data: torch.FloatTensor, labels_1: torch.LongTensor, labels_2: torch.LongTensor):
        '''
        A customized Dataset class with two gruop of labels,
        used for bypass detection backdoor attack ('bypass_embed')
        '''
        self.data = data
        self.labels_1 = labels_1
        self.labels_2 = labels_2

    def __getitem__(self, index):
        x = self.data[index]
        y1 = self.labels_1[index]
        y2 = self.labels_2[index]
        return x, y1, y2

    def __len__(self):
        return len(self.data)
