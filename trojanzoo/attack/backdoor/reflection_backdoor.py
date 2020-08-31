from trojanzoo.attack import Attack
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils.data import MyDataset
from trojanzoo.utils import save_tensor_as_img

from typing import Union, List
from .badnet import BadNet
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import math
import copy
import random
import numpy as np

class Reflection_Backdoor(BadNet):
    name: str = 'reflection_backdoor'

    def __init__(self, reflect_num: int=10, selection_step: int=20, poison_num: int=500, **kwargs):
        super().__init__(**kwargs)

        self.reflect_num: int = reflect_num
        self.selection_step: int = selection_step
        self.m: int = self.reflect_num
        self.poison_num: int = poison_num

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

        loader = self.dataset.get_dataloader(mode='train', batch_size=self.reflect_num, classes=[self.target_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
        self.reflect_set, self.reflect_labels = next(iter(loader)) # _images, _labels = next(iter(loader))
        self.W = torch.zeros(reflect_num)

        self.trainset = self.dataset.get_dataset(mode='train')
        self.validset = self.dataset.get_dataset(mode='valid')
        self.train_subset, _ = self.dataset.split_set(self.trainset, length=self.poison_num)
        self.valid_subset, _ = self.dataset.split_set(self.validset, length=self.poison_num)
    
    def attack(self, epoch: int, save=False, **kwargs):
        # indices
        pick_img_ind = np.random.choice(len(range(self.reflect_num)), self.m, replace=False).tolist()
        ref_images = self.reflect_set[pick_img_ind]
        ref_labels = self.reflect_labels[pick_img_ind]

        for _ in range(self.selection_step):
            train_imgs, train_labels = next(iter(torch.utils.data.DataLoader(self.train_subset, batch_size=len(self.train_subset), num_workers=0)))
            valid_imgs, valid_labels = next(iter(torch.utils.data.DataLoader(self.valid_subset, batch_size=len(self.train_subset), num_workers=0)))
            train_labels.fill_(self.target_class)
            valid_labels.fill_(self.target_class)
            
            state_dict = copy.deepcopy(self.model.state_dict())
            for i in range(len(ref_images)):
                # locally change
                self.mark.mark = self.conv2d(ref_images[i].mean(0).unsqueeze(0).unsqueeze(0))
                posion_train_imgs = self.mark.add_mark(train_imgs).detach()
                posion_valid_imgs = self.mark.add_mark(valid_imgs).detach()

                poison_train_subset = MyDataset(posion_train_imgs, train_labels.int().tolist())
                poison_valid_subset = MyDataset(posion_valid_imgs, valid_labels.int().tolist())
                posion_trainset = ConcatDataset([self.trainset, poison_train_subset])
                posion_validset = ConcatDataset([self.validset, poison_valid_subset])

                poison_train_loader = self.dataset.get_dataloader(mode='train', dataset=posion_trainset)
                poison_valid_loader = self.dataset.get_dataloader(mode='valid', dataset=posion_validset)

                self.model._train(epoch, loader_train=poison_train_loader, loader_valid=poison_valid_loader,
                                  validate_func=self.validate_func, **kwargs)
                _, attack_acc, _ = self.model._validate(loader=poison_valid_loader, **kwargs)
                self.W[pick_img_ind[i]] = attack_acc
                
                # restore model
                self.model.load_state_dict(state_dict)

            # update self.W
            if self.m < self.reflect_num:
                other_img_ind = list(set(range(self.reflect_num)) - set(pick_img_ind))
                self.W[other_img_ind] = self.W.median()

            # re-pick top m reflection images
            pick_img_ind = torch.argsort(self.W).tolist()[:self.m]
            ref_images = self.reflect_set[pick_img_ind]

        best_mark_ind = torch.argsort(self.W).tolist()[0]
        self.mark.mark = self.conv2d(ref_images[i].mean(0).unsqueeze(0).unsqueeze(0))
        posion_train_imgs = self.mark.add_mark(train_imgs).detach()
        posion_valid_imgs = self.mark.add_mark(valid_imgs).detach()

        poison_train_subset = MyDataset(posion_train_imgs, train_labels.int().tolist())
        poison_valid_subset = MyDataset(posion_valid_imgs, valid_labels.int().tolist())
        posion_trainset = ConcatDataset([self.trainset, poison_train_subset])
        posion_validset = ConcatDataset([self.validset, poison_valid_subset])

        poison_train_loader = self.dataset.get_dataloader(mode='train', dataset=posion_trainset)
        poison_valid_loader = self.dataset.get_dataloader(mode='valid', dataset=posion_validset)

        # final training, see performance of best reflection trigger
        self.model._train(epoch, loader_train=poison_train_loader, loader_valid=poison_valid_loader,
                          validate_func=self.validate_func, save=save, save_fn=self.save, **kwargs)