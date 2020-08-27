from trojanzoo.attack import Attack
from trojanzoo.utils.mark import Watermark
from trojanzoo.utils import save_tensor_as_img

from typing import Union, List

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np

class Reflection_Backdoor(BadNet):
    name: str = 'reflection_backdoor'

    def __init__(self, reflect_num: int=20, selection_step: int=50, poison_num: int=1000, **kwargs):
        super().__init__(**kwargs)

        self.reflect_num: int = reflect_num
        self.selection_step: int = selection_step
        self.m = self.reflect_num//2
        self.poison_num = poison_num

        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], device='cpu')
        self.conv2d = nn.Conv2d(1, 1, 3, bias=False)
        self.conv2d.weight = nn.Parameter(kernel.view_as(self.conv2d.weight))

        loader = self.dataset.get_dataloader(mode='train', batch_size=self.reflect_num, classes=[self.target_class],
                                                 shuffle=True, num_workers=0, pin_memory=False)
        self.reflect_set, self.reflect_labels = next(iter(loader)) # _images, _labels = next(iter(loader))
        self.W = torch.ones(reflect_num)

        other_classes = list(range(self.dataset.num_classes))
        other_classes.pop(self.target_class)
        self.train_loader = self.dataset.get_dataloader(mode='train', batch_size=self.poison_num, classes=other_classes,
                                                        shuffle=True, num_workers=0, pin_memory=False)
        self.valid_loader = self.dataset.get_dataloader(mode='validate',batch_size=self.poison_num, classes=other_classes,
                                                        shuffle=True, num_workers=0, pin_memory=False)
    def attack(self, **kwargs):
        # indices
        pick_img_ind = np.random.choice(len(range(self.reflect_num)), self.m, replace=False).tolist()
        ref_images = self.reflect_set[pick_img_ind]
        ref_labels = self.reflect_labels[pick_img_ind]

        for _ in range(self.selection_step):
            # todo: select x samples from trainset
            for i in range(len(ref_images)):
                self.mark.mark = self.conv2d(ref_images[i])
                # todo: add trigger into trainset & validate set

                # todo: train model
                # todo: cal effectiveness in validate set
                self.W[pick_img_ind[i]] = None # todo: give attack success rate
            
            # update self.W
            other_img_ind = list(set(range(self.reflect_num)) - set(pick_img_ind))
            self.W[other_img_ind] = self.W.median()

            # re-pick top m reflection images
            pick_img_ind = torch.argsort(self.W).tolist()[:self.m]
            ref_images = self.reflect_set[pick_img_ind]
