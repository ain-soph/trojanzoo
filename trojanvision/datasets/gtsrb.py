#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imagefolder import ImageFolder
import torchvision.transforms as transforms
import numpy as np


class GTSRB(ImageFolder):

    name = 'gtsrb'
    n_dim = (32, 32)
    num_classes = 43
    valid_set = False
    url = {'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip'}
    org_folder_name = {'train': 'GTSRB/Training'}

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.3403, 0.3121, 0.3214],
                                                           'std': [0.2724, 0.2608, 0.2669], },
                 loss_weights: bool = True, **kwargs):
        return super().__init__(norm_par=norm_par, loss_weights=loss_weights, **kwargs)

    @staticmethod
    def get_transform(mode: str) -> transforms.Compose:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
        return transform

    def initialize_npz(self, mode_list: list[str] = ['train', 'valid'],
                       transform: transforms.Compose = transforms.Compose([transforms.Resize((32, 32)),
                                                                           transforms.Lambda(lambda x: np.array(x))]),
                       **kwargs):
        super().initialize_npz(mode_list=mode_list, transform=transform, **kwargs)
