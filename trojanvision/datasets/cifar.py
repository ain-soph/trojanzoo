#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imageset import ImageSet

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Union


class CIFAR10(ImageSet):

    name = 'cifar10'
    num_classes = 10
    n_dim = (32, 32)
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.4914, 0.4822, 0.4465],
                                                           'std': [0.2023, 0.1994, 0.2010], },
                 **kwargs):
        return super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.CIFAR10(root=self.folder_path, train=True, download=True)
        datasets.CIFAR10(root=self.folder_path, train=False, download=True)

    @staticmethod
    def get_transform(mode: str) -> Union[transforms.Compose, transforms.ToTensor]:
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        return transform

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs) -> datasets.CIFAR10:
        assert mode in ['train', 'valid']
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return datasets.CIFAR10(root=self.folder_path, train=(mode == 'train'), transform=transform, **kwargs)


class CIFAR100(CIFAR10):
    name = 'cifar100'
    num_classes = 100

    def initialize(self):
        datasets.CIFAR100(root=self.folder_path, train=True, download=True)
        datasets.CIFAR100(root=self.folder_path, train=False, download=True)

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs) -> datasets.CIFAR100:
        assert mode in ['train', 'valid']
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return datasets.CIFAR100(root=self.folder_path, train=(mode == 'train'), transform=transform, **kwargs)
