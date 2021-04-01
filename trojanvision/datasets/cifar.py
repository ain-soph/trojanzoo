#!/usr/bin/env python3

# from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from .imageset import ImageSet

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
from typing import Union


class CIFAR10(ImageSet):

    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--cutout', action='store_true', help='use cutout')
        group.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.4914, 0.4822, 0.4465],
                                                           'std': [0.2023, 0.1994, 0.2010], },
                 cutout: bool = False, cutout_length: int = 16, **kwargs):
        self.cutout = cutout
        self.cutout_length = cutout_length
        super().__init__(norm_par=norm_par, **kwargs)
        if cutout:
            self.param_list['cifar10'] = ['cutout_length']

    def initialize(self):
        datasets.CIFAR10(root=self.folder_path, train=True, download=True)
        datasets.CIFAR10(root=self.folder_path, train=False, download=True)

    def get_transform(self, mode: str) -> Union[transforms.Compose, transforms.ToTensor]:
        if mode != 'train':
            return transforms.ToTensor()
        transform_list = [
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ]
        if self.cutout:
            # transforms.RandomErasing(value=self.norm_par['mean'])
            from trojanvision.utils.data import Cutout
            import torch
            fill_values = torch.tensor(self.norm_par['mean']).view(-1, 1, 1)
            transform_list.append(Cutout(self.cutout_length, fill_values=fill_values))
        return transforms.Compose(transform_list)

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
