# -*- coding: utf-8 -*-
from ..imageset import ImageSet
from trojanzoo.imports import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CIFAR10(ImageSet):

    def __init__(self, name='cifar10', n_dim=(32, 32), num_classes=10,
                 norm_par={'mean': [0.4914, 0.4822, 0.4465],
                           'std': [0.2023, 0.1994, 0.2010], },
                 **kwargs):
        super().__init__(name=name, n_dim=n_dim, num_classes=num_classes,
                         norm_par=norm_par, **kwargs)

    def initialize(self):
        trainset = datasets.CIFAR10(
            root=self.folder_path, train=True, download=True)
        validset = datasets.CIFAR10(
            root=self.folder_path, train=False, download=True)

    def get_transform(self, mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(self.n_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.ToTensor()
        return transform

    def get_full_dataset(self, mode, transform=None, _class=datasets.CIFAR10):
        if mode == 'test':
            raise ValueError(
                self.name+' only has \"train\" and \"valid\" originally.')
        return _class(root=self.folder_path, train=(mode == 'train'), transform=self.get_transform(mode))


class CIFAR100(CIFAR10):

    def __init__(self, name='cifar100', num_classes=100, **kwargs):
        super().__init__(name=name, num_classes=num_classes, **kwargs)

    def initialize(self):
        trainset = datasets.CIFAR100(
            root=self.folder_path, train=True, download=True)
        validset = datasets.CIFAR100(
            root=self.folder_path, train=False, download=True)

    def get_full_dataset(self, mode, transform=None, _class=datasets.CIFAR100):
        return super().get_full_dataset(mode, transform=transform, _class=_class)
