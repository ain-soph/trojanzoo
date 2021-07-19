#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

import torchvision.datasets as datasets


class CIFAR10(ImageSet):

    name = 'cifar10'
    num_classes = 10
    data_shape = [3, 32, 32]
    class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [0.49139968, 0.48215827, 0.44653124],
                                                           'std': [0.24703233, 0.24348505, 0.26158768], },
                 **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.CIFAR10(root=self.folder_path, train=True, download=True)
        datasets.CIFAR10(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR10:
        assert mode in ['train', 'valid']
        return datasets.CIFAR10(root=self.folder_path, train=(mode == 'train'), **kwargs)


class CIFAR100(CIFAR10):
    name = 'cifar100'
    num_classes = 100

    def initialize(self):
        datasets.CIFAR100(root=self.folder_path, train=True, download=True)
        datasets.CIFAR100(root=self.folder_path, train=False, download=True)

    def _get_org_dataset(self, mode: str, **kwargs) -> datasets.CIFAR100:
        assert mode in ['train', 'valid']
        return datasets.CIFAR100(root=self.folder_path, train=(mode == 'train'), **kwargs)
