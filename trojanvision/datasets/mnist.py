#!/usr/bin/env python3

from .imageset import ImageSet

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Union


class MNIST(ImageSet):

    name: str = 'mnist'
    num_classes: int = 10
    data_shape = [1, 28, 28]

    def __init__(self, norm_par={'mean': [0.1307], 'std': [0.3081]}, **kwargs):
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        datasets.MNIST(root=self.folder_path, train=True, download=True)
        datasets.MNIST(root=self.folder_path, train=False, download=True)

    @staticmethod
    def get_transform(**kwargs) -> transforms.ToTensor:
        return transforms.ToTensor()

    def get_org_dataset(self, mode, transform: Union[str, object] = 'default', **kwargs):
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        assert mode in ['train', 'valid']
        return datasets.MNIST(root=self.folder_path, train=(mode == 'train'), transform=transform, **kwargs)
