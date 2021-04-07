#!/usr/bin/env python3

from .imageset import ImageSet
from trojanvision.utils.datasets.imagenet16 import ImageNet16 as Dataset

import torchvision.transforms as transforms

import argparse
from typing import Union


class ImageNet16(ImageSet):

    name = 'imagenet16'
    num_classes = 1000
    data_shape = [3, 16, 16]

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--cutout', action='store_true', help='use cutout')
        group.add_argument('--cutout_length', type=int, default=8, help='cutout length')
        group.add_argument('--num_classes', type=int, help='number of classes')

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [122.68 / 255, 116.66 / 255, 104.01 / 255],
                                                           'std': [63.22 / 255, 61.26 / 255, 65.09 / 255], },
                 cutout: bool = False, cutout_length: int = 8, num_classes: int = None, **kwargs):
        self.cutout = cutout
        self.cutout_length = cutout_length
        self.num_classes = ImageNet16.num_classes if num_classes is None else num_classes
        super().__init__(norm_par=norm_par, **kwargs)
        if cutout:
            self.param_list['imagenet16'] = ['cutout_length']

    def initialize(self):
        raise NotImplementedError('You need to download Google Folder "1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40" manually.')

    def get_transform(self, mode: str) -> Union[transforms.Compose, transforms.ToTensor]:
        if mode != 'train':
            return transforms.ToTensor()
        transform_list = [
            transforms.RandomCrop((16, 16), padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if self.cutout:
            from trojanvision.utils.data import Cutout
            import torch
            fill_values = torch.tensor(self.norm_par['mean']).view(-1, 1, 1)
            transform_list.append(Cutout(self.cutout_length, fill_values=fill_values))
        return transforms.Compose(transform_list)

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs):
        assert mode in ['train', 'valid']
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return Dataset(root=self.folder_path, train=(mode == 'train'), transform=transform,
                       use_num_of_class_only=self.num_classes if self.num_classes < 1000 else None, **kwargs)
