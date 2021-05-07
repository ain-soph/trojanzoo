#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet
from trojanvision.utils.datasets.imagenet16 import ImageNet16 as Dataset

import argparse
from typing import Union


class ImageNet16(ImageSet):

    name = 'imagenet16'
    num_classes = 1000
    data_shape = [3, 16, 16]

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--num_classes', type=int, help='number of classes')
        return group

    def __init__(self, norm_par: dict[str, list[float]] = {'mean': [122.68 / 255, 116.66 / 255, 104.01 / 255],
                                                           'std': [63.22 / 255, 61.26 / 255, 65.09 / 255], },
                 num_classes: int = None, **kwargs):
        self.num_classes = ImageNet16.num_classes if num_classes is None else num_classes
        super().__init__(norm_par=norm_par, **kwargs)

    def initialize(self):
        raise NotImplementedError('You need to download Google Folder "1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40" manually.')

    def get_org_dataset(self, mode: str, transform: Union[str, object] = 'default', **kwargs):
        assert mode in ['train', 'valid']
        if transform == 'default':
            transform = self.get_transform(mode=mode)
        return Dataset(root=self.folder_path, train=(mode == 'train'), transform=transform,
                       use_num_of_class_only=self.num_classes if self.num_classes < 1000 else None, **kwargs)
