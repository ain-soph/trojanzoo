# -*- coding: utf-8 -*-

from torch.utils import data
from .dataset import Dataset
from .imageset import ImageSet
from .imagefolder import ImageFolder
from .image import *

from trojanzoo.utils.config import Config
from trojanzoo.utils.output import ansi

import argparse
import sys
from typing import Type

class_dict = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'gtsrb': GTSRB,
    'imagenet': ImageNet,
    'sample_imagenet': Sample_ImageNet,
    'isic2018': ISIC2018,
    # 'isic2019': ISIC2019,
    # 'vggface': VGGface,
    'vggface2': VGGface2,
    'sample_vggface2': Sample_VGGface2,
}


def register(name: str, _class: type):
    class_dict[name] = _class


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None) -> argparse._ArgumentGroup:
    if dataset_name is None:
        dataset_name = get_dataset_name()
    DatasetType = Dataset
    if dataset_name is not None:
        DatasetType: Type[Dataset] = class_dict[dataset_name]
    group = parser.add_argument_group('{yellow}dataset{reset}'.format(**ansi))
    DatasetType.add_argument(group)
    return group


def create(dataset_name: str = None, **kwargs) -> Dataset:
    if dataset_name is None:
        dataset_name = Config.config['dataset']['default_dataset']
    result = Config.combine_param(config=Config.config['dataset'], dataset_name=dataset_name, **kwargs)
    DatasetType: Type[Dataset] = class_dict[dataset_name]
    return DatasetType(**result)


def get_dataset_name() -> str:
    argv = sys.argv
    try:
        idx = argv.index('--dataset')
        dataset_name: str = argv[idx + 1]
    except ValueError as e:
        try:
            idx = argv.index('-d')
            dataset_name: str = argv[idx + 1]
        except ValueError as e:
            return None
    return dataset_name
