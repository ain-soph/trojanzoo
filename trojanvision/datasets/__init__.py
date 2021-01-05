#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .imageset import ImageSet
from .mnist import MNIST
from .cifar import CIFAR10, CIFAR100
from .gtsrb import GTSRB
from .imagenet import ImageNet, Sample_ImageNet
from .vggface2 import VGGface2, Sample_VGGface2
from .isic import ISIC2018
from trojanvision.configs import Config, config
import trojanzoo.datasets

import argparse
from typing import Union

class_dict: dict[str, ImageSet] = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'gtsrb': GTSRB,
    'imagenet': ImageNet,
    'sample_imagenet': Sample_ImageNet,
    'isic2018': ISIC2018,
    'vggface2': VGGface2,
    'sample_vggface2': Sample_VGGface2,
}


def add_argument(parser: argparse.ArgumentParser, dataset_name: str = None, dataset: Union[str, ImageSet] = None,
                 config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict) -> argparse._ArgumentGroup:
    trojanzoo.datasets.add_argument(parser=parser, dataset_name=dataset_name, dataset=dataset,
                                    config=config, class_dict=class_dict)


def create(dataset_name: str = None, dataset: str = None,
           config: Config = config, class_dict: dict[str, type[ImageSet]] = class_dict, **kwargs) -> ImageSet:
    return trojanzoo.datasets.create(dataset_name=dataset_name, dataset=dataset,
                                     config=config, class_dict=class_dict, **kwargs)
