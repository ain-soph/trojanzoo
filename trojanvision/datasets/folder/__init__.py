#!/usr/bin/env python3

from trojanvision.datasets.imagefolder import ImageFolder

from .cub200 import CUB200, CUB200_2011
from .gtsrb import GTSRB
from .imagenet import ImageNet, Sample_ImageNet
from .isic import ISIC2018
from .vggface2 import VGGface2, Sample_VGGface2


__all__ = ['CUB200', 'CUB200_2011', 'GTSRB', 'ImageNet', 'ISIC2018',
           'VGGface2', 'Sample_ImageNet', 'Sample_VGGface2', ]

class_dict: dict[str, ImageFolder] = {
    'cub200': CUB200,
    'cub200_2011': CUB200_2011,
    'gtsrb': GTSRB,
    'imagenet': ImageNet,
    'isic2018': ISIC2018,
    'vggface2': VGGface2,
    'sample_imagenet': Sample_ImageNet,
    'sample_vggface2': Sample_VGGface2,
}
