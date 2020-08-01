# -*- coding: utf-8 -*-

from .dataset import Dataset
from .imageset import ImageSet
from .imagefolder import ImageFolder

from .image import *

class_dict = {
    'dataset': 'Dataset',
    'imageset': 'ImageSet',
    'graphset': 'GraphSet',

    'imagefolder': 'ImageFolder',

    'mnist': 'MNIST',

    'cifar10': 'CIFAR10',
    'cifar100': 'CIFAR100',

    'gtsrb': 'GTSRB',

    'imagenet': 'ImageNet',
    'sample_imagenet': 'Sample_ImageNet',

    'isic': 'ISIC',
    'isic2018': 'ISIC2018',
    'isic2019': 'ISIC2019',

    'vggface': 'VGGface',
    'vggface2': 'VGGface2',
    'sample_vggface2': 'Sample_VGGface2',
}
