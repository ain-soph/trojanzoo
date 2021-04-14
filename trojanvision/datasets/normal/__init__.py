#!/usr/bin/env python3

from trojanvision.datasets.imageset import ImageSet

from .cifar import CIFAR10, CIFAR100
from .imagenet16 import ImageNet16
from .mnist import MNIST

__all__ = ['CIFAR10', 'CIFAR100', 'ImageNet16', 'MNIST']

class_dict: dict[str, ImageSet] = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'imagenet16': ImageNet16,
    'mnist': MNIST,
}
