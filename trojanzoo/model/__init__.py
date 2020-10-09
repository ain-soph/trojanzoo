# -*- coding: utf-8 -*-

from .model import Model
from .imagemodel import ImageModel
# from .graph import *
from .image import *

class_dict = {
    'net': 'Net',
    'alexnet': 'AlexNet',
    'resnet': 'ResNet',
    'resnetcomp': 'ResNetcomp',
    'vgg': 'VGG',
    'vggcomp': 'VGGcomp',
    'densenet': 'DenseNet',
    'densenetcomp': 'DenseNetcomp',
    'latentnet': 'LatentNet',
    'magnet': 'MagNet',
}
