# -*- coding: utf-8 -*-

from .model import Model
from .imagemodel import ImageModel
# from .graph import *
from .image import *

class_dict = {
    'net': 'Net',
    'resnet': 'ResNet',
    'resnetcomp': 'ResNetcomp',
    'vgg': 'VGG',
    'vggcomp': 'VGGcomp',
    'densenet': 'DenseNet',
    'latentnet': 'LatentNet',
    'magnet': 'MagNet',
}
