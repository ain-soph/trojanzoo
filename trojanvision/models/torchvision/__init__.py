#!/usr/bin/env python3

from trojanvision.models.imagemodel import ImageModel

from .alexnet import AlexNet
from .densenet import DenseNet
from .efficientnet import EfficientNet
from .mnasnet import MNASNet
from .mobilenet import MobileNet
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .vgg import VGG

__all__ = ['AlexNet', 'DenseNet', 'EfficientNet', 'MNASNet',
           'MobileNet', 'ResNet', 'ShuffleNetV2', 'VGG']

class_dict: dict[str, type[ImageModel]] = {
    'alexnet': AlexNet,
    'densenet': DenseNet,
    'efficientnet': EfficientNet,
    'mnasnet': MNASNet,
    'mobilenet': MobileNet,
    'resnet': ResNet,
    'resnext': ResNet,
    'shufflenetv2': ShuffleNetV2,
    'vgg': VGG,
}
