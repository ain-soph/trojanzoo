#!/usr/bin/env python3

from trojanvision.models.imagemodel import ImageModel

from .alexnet import AlexNet
from .bit import BiT
from .densenet import DenseNet
from .dla import DLA
from .dpn import DPN
from .mobilenet import MobileNet
from .net import Net
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .vgg import VGG

__all__ = ['AlexNet', 'BiT', 'DenseNet', 'DLA', 'DPN',
           'MobileNet', 'Net', 'ResNet', 'ShuffleNetV2', 'VGG']

class_dict: dict[str, type[ImageModel]] = {
    'alexnet': AlexNet,
    'bit': BiT,
    'densenet': DenseNet,
    'dla': DLA,
    'dpn': DPN,
    'mobilenet': MobileNet,
    'net': Net,
    'resnet': ResNet,
    'resnext': ResNet,
    'shufflenetv2': ShuffleNetV2,
    'vgg': VGG,
}
