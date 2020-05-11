# -*- coding: utf-8 -*-
from .resnet import ResNet
from ..image_cnn import _Image_CNN, Image_CNN

from package.imports.universal import *
from collections import OrderedDict

import torchvision.models as models


class _ResNetNew(_Image_CNN):
    """docstring for ResNetNew"""

    def __init__(self, layer=18, **kwargs):
        super(_ResNetNew, self).__init__(**kwargs)
        _model = models.__dict__['resnet'+str(layer)](num_classes=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', _model.relu),  # nn.ReLU(inplace=True)
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4)
        ]))
        self.avgpool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.fc)  # nn.Linear(512 * block.expansion, num_classes)
        ]))
        # block.expansion = 1 if BasicBlock and 4 if Bottleneck
        # ResNet 18,34 use BasicBlock, 50 and higher use Bottleneck


class ResNetNew(ResNet):
    """docstring for ResNetNew"""

    def __init__(self, name='resnetnew', layer=None, model_class=_ResNetNew, default_layer=18, **kwargs):
        super(ResNetNew, self).__init__(
            name=name, layer=layer, model_class=model_class, default_layer=default_layer, **kwargs)
    def load_official_weights(self, output=True):
        pass