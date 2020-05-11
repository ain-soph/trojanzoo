# -*- coding: utf-8 -*-
from ..image_cnn import _Image_CNN, Image_CNN
from package.imports.universal import *

from collections import OrderedDict

from torchvision.models.resnet import model_urls
import torchvision.models as models
from torch.utils import model_zoo


class _ResNet(_Image_CNN):
    """docstring for ResNet"""

    def __init__(self, layer=50, **kwargs):
        super(_ResNet, self).__init__(**kwargs)
        _model = models.__dict__[
            'resnet'+str(layer)](num_classes=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', _model.relu),  # nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ('maxpool', _model.maxpool),
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


class ResNet(Image_CNN):
    """docstring for ResNet"""

    def __init__(self, name='resnet', layer=None, model_class=_ResNet, default_layer=50, **kwargs):
        name, layer = ResNet.split_name(name, layer, default_layer=default_layer)
        name = name+str(layer)
        self.layer = layer
        super(ResNet, self).__init__(
            name=name, model_class=model_class, layer=layer, **kwargs)

    def load_official_weights(self, output=True):
        if output:
            print("********Load From Official Website!********")
        _dict = model_zoo.load_url(model_urls['resnet'+str(self.layer)])
        self._model.features.load_state_dict(_dict, strict=False)
        if self.num_classes == 1000:
            self._model.classifier.load_state_dict(_dict, strict=False)
