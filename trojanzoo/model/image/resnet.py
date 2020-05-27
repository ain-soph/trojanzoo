# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models


class _ResNet(_ImageModel):

    def __init__(self, layer=18, **kwargs):
        super().__init__(**kwargs)
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

    def get_all_layer(self, x, layer_input='input'):
        od = OrderedDict()
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for l, block in self.features.named_children():
            if 'conv' in l:
                if record:
                    x = block(x)
                    od['features.'+l] = x
                elif 'features.'+l == layer_input:
                    record = True
            else:
                for name, module in block.named_children():
                    if record:
                        x = module(x)
                        od['features.'+l+'.'+name] = x
                    elif 'features.'+l+'.'+name == layer_input:
                        record = True
        if record:
            x = self.avgpool(x)
            od['avgpool'] = x
            x = x.flatten(start_dim=1)
            od['features'] = x
        elif layer_input == 'features':
            record = True

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                od['classifier.'+name] = x
            elif 'classifier.'+name == layer_input:
                record = True
        y = x
        od['classifier'] = y
        od['logits'] = y
        od['output'] = y
        return od

    def get_layer_name(self):
        layer_name = []
        for l, block in self.features.named_children():
            if 'conv' in l:
                layer_name.append('features.'+l)
            else:
                for name, _ in block.named_children():
                    if 'relu' not in name and 'bn' not in name:
                        layer_name.append('features.'+l+'.'+name)
        layer_name.append('avgpool')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name:
                layer_name.append('classifier.'+name)
        return layer_name


class ResNet(ImageModel):

    def __init__(self, name='resnet', layer=None, model_class=_ResNet, default_layer=18, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['resnet'+str(self.layer)]
        _dict = model_zoo.load_url(url)
        self._model.features.load_state_dict(_dict, strict=False)
        if self.num_classes == 1000:
            self._model.classifier.load_state_dict(_dict, strict=False)
        if verbose:
            print(
                'Model {name} loaded From Official Website: '.format(self.name), url)


class _ResNetcomp(_ResNet):

    def __init__(self, layer=18, **kwargs):
        super().__init__(**kwargs)
        _model = models.__dict__[
            'resnet'+str(layer)](num_classes=self.num_classes)
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


class ResNetcomp(ResNet):

    def __init__(self, name='resnetcomp', layer=None, model_class=_ResNetcomp, default_layer=18, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['resnet'+str(self.layer)]
        _dict = model_zoo.load_url(url)
        _dict = {key: value for (key, value)
                 in _dict.items() if key != 'conv1.weight'}
        self._model.features.load_state_dict(_dict, strict=False)
        if self.num_classes == 1000:
            self._model.classifier.load_state_dict(_dict, strict=False)
        if verbose:
            print(
                'Model {name} loaded From Official Website: '.format(self.name), url)
