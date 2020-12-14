# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models as models
from torchvision.models.resnet import model_urls


class _ResNet(_ImageModel):

    def __init__(self, layer=18, **kwargs):
        super().__init__(**kwargs)
        layer = int(layer)
        _model: ResNet = models.__dict__[
            'resnet' + str(layer)](num_classes=self.num_classes)
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
        self.pool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.fc)  # nn.Linear(512 * block.expansion, num_classes)
        ]))
        # block.expansion = 1 if BasicBlock and 4 if Bottleneck
        # ResNet 18,34 use BasicBlock, 50 and higher use Bottleneck

    def get_all_layer(self, x: torch.Tensor, layer_input='input'):
        _dict = {}
        record = False

        if layer_input == 'input':
            x = self.preprocess(x)
            record = True

        for layer_name, layer in self.features.named_children():
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children():
                    if record:
                        x = block(x)
                        _dict['features.' + layer_name + '.' + block_name] = x
                    if 'features.' + layer_name + '.' + block_name == layer_input:
                        record = True
                if record:
                    _dict['features.' + layer_name] = x
            elif record:
                x = layer(x)
                _dict['features.' + layer_name] = x
            if 'features.' + layer_name == layer_input:
                record = True
        if layer_input == 'features':
            record = True
        if record:
            _dict['features'] = x
            x = self.pool(x)
            _dict['pool'] = x
            x = self.flatten(x)
            _dict['flatten'] = x

        for name, module in self.classifier.named_children():
            if record:
                x = module(x)
                _dict['classifier.' + name] = x
            elif 'classifier.' + name == layer_input:
                record = True
        _dict['classifier'] = x
        _dict['logits'] = x
        _dict['output'] = x
        return _dict

    def get_layer_name(self, extra=True):
        layer_name_list = []
        for layer_name, layer in self.features.named_children():
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children():
                    if 'bn' not in block_name and 'relu' not in block_name:
                        layer_name_list.append('features.' + layer_name + '.' + block_name)
            elif 'bn' not in layer_name and 'relu' not in layer_name:
                layer_name_list.append('features.' + layer_name)
        if extra:
            layer_name_list.append('pool')
            layer_name_list.append('flatten')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name_list.append('classifier.' + name)
        return layer_name_list


class ResNet(ImageModel):

    def __init__(self, name='resnet', layer=None, model_class=_ResNet, default_layer=18, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class,
                         default_layer=default_layer, **kwargs)

    def load_official_weights(self, verbose=True):
        url = model_urls['resnet' + str(self.layer)]
        _dict = model_zoo.load_url(url)
        self._model.features.load_state_dict(_dict, strict=False)
        if self.num_classes == 1000:
            self._model.classifier.load_state_dict(_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')


class _ResNetcomp(_ResNet):

    def __init__(self, layer=18, **kwargs):
        super().__init__(**kwargs)
        layer = int(layer)
        _model = models.__dict__[
            'resnet' + str(layer)](num_classes=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', _model.relu),  # nn.ReLU(inplace=True)
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4)
        ]))
        self.pool = _model.avgpool  # nn.AdaptiveAvgPool2d((1, 1))
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
        url = model_urls['resnet' + str(self.layer)]
        _dict = model_zoo.load_url(url)
        _dict = {key: value for (key, value)
                 in _dict.items() if key != 'conv1.weight'}
        self._model.features.load_state_dict(_dict, strict=False)
        if self.num_classes == 1000:
            self._model.classifier.load_state_dict(_dict, strict=False)
        if verbose:
            print(f'Model {self.name} loaded From Official Website: {url}')
