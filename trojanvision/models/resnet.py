#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .imagemodel import _ImageModel, ImageModel
import trojanvision.utils.resnet_s

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.resnet import model_urls
from collections import OrderedDict
from collections.abc import Callable


class _ResNet(_ImageModel):

    def __init__(self, layer: int = 18, **kwargs):
        super().__init__(**kwargs)
        layer = int(layer)
        ModelClass: Callable[..., torchvision.models.ResNet] = getattr(torchvision.models, 'resnet' + str(layer))
        _model = ModelClass(num_classes=self.num_classes)
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

    def get_all_layer(self, x: torch.Tensor, layer_input='input') -> dict[str, torch.Tensor]:
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

    def get_layer_name(self) -> list[str]:
        layer_name_list = []
        for layer_name, layer in self.features.named_children():
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children():
                    if 'bn' not in block_name and 'relu' not in block_name:
                        layer_name_list.append('features.' + layer_name + '.' + block_name)
            elif 'bn' not in layer_name and 'relu' not in layer_name:
                layer_name_list.append('features.' + layer_name)
        layer_name_list.append('pool')
        layer_name_list.append('flatten')
        for name, _ in self.classifier.named_children():
            if 'relu' not in name and 'bn' not in name and 'dropout' not in name:
                layer_name_list.append('classifier.' + name)
        return layer_name_list


class ResNet(ImageModel):

    def __init__(self, name: str = 'resnet', layer: int = 18,
                 model_class: type[_ResNet] = _ResNet, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls['resnet' + str(self.layer)]
        print('get official model weights from: ', url)
        _dict: OrderedDict[str, torch.Tensor] = model_zoo.load_url(url, **kwargs)
        new_dict = OrderedDict()
        for i, (key, value) in enumerate(_dict.items()):
            prefix = 'features.' if i < len(_dict) - 2 else 'classifier.'
            new_dict[prefix + key] = value
        return new_dict


class _ResNetcomp(_ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv: nn.Conv2d = self.features.conv1
        self.features.conv1 = nn.Conv2d(conv.in_channels, conv.out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)


class ResNetcomp(ResNet):

    def __init__(self, name: str = 'resnetcomp', layer: int = 18,
                 model_class: type[_ResNetcomp] = _ResNetcomp, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        _dict[list(_dict.keys())[0]] = self._model.features[0].weight
        return _dict


class _ResNetS(_ResNet):
    def __init__(self, layer: int = 18, **kwargs):
        super().__init__(layer=layer, **kwargs)
        _model = trojanvision.utils.resnet_s.ResNetS(nclasses=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', nn.ReLU(inplace=True)),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4)
        ]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)  # nn.Linear(512 * block.expansion, num_classes)
        ]))


class ResNetS(ResNet):
    def __init__(self, name: str = 'resnets', layer: int = 18,
                 model_class: type[_ResNetS] = _ResNetS, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)
