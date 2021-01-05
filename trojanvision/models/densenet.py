#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models
from torchvision.models.densenet import model_urls
import re
from collections import OrderedDict


class _DenseNet(_ImageModel):

    def __init__(self, layer: int = 121, **kwargs):
        super().__init__(**kwargs)
        ModelClass: type[torchvision.models.DenseNet] = getattr(torchvision.models, 'densenet' + str(layer))
        _model = ModelClass(num_classes=self.num_classes)
        self.features = _model.features
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.classifier)  # nn.Linear(512 * block.expansion, num_classes)
        ]))


class DenseNet(ImageModel):

    def __init__(self, name: str = 'densenet', layer: int = 121,
                 model_class: type[_DenseNet] = _DenseNet, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        url = model_urls['densenet' + str(self.layer)]
        print('get official model weights from: ', url)
        _dict = model_zoo.load_url(url, **kwargs)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                _dict[new_key] = _dict[key]
                del _dict[key]
        _dict['classifier.fc.weight'] = _dict['classifier.weight']
        _dict['classifier.fc.bias'] = _dict['classifier.bias']
        del _dict['classifier.weight']
        del _dict['classifier.bias']
        return _dict


class _DenseNetcomp(_DenseNet):

    def __init__(self, layer: int = 121, **kwargs):
        super().__init__(layer=layer, **kwargs)
        conv = self.features.conv0
        self.features.conv0 = nn.Conv2d(3, conv.out_channels, kernel_size=3, padding=1, bias=False)


class DenseNetcomp(DenseNet):

    def __init__(self, name: str = 'densenetcomp', layer: int = 121,
                 model_class: type[_DenseNetcomp] = _DenseNetcomp, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
        keys_list: list[str] = list(_dict.keys())
        _dict[keys_list[0]] = self._model.features[0].weight
        _dict[keys_list[1]] = self._model.features[0].bias
        return _dict
