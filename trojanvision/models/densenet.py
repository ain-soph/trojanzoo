#!/usr/bin/env python3
from .imagemodel import _ImageModel, ImageModel

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.densenet import model_urls as urls
import re
from collections import OrderedDict


class _DenseNet(_ImageModel):

    def __init__(self, layer: int = 121, comp: bool = False, **kwargs):
        super().__init__(**kwargs)
        ModelClass: type[torchvision.models.DenseNet] = getattr(torchvision.models, f'densenet{layer:d}')
        _model = ModelClass(num_classes=self.num_classes)
        self.features = _model.features
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.classifier)  # nn.Linear(512 * block.expansion, num_classes)
        ]))
        if comp:
            conv0: nn.Conv2d = self.features.conv0
            conv0 = nn.Conv2d(conv0.in_channels, conv0.out_channels,
                              kernel_size=3, padding=1, bias=False)
            module_list = [('conv0', conv0)]
            module_list.extend(list(self.features.named_children())[4:])
            self.features = nn.Sequential(OrderedDict(module_list))


class DenseNet(ImageModel):
    model_urls = urls

    def __init__(self, name: str = 'densenet', layer: int = 121,
                 model: type[_DenseNet] = _DenseNet, **kwargs):
        comp = True if 'comp' in name else False
        super().__init__(name=name, layer=layer, model=model, comp=comp, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _dict = super().get_official_weights(**kwargs)
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
