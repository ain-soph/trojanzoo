#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.hub
import torch.nn as nn
from collections import OrderedDict


class _ProxylessNAS(_ImageModel):

    def __init__(self, target_platform: str = 'proxyless_cifar', **kwargs):
        super().__init__(**kwargs)
        _model = torch.hub.load('ain-soph/ProxylessNAS', target_platform)
        self.features = nn.Sequential(*_model.blocks)
        self.classifier = nn.Sequential(OrderedDict(list(_model.classifier.named_children())))


class ProxylessNAS(ImageModel):
    def __init__(self, name: str = 'proxylessnas', target_platform: str = 'proxyless_cifar',
                 model: type[_ProxylessNAS] = _ProxylessNAS, **kwargs):
        self.target_platform = target_platform
        super().__init__(name=name, model=model, target_platform=target_platform, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _model = torch.hub.load('ain-soph/ProxylessNAS', self.target_platform)
        features: nn.ModuleList = _model.blocks
        classifier: nn.Sequential = _model.classifier
        _dict = OrderedDict()
        _dict.update(features.state_dict(prefix='features.'))
        _dict.update(classifier.state_dict(prefix='classifier.'))
        return _dict
