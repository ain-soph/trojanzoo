#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.hub
import torch.nn as nn

import argparse
from collections import OrderedDict


class _ProxylessNAS(_ImageModel):

    def __init__(self, target_platform: str = 'proxyless_cifar', **kwargs):
        super().__init__(**kwargs)
        _model = torch.hub.load('ain-soph/ProxylessNAS', target_platform, pretrained=False)
        self.features = nn.Sequential()
        if _model.__class__.__name__ == 'ProxylessNASNets':
            self.features.add_module('first_conv', _model.first_conv)
            self.features.add_module('blocks', nn.Sequential(*_model.blocks))
            if hasattr(_model, 'feature_mix_layer'):
                self.features.add_module('feature_mix_layer', _model.feature_mix_layer)
        else:
            assert _model.__class__.__name__ == 'PyramidTreeNet', _model.__class__.__name__
            self.features.add_module('blocks', nn.Sequential(*_model.blocks[:-1]))
        classifier: nn.Module = _model.classifier
        fc: nn.Linear = list(classifier.children())[0]
        self.classifier = self.define_classifier(conv_dim=fc.in_features,
                                                 num_classes=self.num_classes,
                                                 fc_depth=1)


class ProxylessNAS(ImageModel):
    available_models = ['proxylessnas']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--target_platform')
        return group

    def __init__(self, name: str = 'proxylessnas', target_platform: str = 'proxyless_cifar',
                 model: type[_ProxylessNAS] = _ProxylessNAS, **kwargs):
        self.target_platform = target_platform
        super().__init__(name=name, model=model, target_platform=target_platform, **kwargs)

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        _model: nn.Module = torch.hub.load('ain-soph/ProxylessNAS', self.target_platform, pretrained=True)
        _dict = OrderedDict()
        if hasattr(_model, 'first_conv'):
            first_conv: nn.Module = _model.first_conv
            _dict.update(first_conv.state_dict(prefix='features.first_conv.'))
        blocks: nn.ModuleList = _model.blocks
        if self.target_platform == 'proxyless_cifar':
            blocks = blocks[:-1]
        _dict.update(blocks.state_dict(prefix='features.blocks.'))
        if hasattr(_model, 'feature_mix_layer'):
            feature_mix_layer: nn.Module = _model.feature_mix_layer
            _dict.update(feature_mix_layer.state_dict(prefix='features.feature_mix_layer.'))
        fc: nn.Linear = list(_model.classifier.children())[0]
        _dict.update(fc.state_dict(prefix='classifier.fc.'))
        return _dict
