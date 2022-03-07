#!/usr/bin/env python3
from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch
import torch.hub
import torch.nn as nn

import argparse
from collections import OrderedDict


class _ProxylessNAS(_ImageModel):

    def __init__(self, target_platform: str = 'proxyless_cifar', **kwargs):
        _model = torch.hub.load('ain-soph/ProxylessNAS', target_platform, pretrained=False)
        if 'num_features' not in kwargs.keys():
            classifier: nn.Module = getattr(_model, 'classifier')
            fc: nn.Linear = list(classifier.children())[0]
            kwargs['num_features'] = [fc.in_features]
        super().__init__(**kwargs)
        self.features = nn.Sequential()
        if _model.__class__.__name__ == 'ProxylessNASNets':
            self.features.add_module('first_conv', _model.first_conv)
            self.features.add_module('blocks', nn.Sequential(*_model.blocks))
            if hasattr(_model, 'feature_mix_layer'):
                self.features.add_module('feature_mix_layer', _model.feature_mix_layer)
        else:
            assert _model.__class__.__name__ == 'PyramidTreeNet', _model.__class__.__name__
            self.features.add_module('blocks', nn.Sequential(*_model.blocks[:-1]))


class ProxylessNAS(ImageModel):
    r"""ProxylessNAS proposed by Han Cai from MIT in ICLR 2019.

    :Available model names:

        .. code-block:: python3

            ['proxylessnas']

    See Also:
        * paper: `ProxylessNAS\: Direct Neural Architecture Search on Target Task and Hardware`_
        * code: https://github.com/MIT-HAN-LAB/ProxylessNAS

    Args:
        target_platform (str): Target platform to load using :any:`torch.hub.load`.
            Choose from ``['proxyless_cpu', 'proxyless_gpu', 'proxyless_mobile', 'proxyless_mobile_14', 'proxyless_cifar']``
            Defaults to ``'proxyless_cifar'``.

    .. _ProxylessNAS\: Direct Neural Architecture Search on Target Task and Hardware:
        https://arxiv.org/abs/1812.00332
    """  # noqa: E501
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
