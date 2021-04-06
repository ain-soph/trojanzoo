#!/usr/bin/env python3

from .imagemodel import _ImageModel, ImageModel
from trojanvision.datasets import ImageNet

import torch
import torch.nn as nn
import torch.hub
import numpy as np
import os
from collections import OrderedDict


class _BiT(_ImageModel):
    def __init__(self, name: str = 'bit-m-r50x1', **kwargs):
        super().__init__(**kwargs)
        name = name.upper().replace('BIT', 'BiT').replace('X', 'x')
        from trojanvision.utils.model_archs.bit import KNOWN_MODELS
        _model = KNOWN_MODELS[name](head_size=1)
        self.features = nn.Sequential()
        self.features.add_module('root', _model.root)
        self.features.add_module('body', _model.body)
        for name, module in _model.root.named_children():
            self.features.add_module(name=name, module=module)
        for name, module in _model.body.named_children():
            self.features.add_module(name=name, module=module)
        self.features.add_module('gn', module=getattr(_model.head, 'gn'))
        self.features.add_module('relu', module=getattr(_model.head, 'relu'))
        self.pool: nn.AdaptiveAvgPool2d = getattr(_model.head, 'avg')
        final_layer: nn.Conv2d = getattr(_model.head, 'conv')
        self.classifier = self.define_classifier(conv_dim=final_layer.in_channels,
                                                 num_classes=self.num_classes, fc_depth=1)


class BiT(ImageModel):

    def __init__(self, name: str = 'bit',
                 pretrained_dataset: str = 'm', layer: int = 50, width_factor: int = 1,
                 model: type[_BiT] = _BiT, norm_par: dict[str, list[float]] = None, **kwargs):
        name = self.parse_name(name, pretrained_dataset, layer, width_factor)
        if norm_par is None:
            norm_par = {'mean': [0.5, 0.5, 0.5],
                        'std': [0.5, 0.5, 0.5], }
        super().__init__(name=name, width_factor=width_factor,
                         model=model, norm_par=norm_par, **kwargs)

    @staticmethod
    def parse_name(name: str, pretrained_dataset: str = 'm', layer: int = 50, width_factor: int = 1) -> str:
        name_list = name.lower().split('-')
        assert name_list[0] == 'bit'
        if len(name_list) != 1:
            for element in name_list[1:]:
                if element[0] == 'r':
                    sub_list = element[1:].split('x')
                    layer = int(sub_list[0])
                    if len(sub_list) == 2:
                        width_factor = int(sub_list[1])
                else:
                    assert len(element) == 1
                    pretrained_dataset = element
        return '-'.join(['bit', pretrained_dataset, f'r{layer:d}x{width_factor:d}'])

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        # TODO: map_location argument
        file_name = self.name.upper().replace('BIT', 'BiT').replace('X', 'x')
        if isinstance(self.dataset, ImageNet):
            file_name += '-ILSVRC2012'
        file_name += '.npz'
        url = f'https://storage.googleapis.com/bit_models/{file_name}'
        print('get official model weights from: ', url)
        file_path = os.path.join(torch.hub.get_dir(), 'bit', file_name)
        if not os.path.exists(file_path):
            torch.hub.download_url_to_file(url, file_path)
        weights: dict[str, np.ndarray] = np.load(file_path)
        _dict = OrderedDict()
        from trojanvision.utils.model_archs.bit import tf2th
        _dict['features.conv.weight'] = tf2th(weights['resnet/root_block/standardized_conv2d/kernel'])
        for block_num in range(4):
            block_name = f'block{block_num+1:d}'
            block: nn.Sequential = getattr(self._model.features, block_name)
            for unit_name, unit in block.named_children():
                prefix = '/'.join(['resnet', block_name, unit_name, ''])
                dict_prefix = '.'.join(['features', block_name, unit_name, ''])
                _dict[dict_prefix + 'gn1.weight'] = tf2th(weights[prefix + 'a/group_norm/gamma'])
                _dict[dict_prefix + 'gn1.bias'] = tf2th(weights[prefix + 'a/group_norm/beta'])
                _dict[dict_prefix + 'conv1.weight'] = tf2th(weights[prefix + 'a/standardized_conv2d/kernel'])
                _dict[dict_prefix + 'gn2.weight'] = tf2th(weights[prefix + 'b/group_norm/gamma'])
                _dict[dict_prefix + 'gn2.bias'] = tf2th(weights[prefix + 'b/group_norm/beta'])
                _dict[dict_prefix + 'conv2.weight'] = tf2th(weights[prefix + 'b/standardized_conv2d/kernel'])
                _dict[dict_prefix + 'gn3.weight'] = tf2th(weights[prefix + 'c/group_norm/gamma'])
                _dict[dict_prefix + 'gn3.bias'] = tf2th(weights[prefix + 'c/group_norm/beta'])
                _dict[dict_prefix + 'conv3.weight'] = tf2th(weights[prefix + 'c/standardized_conv2d/kernel'])
                if hasattr(unit, 'downsample'):
                    weight = tf2th(weights[prefix + 'a/proj/standardized_conv2d/kernel'])
                    _dict[dict_prefix + 'downsample.weight'] = weight
        _dict['features.gn.weight'] = tf2th(weights['resnet/group_norm/gamma'])
        _dict['features.gn.bias'] = tf2th(weights['resnet/group_norm/beta'])
        _dict['classifier.fc.weight'] = tf2th(weights['resnet/head/conv2d/kernel']).flatten(1)
        _dict['classifier.fc.bias'] = tf2th(weights['resnet/head/conv2d/bias'])
        return _dict
