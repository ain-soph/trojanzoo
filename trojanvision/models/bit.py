#!/usr/bin/env python3

from .imagemodel import _ImageModel, ImageModel
from trojanvision.datasets import ImageNet
from trojanvision.utils.bit import KNOWN_MODELS, tf2th

import torch
import torch.nn as nn
import torch.hub
import numpy as np
import os
import re
from collections import OrderedDict


class _BiT(_ImageModel):
    def __init__(self, name: str = 'BiT-M-R50x1', **kwargs):
        super().__init__(**kwargs)
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

    def __init__(self, name: str = 'BiT', pretrained_dataset: str = 'M',
                 layer: int = 50, width_factor: int = 1,
                 model: type[_BiT] = _BiT, **kwargs):
        name, pretrained_dataset, layer = self.parse_name(name, pretrained_dataset, layer)
        if 'norm_par' not in kwargs.keys():
            kwargs['norm_par'] = {'mean': [0.5, 0.5, 0.5],
                                  'std': [0.5, 0.5, 0.5], }
        super().__init__(name=name, layer=layer, width_factor=width_factor,
                         model=model, **kwargs)

    @staticmethod
    def parse_name(name: str, pretrained_dataset: str = 'M', layer: int = 50) -> tuple[str, str, int]:
        if name[:3] == 'bit':
            name = 'BiT' + name[3:]
        full_name_list: list[str] = re.findall(r'[0-9]+|[A-Za-z]+|_', name)
        name_list = full_name_list[0].split('-')
        if len(name_list) == 1:
            name_list.append(pretrained_dataset)
            name_list.append('R')
        elif len(name_list) == 2:
            if name_list[1] == 'R':
                name_list.insert(1, pretrained_dataset)
            else:
                name_list.append('R')
        name_list[1] = name_list[1].upper()
        name_list[2] = name_list[2].upper()
        assert name_list[1] in ['S', 'M', 'L'] and name_list[2] == 'R', name
        pretrained_dataset = name_list[1]
        layer = int(name_list[3])
        full_name_list[0] = '-'.join(name_list)
        return ''.join(full_name_list), pretrained_dataset, layer

    def get_official_weights(self, **kwargs) -> OrderedDict[str, torch.Tensor]:
        # TODO: map_location argument
        # TODO: model save_dir defaults to torch.hub.get_dir()
        file_name = f'{self.name}.npz'
        if isinstance(self.dataset, ImageNet):
            file_name = f'{self.name}-ILSVRC2012.npz'
        url = f'https://storage.googleapis.com/bit_models/{file_name}'
        print('get official model weights from: ', url)
        file_path = os.path.join(torch.hub.get_dir(), 'bit', file_name)
        if not os.path.exists(file_path):
            torch.hub.download_url_to_file(url, file_path)
        weights: dict[str, np.ndarray] = np.load(file_path)
        _dict = OrderedDict()
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
